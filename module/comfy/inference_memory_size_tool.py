import csv
import logging
import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime

import torch
from nodes import NODE_CLASS_MAPPINGS

_logger = logging.getLogger(__name__)


@dataclass
class InferenceMemorySizeCSVPoint:
    model_cls: str = ""
    batch_size: int = 0
    width: int = 0
    height: int = 0
    embedding_size: int = 0
    inference_memory_size: int = 0
    memory_history_snapshot: str = ""
    model_dtype: str = ""


def csv_dump(objects, filename):
    with open(filename, "w") as f:
        flds = [fld.name for fld in fields(objects[0])]
        w = csv.DictWriter(f, flds)
        w.writeheader()
        w.writerows([asdict(object) for object in objects])


def csv_load(object_cls, filename):
    with open(filename, "r") as f:
        results = csv.DictReader(f)
        return [object_cls(**result) for result in results]


def format_size(sz, pref_sz=None):
    if pref_sz is None:
        pref_sz = sz
    prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_sz < 768 * 1024:
            break
        prefix = new_prefix
        sz //= 1024
        pref_sz /= 1024
    return f"{sz:6d} {prefix}"


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
PROF_OUT_DIR = ""

profile_node_ids = ["KSampler", "SamplerCustom"]

for profile_node_id in profile_node_ids:
    profile_node_cls = NODE_CLASS_MAPPINGS[profile_node_id]
    profile_node_function_name = profile_node_cls.FUNCTION

    class ProfileProxy(profile_node_cls):
        FUNCTION = "memory_stats_proxy_exec"
        ORIGIN_FUNCTION = profile_node_function_name

        def profile_proxy_exec(self, *args, **kwargs):
            self.timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                origin_function = getattr(self, self.ORIGIN_FUNCTION)
                result = origin_function(*args, **kwargs)

            prof.export_memory_timeline(os.path.join(PROF_OUT_DIR, f"{self.timestamp}.json"), device="cuda:0")
            prof.mem_tl.export_memory_timeline_html(
                os.path.join(PROF_OUT_DIR, f"{self.timestamp}.html"), device_str="cuda:0"
            )
            prof.mem_tl.export_memory_timeline_raw(
                os.path.join(PROF_OUT_DIR, f"{self.timestamp}.raw.json"), device_str="cuda:0"
            )
            return result

        def record_memory_history_proxy_exec(self, *args, **kwargs):
            # self.timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

            torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

            origin_function = getattr(self, self.ORIGIN_FUNCTION)
            result = origin_function(*args, **kwargs)

            try:
                torch.cuda.memory._dump_snapshot(os.path.join(PROF_OUT_DIR, f"{self.timestamp}.pickle"))
            except Exception as e:
                _logger.error(f"Failed to capture memory snapshot {e}")

            # Stop recording memory snapshot history.
            torch.cuda.memory._record_memory_history(enabled=None)
            return result

        def memory_stats_proxy_exec(self, *args, **kwargs):
            self.timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            # When dynamic weights are not involved, we can use this simple method to determine how much memory we need.
            # When the weights are only transferred when needed,
            # we can run them once with a very small input when we don't store the weights at all on the device,
            # determine a minimum value z, and then simply add the curve under normal conditions

            prestats = torch.cuda.memory_stats()
            _logger.info(torch.cuda.memory_summary())

            prestats_alloc = prestats["requested_bytes.all.current"]

            torch.cuda.reset_peak_memory_stats()

            # origin_function = getattr(self, self.ORIGIN_FUNCTION)
            # result = origin_function(*args, **kwargs)
            result = self.record_memory_history_proxy_exec(*args, **kwargs)

            stats = torch.cuda.memory_stats()
            _logger.info(torch.cuda.memory_summary())

            stats_alloc_peak = stats["requested_bytes.all.peak"]

            inference_memory_size = stats_alloc_peak - prestats_alloc
            _logger.info(f"inference_memory_size : {inference_memory_size} ({format_size(inference_memory_size)})")
            # for SD
            # inference_memory_size = dtype_size * batch_size * width * height * X
            # TODO: find X and where embedding_size

            model = kwargs.get("model", None)
            latent_image = kwargs.get("latent_image", None)

            point = InferenceMemorySizeCSVPoint()
            point.inference_memory_size = inference_memory_size
            point.memory_history_snapshot = self.timestamp

            if latent_image is not None:
                B, C, H, W = latent_image["samples"].shape
                point.batch_size = B
                point.width = W
                point.height = H
            if model is not None:
                point.model_cls = model.model.model_config.__class__.__name__
                point.model_dtype = str(model.model_dtype())

            csv_path = os.path.join(PROF_OUT_DIR, "inference_memory_size.csv")
            if os.path.exists(csv_path):
                points = csv_load(InferenceMemorySizeCSVPoint, csv_path)
            else:
                points = []
            points.append(point)
            csv_dump(points, csv_path)

            return result

    NODE_CLASS_MAPPINGS[profile_node_id] = ProfileProxy
