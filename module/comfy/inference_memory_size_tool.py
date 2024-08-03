import csv
import logging
from dataclasses import asdict, dataclass, fields

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


def csv_dump(objects, filename):
    with open(filename, "w") as f:
        flds = [fld.name for fld in fields(objects[0])]
        w = csv.DictWriter(f, flds)
        w.writeheader()
        w.writerows([asdict(object) for object in objects])


def csv_load(object_cls, filename):
    with open(filename, "r") as f:
        flds = [fld.name for fld in fields(object_cls)]
        results = csv.DictReader(f, flds)
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


profile_node_ids = ["KSampler", "SamplerCustom"]

for profile_node_id in profile_node_ids:
    profile_node_cls = NODE_CLASS_MAPPINGS[profile_node_id]
    profile_node_function_name = profile_node_cls.FUNCTION

    class ProfileProxy(profile_node_cls):
        FUNCTION = "profile_proxy_exec"
        ORIGIN_FUNCTION = profile_node_function_name

        def profile_proxy_exec(self, *args, **kwargs):
            prestats = torch.cuda.memory_stats()
            _logger.info(torch.cuda.memory_summary())

            prestats_alloc = prestats["requested_bytes.all.current"]

            torch.cuda.reset_peak_memory_stats()

            origin_function = getattr(self, self.ORIGIN_FUNCTION)
            result = origin_function(*args, **kwargs)

            stats = torch.cuda.memory_stats()
            _logger.info(torch.cuda.memory_summary())

            stats_alloc_peak = stats["requested_bytes.all.peak"]

            inference_memory_size = stats_alloc_peak - prestats_alloc
            _logger.info(f"inference_memory_size : {inference_memory_size} ({format_size(inference_memory_size)})")
            # for SD
            # inference_memory_size = batch_size * width * height * X
            # TODO: find X and where embedding_size

            return result

    NODE_CLASS_MAPPINGS[profile_node_id] = ProfileProxy
