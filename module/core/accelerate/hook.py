import copy
import enum
import hashlib
import inspect  # Added for enum checking
import json
import logging
import os
import pathlib
import sys
import typing
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
from accelerate import dispatch_model
from accelerate.hooks import (
    AlignDevicesHook,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
    clear_device_cache,
    find_device,
    named_module_tensors,
    remove_hook_from_module,
)
from accelerate.utils import get_balanced_memory
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)  # For example
from torch.cuda import nvtx

# --- Global Module Logger ---
logger = logging.getLogger(__name__)

T = TypeVar("T")


# --- Utility Functions ---
def _parse_device_str(device_str: Union[str, int, torch.device]) -> torch.device:
    if isinstance(device_str, torch.device):
        return device_str
    if isinstance(device_str, int):
        return torch.device("cuda", device_str)

    s = str(device_str).lower()
    if s == "cpu" or s == "meta":
        return torch.device(s)
    if ":" in s:
        return torch.device(s)
    if s.isdigit():
        return torch.device("cuda", int(s))
    raise ValueError(f"Invalid device string: {device_str}")


# --- JSON Serialization/Deserialization Helpers ---
def _json_custom_default_encoder(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.device):
        return {"type": obj.type, "index": obj.index}
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, defaultdict):
        return dict(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)  # e.g., "torch.float32"
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable: {obj}")


def _reconstruct_from_data(data: Any, expected_type: Type[T]) -> T:
    if data is None:
        return None  # type: ignore

    origin_type = typing.get_origin(expected_type)
    args_type = typing.get_args(expected_type)

    if origin_type is Union:
        if type(None) in args_type and data is None:
            return None  # type: ignore

        non_none_types = [t for t in args_type if t is not type(None)]
        if len(non_none_types) == 1:
            return _reconstruct_from_data(data, non_none_types[0])

        for member_type in non_none_types:
            try:
                reconstructed = _reconstruct_from_data(data, member_type)
                if is_dataclass(member_type) and isinstance(reconstructed, member_type):
                    return reconstructed  # type: ignore
                if not is_dataclass(member_type) and isinstance(reconstructed, member_type):
                    return reconstructed  # type: ignore
                if reconstructed is data and isinstance(data, member_type):
                    return reconstructed  # type: ignore
            except (TypeError, ValueError, AttributeError):
                continue
        logger.warning(
            f"Could not definitively reconstruct Union type {expected_type} from data: {str(data)[:100]}. Returning raw data as fallback."
        )
        return data  # Fallback

    if is_dataclass(expected_type) and isinstance(data, dict):
        kwargs = {}
        try:
            # Provide global namespace of the module where expected_type is defined for get_type_hints
            module_globals = sys.modules[expected_type.__module__].__dict__
            field_type_hints = typing.get_type_hints(expected_type, globalns=module_globals)
        except Exception as e_th:
            logger.warning(
                f"Could not get precise type hints for dataclass {expected_type.__name__}: {e_th}. Falling back to basic field types."
            )
            field_type_hints = {f.name: f.type for f in fields(expected_type)}

        for f_obj in fields(expected_type):
            field_name = f_obj.name
            actual_field_type = field_type_hints.get(field_name, f_obj.type)  # Fallback to f_obj.type if not in hints
            if field_name in data:
                reconstructed_value = _reconstruct_from_data(data[field_name], actual_field_type)
                kwargs[field_name] = reconstructed_value
        try:
            return expected_type(**kwargs)  # type: ignore[return-value]
        except Exception as e_dc_init:
            logger.error(
                f"Failed to instantiate dataclass {expected_type.__name__} with kwargs from JSON: {kwargs}. Error: {e_dc_init}",
                exc_info=True,
            )
            raise TypeError(f"Dataclass {expected_type.__name__} instantiation failed.") from e_dc_init

    elif origin_type is list and isinstance(data, list):
        if not args_type:
            return data  # type: ignore Plain list
        element_type = args_type[0]
        return [_reconstruct_from_data(item, element_type) for item in data]  # type: ignore

    elif origin_type is dict and isinstance(data, dict):
        if not args_type or len(args_type) != 2:
            return data  # type: ignore Plain dict
        _key_type, value_type = args_type
        return {key: _reconstruct_from_data(val, value_type) for key, val in data.items()}  # type: ignore

    elif expected_type is torch.device and isinstance(data, dict) and "type" in data:
        return torch.device(data["type"], data.get("index"))

    elif expected_type is pathlib.Path and isinstance(data, str):
        return pathlib.Path(data)  # type: ignore

    elif inspect.isclass(expected_type) and issubclass(expected_type, enum.Enum):
        return expected_type(data)

    elif isinstance(data, str) and expected_type is torch.dtype:
        try:
            dtype_name = data.split(".")[-1]
            if not hasattr(torch, dtype_name):  # e.g. if "float16" but torch uses "float16"
                if dtype_name == "float16":
                    dtype_name = "half"  # common alternative name
                elif dtype_name == "float32":
                    dtype_name = "float"
                elif dtype_name == "float64":
                    dtype_name = "double"
                # Add more mappings if necessary
            return getattr(torch, dtype_name)
        except AttributeError:
            logger.error(f"Could not convert string '{data}' to torch.dtype.")
            raise TypeError(f"Cannot convert '{data}' to torch.dtype")

    if expected_type is Any or isinstance(data, expected_type):  # type: ignore
        return data

    try:  # Attempt direct coercion for simple types (e.g. "1" -> 1 if int expected)
        return expected_type(data)  # type: ignore
    except (TypeError, ValueError):
        logger.debug(
            f"Data type {type(data)} for value '{str(data)[:50]}' not instance of {expected_type} and direct coercion failed. Returning as is."
        )
        return data  # type: ignore


def save_to_json_file(data_object: Any, filepath: str):
    try:
        base_dir = os.path.dirname(filepath)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data_object, f, indent=4, default=_json_custom_default_encoder)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {e}", exc_info=True)
        raise


def load_from_json_file(filepath: str, expected_type: Type[T]) -> Optional[T]:
    if not os.path.exists(filepath):
        logger.debug(f"File not found for loading: {filepath}")
        return None
    try:
        with open(filepath, "r") as f:
            raw_data_from_json = json.load(f)
        reconstructed_instance = _reconstruct_from_data(raw_data_from_json, expected_type)
        logger.info(
            f"Data loaded and reconstructed from {filepath} as type {expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)}."
        )
        return reconstructed_instance
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error loading {filepath}: {e}", exc_info=True)
    except (TypeError, ValueError) as e:  # Errors from _reconstruct_from_data
        logger.error(f"Type reconstruction error for {expected_type} from {filepath}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to load/reconstruct {filepath} as {expected_type}: {e}", exc_info=True)
    return None


# --- Dataclasses for Profiling and Planning ---
@dataclass
class ModuleStats:
    exec_times: List[float] = field(default_factory=list)
    peak_vram_usages: List[int] = field(default_factory=list)
    weight_size: int = 0


@dataclass
class AverageModuleStats:
    avg_exec_time: float = 0.0
    max_peak_vram_delta: int = 0
    weight_size: int = 0

    def get_runtime_footprint(self) -> int:
        return self.weight_size + self.max_peak_vram_delta


@dataclass
class AverageProfilingStats:
    avg_module_stats: Dict[str, AverageModuleStats] = field(default_factory=dict)
    avg_move_times: Dict[str, float] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    module_vram_footprint: Dict[str, int] = field(default_factory=dict)


@dataclass
class PrefetchInstruction:
    module_to_prefetch: str
    target_device: torch.device
    trigger_module: str


@dataclass
class ProfilingData:
    module_stats: Dict[str, ModuleStats] = field(default_factory=dict)
    move_times: Dict[str, List[float]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    module_VRAM_footprint: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.module_stats = defaultdict(ModuleStats, self.module_stats or {})
        self.move_times = defaultdict(list, self.move_times or {})

    def record_execution(self, name: str, exec_time: Optional[float], peak_vram_delta: Optional[int]):
        if name not in self.execution_order:
            self.execution_order.append(name)
        stats_entry = self.module_stats[name]
        if exec_time is not None:
            stats_entry.exec_times.append(exec_time)
        if peak_vram_delta is not None:
            stats_entry.peak_vram_usages.append(peak_vram_delta)

    def record_weight_size(self, name: str, size: int):
        stats_entry = self.module_stats[name]
        if stats_entry.weight_size == 0 and size > 0:
            stats_entry.weight_size = size
        if name not in self.execution_order:
            self.execution_order.append(name)

    def record_move_time(self, src_dev: torch.device, tgt_dev: torch.device, size: int, move_time: float):
        key_str = str((str(src_dev), str(tgt_dev), size))
        self.move_times[key_str].append(move_time)

    def calculate_footprints(self):
        self.module_VRAM_footprint = {}
        for name, stats_data in self.module_stats.items():
            peak_vram_delta = max(stats_data.peak_vram_usages) if stats_data.peak_vram_usages else 0
            self.module_VRAM_footprint[name] = stats_data.weight_size + peak_vram_delta

    def get_avg_stats(self) -> AverageProfilingStats:
        avg_stats_map = {
            name: AverageModuleStats(
                avg_exec_time=sum(data.exec_times) / len(data.exec_times) if data.exec_times else 0.0,
                max_peak_vram_delta=max(data.peak_vram_usages) if data.peak_vram_usages else 0,
                weight_size=data.weight_size,
            )
            for name, data in self.module_stats.items()
        }
        avg_move_times_map = {k: sum(v) / len(v) if v else 0.0 for k, v in self.move_times.items()}
        if not self.module_VRAM_footprint and self.module_stats:
            self.calculate_footprints()
        return AverageProfilingStats(
            avg_stats_map, avg_move_times_map, list(self.execution_order), dict(self.module_VRAM_footprint)
        )

    def save(self, filepath: str):
        self.calculate_footprints()
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> Optional["ProfilingData"]:
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None


@dataclass
class OptimizationPlan:
    optimized_device_map: Dict[str, torch.device] = field(default_factory=dict)
    prefetch_schedule: List[PrefetchInstruction] = field(default_factory=list)
    trigger_index: Dict[str, List[PrefetchInstruction]] = field(
        default_factory=lambda: defaultdict(list), repr=False, compare=False
    )

    def __post_init__(self):
        self._build_trigger_index()

    def _build_trigger_index(self):
        self.trigger_index = defaultdict(list)
        for instr in self.prefetch_schedule:
            self.trigger_index[instr.trigger_module].append(instr)

    def save(self, filepath: str):
        save_to_json_file(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> Optional["OptimizationPlan"]:
        instance = load_from_json_file(filepath, cls)
        if isinstance(instance, cls):
            instance.__post_init__()
            return instance
        return None


# --- Profiler Internals ---
_current_profiling_data_global: Optional[ProfilingData] = None
_profiling_enabled_global: bool = False


@contextmanager
def _profile_run_context(data_store: ProfilingData):
    global _profiling_enabled_global, _current_profiling_data_global
    if _profiling_enabled_global:
        logger.warning("Profiling already enabled (re-entrant call).")
    _profiling_enabled_global, _current_profiling_data_global = True, data_store
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except Exception as e:
                logger.warning(f"Could not reset peak memory stats for device {i}: {e}")
    logger.info("Profiling run context entered.")
    try:
        yield
    finally:
        _profiling_enabled_global, _current_profiling_data_global = False, None  # Reset to initial state
        logger.info("Profiling run context exited.")


class ProfilerHook(ModelHook):
    def __init__(self, module_name: str):
        super().__init__()
        self.module_timing_events: Dict[int, Tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.module_start_vram_max: Dict[int, int] = {}
        self.module_name = module_name
        self.module: Optional[nn.Module] = None

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        if not (_profiling_enabled_global and _current_profiling_data_global):
            return args, kwargs
        name, module_id = self.module_name, id(module)
        try:
            if name and _current_profiling_data_global.module_stats[name].weight_size == 0:
                size = get_module_size(module, include_children=False)
                if size > 0:
                    _current_profiling_data_global.record_weight_size(name, size)
                elif name not in _current_profiling_data_global.execution_order:
                    _current_profiling_data_global.record_execution(name, None, None)
            dev = find_device(module.state_dict())
            if dev and dev.type == "cuda":
                try:
                    self.module_start_vram_max[module_id] = torch.cuda.max_memory_allocated(dev)
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    self.module_timing_events[module_id] = (s, e)
                except Exception as e_cuda:
                    logger.warning(f"ProfilerHook: VRAM/event pre_fwd fail {name} on {dev}: {e_cuda}")
                    if module_id in self.module_timing_events:
                        del self.module_timing_events[module_id]
                    if module_id in self.module_start_vram_max:
                        del self.module_start_vram_max[module_id]
        except Exception as e:
            logger.error(f"ProfilerHook: pre_fwd error [{name}]: {e}", exc_info=True)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        if not (_profiling_enabled_global and _current_profiling_data_global):
            return output
        name, module_id = self.module_name, id(module)
        if name and name not in _current_profiling_data_global.execution_order:
            _current_profiling_data_global.record_execution(name, None, None)
        if module_id not in self.module_timing_events and module_id not in self.module_start_vram_max:
            return output

        time_ms, vram_delta = None, None
        dev = find_device(output if output is not None else module.state_dict())
        if dev and dev.type == "cuda":
            try:
                if module_id in self.module_timing_events:
                    s, e = self.module_timing_events[module_id]
                    e.record()
                    torch.cuda.synchronize(dev)
                    time_ms = s.elapsed_time(e)
                if module_id in self.module_start_vram_max:
                    vram_before = self.module_start_vram_max[module_id]
                    vram_after = torch.cuda.max_memory_allocated(dev)
                    vram_delta = max(0, vram_after - vram_before)
            except Exception as e_post:
                logger.error(f"ProfilerHook: post_fwd CUDA error [{name} on {dev}]: {e_post}", exc_info=True)
            finally:
                if module_id in self.module_timing_events:
                    del self.module_timing_events[module_id]
                if module_id in self.module_start_vram_max:
                    del self.module_start_vram_max[module_id]
        if name:
            _current_profiling_data_global.record_execution(name, time_ms / 1000.0 if time_ms else None, vram_delta)
        else:
            logger.warning(f"ProfilerHook: Skipping recording, missing name for module ID {module_id}.")
        return output


class HeuristicOptimizer:
    def __init__(self, profiling_data: ProfilingData, max_memory_bytes: Dict[str, int]):
        self.profiling_data = profiling_data
        self.max_memory_bytes = {str(k): int(v) for k, v in max_memory_bytes.items()}
        avg_stats_obj = self.profiling_data.get_avg_stats()
        self.avg_stats = avg_stats_obj.avg_module_stats
        self.avg_move_times = avg_stats_obj.avg_move_times
        self.execution_order = avg_stats_obj.execution_order
        self.bandwidth_cache: Dict[Tuple[str, str], float] = {}
        self._init_bandwidth_info()

    def _init_bandwidth_info(self):
        gpus = sorted([k for k in self.max_memory_bytes if k != "cpu"])
        devs = ["cpu"] + gpus
        for s in devs:
            for t in devs:
                if s == t:
                    continue
                self.bandwidth_cache[(s, t)] = 5 if "cpu" in [s, t] else 50  # GB/s
        logger.debug(f"Initialized bandwidth cache: {self.bandwidth_cache}")

    def _estimate_move_time(self, size_bytes: int, src: str, tgt: str) -> float:
        if src == tgt or size_bytes == 0:
            return 0.0
        key = str((src, tgt, size_bytes))
        if (
            key in self.avg_move_times
            and self.avg_move_times[key] > 0
            and len(self.profiling_data.move_times.get(key, [])) >= 1
        ):
            logger.debug(f"Using profiled move time for {key}: {self.avg_move_times[key]:.6f}s")
            return self.avg_move_times[key]
        bw = self.bandwidth_cache.get((src, tgt)) or self.bandwidth_cache.get((tgt, src)) or 10
        est = max(size_bytes / (1024**3) / bw + 0.0001, 0.00001)
        logger.debug(f"Estimated move {size_bytes / (1024**2):.2f}MB {src}->{tgt} (BW {bw}GB/s): {est:.6f}s")
        return est

    def optimize(self) -> OptimizationPlan:
        logger.info("Starting heuristic optimization (prefetch-focused)...")
        opt_map_str: Dict[str, str] = {}
        prefetch_sched: List[PrefetchInstruction] = []
        cpu, gpus = (
            "cpu",
            sorted(
                [k for k in self.max_memory_bytes if k != "cpu" and self.max_memory_bytes[k] > 0],
                key=lambda x: self.max_memory_bytes[x],
                reverse=True,
            ),
        )
        if not gpus:
            logger.warning("No GPUs available/configured. All modules on CPU.")
            return OptimizationPlan({name: torch.device(cpu) for name in self.execution_order}, [])

        gpu_load: Dict[str, int] = defaultdict(int)
        benefit_ratio = 1.0
        cursor, exec_len = 0, len(self.execution_order)

        while cursor < exec_len:
            if self.execution_order[cursor] in opt_map_str:
                cursor += 1
                continue
            accum_time, window = 0.0, []
            last_processed_idx = cursor

            for pf_cand_idx in range(cursor, exec_len):
                last_processed_idx = pf_cand_idx
                pf_mod_name = self.execution_order[pf_cand_idx]
                if pf_mod_name in opt_map_str:
                    break  # Window ends if module already placed

                pf_stats = self.avg_stats.get(pf_mod_name)
                if not pf_stats:
                    logger.warning(f"Stats not found for {pf_mod_name}, skipping.")
                    if pf_cand_idx == cursor:
                        opt_map_str[pf_mod_name] = cpu
                        break  # Place current on CPU
                    continue  # Skip as prefetch candidate

                tgt_gpu_pf = next(
                    (
                        gpu_id
                        for gpu_id in gpus
                        if gpu_load[gpu_id] + pf_stats.get_runtime_footprint() <= self.max_memory_bytes[gpu_id]
                    ),
                    None,
                )

                if not tgt_gpu_pf:  # No GPU can fit this module for execution
                    if pf_cand_idx == cursor:
                        opt_map_str[pf_mod_name] = cpu
                        break  # Must place on CPU
                    else:
                        window.append(pf_mod_name)
                        accum_time += pf_stats.avg_exec_time
                        continue  # Add to window, try next

                move_time = self._estimate_move_time(pf_stats.weight_size, cpu, tgt_gpu_pf)
                if accum_time * benefit_ratio > move_time and window and move_time > 0.003:  # Prefetch viable
                    logger.info(
                        f"Prefetch {pf_mod_name} to {tgt_gpu_pf}. Hide: {accum_time:.4f}s, Move: {move_time:.4f}s. Window: {window}"
                    )
                    for mod in window:
                        opt_map_str[mod] = tgt_gpu_pf  # Place window mods on target GPU
                    opt_map_str[pf_mod_name] = cpu  # Prefetched mod initially on CPU
                    prefetch_sched.append(PrefetchInstruction(pf_mod_name, _parse_device_str(tgt_gpu_pf), window[0]))
                    gpu_load[tgt_gpu_pf] += pf_stats.weight_size  # Add weight to static load
                    cursor = pf_cand_idx + 1
                    window = []
                    accum_time = 0.0  # Advance main cursor, reset window
                    break  # From inner prefetch candidate loop
                else:  # Prefetch not viable yet or first item
                    window.append(pf_mod_name)
                    accum_time += pf_stats.avg_exec_time
            else:  # Inner loop exhausted without break
                cursor = last_processed_idx + 1

            for mod_name in window:  # Place any remaining modules in the last window
                if mod_name in opt_map_str:
                    continue
                stats = self.avg_stats.get(mod_name)
                if not stats:
                    opt_map_str[mod_name] = cpu
                    continue
                tgt_gpu = next(
                    (
                        gid
                        for gid in gpus
                        if gpu_load[gid] + stats.get_runtime_footprint() <= self.max_memory_bytes[gid]
                    ),
                    None,
                )
                if tgt_gpu:
                    opt_map_str[mod_name] = tgt_gpu
                    gpu_load[tgt_gpu] += stats.weight_size
                else:
                    opt_map_str[mod_name] = cpu

        for name in self.execution_order:  # Final check
            if name not in opt_map_str:
                logger.warning(f"Module '{name}' missed. Fallback to CPU.")
                opt_map_str[name] = cpu

        final_map = {name: _parse_device_str(dev_str) for name, dev_str in opt_map_str.items()}
        logger.info(
            f"Heuristic optimization complete. Map: {len(final_map)} entries. Prefetch: {len(prefetch_sched)} instr."
        )
        return OptimizationPlan(final_map, prefetch_sched)


class PrefetchContext:
    def __init__(self, plan: OptimizationPlan, model: nn.Module, num_streams: int, offload_policy: str):
        self.plan, self.model = plan, model
        self.module_map: Dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}
        self.num_streams, self.offload_policy = num_streams, offload_policy
        self.stream_mgr = self._init_stream_mgr()
        self.module_pf_streams: Dict[str, torch.cuda.Stream] = {}  # module_name -> stream for its prefetch

    def _init_stream_mgr(self) -> Dict[str, Any]:
        streams, s_idx = defaultdict(list), defaultdict(int)
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    streams[i] = [torch.cuda.Stream(device=i) for _ in range(self.num_streams)]
            except Exception as e:
                logger.warning(f"Error initializing CUDA prefetch streams: {e}")
        return {"streams": streams, "stream_idx": s_idx}

    def get_stream(self, device: torch.device) -> Optional[torch.cuda.Stream]:
        if not (
            device.type == "cuda"
            and torch.cuda.is_available()
            and device.index is not None
            and device.index < torch.cuda.device_count()
        ):
            return None
        pool = self.stream_mgr["streams"].get(device.index, [])
        if not pool:
            return None
        idx = self.stream_mgr["stream_idx"][device.index]
        stream = pool[idx % len(pool)]
        self.stream_mgr["stream_idx"][device.index] = (idx + 1) % len(pool)
        return stream

    def set_module_prefetch_stream(self, name: str, stream: torch.cuda.Stream):
        self.module_pf_streams[name] = stream

    def get_module_prefetch_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        return self.module_pf_streams.get(name)

    def clear_all_module_prefetch_streams(self):
        self.module_pf_streams.clear()


class PrefetchingWaitHook(ModelHook):
    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module, exec_dev: torch.device):
        super().__init__()
        self.ctx, self.name, self.mod_inst, self.exec_dev = ctx, name, mod_inst, exec_dev
        self.tied_ptrs_to_rm: Set[Tuple[int, torch.device]] = set()
        self.pf_submod_hf_hook: Optional[AlignDevicesHook] = None
        logger.debug(f"PrefetchingWaitHook for {self.name} on {self.exec_dev}")

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        if module is not self.mod_inst:
            logger.warning(f"WaitHook for {self.name} called on wrong mod")
            return args, kwargs
        pf_stream = self.ctx.get_module_prefetch_stream(self.name)
        if pf_stream:
            comp_stream = torch.cuda.current_stream(self.exec_dev)
            logger.debug(
                f"Mod {self.name} on dev {self.exec_dev} (stream {comp_stream.stream_id}) waiting for pf stream {pf_stream.stream_id}"
            )
            comp_stream.wait_stream(pf_stream)
            if self.pf_submod_hf_hook:
                self.pf_submod_hf_hook.tied_pointers_to_remove = self.tied_ptrs_to_rm
        return args, kwargs


class PrefetchingHook(ModelHook):  # Placed on trigger module
    def __init__(self, ctx: PrefetchContext, name: str, mod_inst: nn.Module):
        super().__init__()
        self.ctx, self.name, self.mod_inst = ctx, name, mod_inst
        logger.debug(f"PrefetchingHook (trigger) for {self.name}")

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        if module is not self.mod_inst:
            logger.warning(f"TriggerHook for {self.name} called on wrong mod")
            return args, kwargs
        logger.debug(f"TriggerHook pre_fwd for {self.name} (id {id(module)})")
        nvtx.range_push(f"pf_trigger_{self.name}")
        trigger_dev = find_device(module.state_dict())

        for instr in self.ctx.plan.trigger_index.get(self.name, []):
            pf_mod_name, pf_tgt_dev = instr.module_to_prefetch, instr.target_device
            mod_to_pf = self.ctx.module_map.get(pf_mod_name)
            if not mod_to_pf:
                logger.warning(f"Mod '{pf_mod_name}' for prefetch not found")
                continue
            pf_stream = self.ctx.get_stream(pf_tgt_dev)
            if not pf_stream:
                logger.warning(f"No pf stream for {pf_tgt_dev}. Cannot prefetch {pf_mod_name}.")
                continue

            if trigger_dev and trigger_dev.type == "cuda" and pf_tgt_dev.type == "cuda":
                pf_stream.wait_stream(torch.cuda.current_stream(trigger_dev))
            self.do_prefetch(pf_mod_name, pf_tgt_dev, mod_to_pf, pf_stream)
        nvtx.range_pop()
        return args, kwargs

    def do_prefetch(self, pf_name: str, pf_dev: torch.device, pf_mod: nn.Module, pf_stream: torch.cuda.Stream):
        nvtx.range_push(f"pf_task_{pf_name}_on_{pf_stream.stream_id}")
        try:
            pf_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(pf_stream):
                hook = getattr(pf_mod, "_hf_hook", None)
                align_hook, wait_hook = None, None
                if isinstance(hook, SequentialHook):
                    for h_ in hook.hooks:
                        if isinstance(h_, AlignDevicesHook):
                            align_hook = h_
                        elif isinstance(h_, PrefetchingWaitHook):
                            wait_hook = h_
                elif isinstance(hook, AlignDevicesHook):
                    align_hook = hook

                if not align_hook:
                    logger.error(f"No AlignHook on {pf_name}. Abort prefetch.")
                    nvtx.range_pop()
                    return
                if not wait_hook and self.ctx.plan.prefetch_schedule:
                    logger.error(f"No WaitHook on {pf_name} with active prefetch. Abort.")
                    nvtx.range_pop()
                    return
                if wait_hook:
                    wait_hook.tied_ptrs_to_rm.clear()
                    wait_hook.pf_submod_hf_hook = align_hook
                    wait_hook.exec_dev = pf_dev

                w_map, tied_map = getattr(align_hook, "weights_map", None), getattr(align_hook, "tied_params_map", None)
                if w_map is None or tied_map is None:
                    logger.error(f"AlignHook on {pf_name} not init. No prefetch.")
                    nvtx.range_pop()
                    return
                align_hook.execution_device = pf_dev

                for name, val in named_module_tensors(pf_mod, True, False, True):
                    if val.device.type == "meta":
                        map_val = w_map.get(name)
                        if map_val is None:
                            logger.warning(f"Meta tensor '{name}' in '{pf_name}' no value in weights_map.")
                            continue
                        if map_val.device.type == "cpu" and not map_val.is_pinned():
                            try:
                                map_val = map_val.pin_memory()
                                w_map.dataset.state_dict[w_map.prefix + name] = map_val
                            except RuntimeError as e_pin:
                                logger.warning(f"Could not pin {name} for prefetch: {e_pin}. Using unpinned.")

                        if map_val.data_ptr() not in tied_map:
                            tied_map[map_val.data_ptr()] = {}

                        tgt_tensor = torch.empty_like(map_val, device=pf_dev)
                        tgt_tensor.copy_(map_val, non_blocking=True)
                        param_tgt = torch.nn.Parameter(tgt_tensor, requires_grad=map_val.requires_grad)
                        tied_map[map_val.data_ptr()][pf_dev] = param_tgt
                        if wait_hook:
                            wait_hook.tied_ptrs_to_rm.add((map_val.data_ptr(), pf_dev))
                    elif val.device != pf_dev:
                        logger.debug(f"Prefetching existing tensor {name} from {val.device} to {pf_dev}")
                        val.data = val.data.to(pf_dev, non_blocking=True)
                logger.info(f"Prefetch task for {pf_name} to {pf_dev} submitted on stream {pf_stream.stream_id}.")
            self.ctx.set_module_prefetch_stream(pf_name, pf_stream)
        except Exception as e_pf:
            logger.error(f"Error in do_prefetch for {pf_name}: {e_pf}", exc_info=True)
        finally:
            nvtx.range_pop()


def get_module_size(module: nn.Module, include_children: bool = True) -> int:
    s = sum(p.numel() * p.element_size() for p in module.parameters(False) if p is not None and p.device.type != "meta")
    s += sum(b.numel() * b.element_size() for b in module.buffers(False) if b is not None and b.device.type != "meta")
    if include_children:
        s += sum(get_module_size(c, True) for c in module.children())
    return s


def infer_fine_grained_device_map(
    model: nn.Module,
    max_memory: Optional[Dict[str, int]],  # keys are str
    no_split: Optional[List[str]],
    verbose: bool,
) -> Dict[str, str]:
    no_split = no_split or []
    dev_map: Dict[str, str] = {}
    frozen: Set[str] = set()
    default_dev = "cpu"
    if max_memory:
        gpus = [k for k, v in max_memory.items() if k != "cpu" and v > 0]
        if gpus:
            default_dev = min(gpus)
            logger.debug(f"Initial map using {default_dev} for no_split.")

    def _traverse(mod: nn.Module, path: str = ""):
        nonlocal dev_map, frozen
        cls_name = mod.__class__.__name__
        is_frozen = any(path.startswith(p + ".") for p in frozen if p)
        if is_frozen:
            pass
        elif cls_name in no_split:
            if path:
                dev_map[path] = default_dev
                frozen.add(path)
            for k_rem in [k for k in dev_map if k.startswith(path + ".")]:
                del dev_map[k_rem]
        elif path and (any(True for _ in mod.parameters(False)) or any(True for _ in mod.buffers(False))):
            dev_map[path] = default_dev
        for name, child in mod.named_children():
            child_path = f"{path}.{name}" if path else name
            if not any(child_path.startswith(p + ".") for p in frozen if p) and child_path not in frozen:
                _traverse(child, child_path)

    _traverse(model)
    if not dev_map and (any(True for _ in model.parameters(False)) or any(True for _ in model.buffers(False))):
        dev_map[""] = default_dev
    return dev_map


class ConfigSignatureGenerator:
    def _serialize_value_recursive(self, val: Any, path: List[str], sig_parts: List[str], raw: Dict, md=3, cd=0, mc=5):
        if cd > md:
            sig_parts.append(f"{'_'.join(path)}_depthlimit")
            raw["status"] = "depth_limit"
            return
        ps = "_".join(path)
        if isinstance(val, torch.Tensor):
            s, dt = "_".join(map(str, val.shape)), str(val.dtype).split(".")[-1]
            sig_parts.append(f"{ps}_T_s{s}_dt{dt}")
            raw.update({"type": "Tensor", "shape": list(val.shape), "dtype": str(val.dtype)})
        elif isinstance(val, (list, tuple)):
            tc = "L" if isinstance(val, list) else "Tu"
            sig_parts.append(f"{ps}_{tc}_len{len(val)}")
            raw.update({"type": val.__class__.__name__, "len": len(val), "elements": []})
            for i, item in enumerate(val):
                if i >= mc:
                    sig_parts.append(f"{ps}_{i}_itemlimit")
                    raw["elements"].append({"status": "item_limit"})
                    break
                inode = {}
                raw["elements"].append(inode)
                self._serialize_value_recursive(item, path + [str(i)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, dict):
            sig_parts.append(f"{ps}_D_len{len(val)}")
            raw.update({"type": "Dict", "len": len(val), "items": {}})
            try:
                keys = sorted(list(val.keys()), key=str)
            except TypeError:
                logger.debug(f"Dict keys {ps} not sortable by str.")
                keys = list(val.keys())
            for i, k_ in enumerate(keys):
                if i >= mc:
                    sig_parts.append(f"{ps}_{str(k_)}_itemlimit")
                    raw["items"][str(k_)] = {"status": "item_limit"}
                    break
                inode = {}
                raw["items"][str(k_)] = inode
                self._serialize_value_recursive(val[k_], path + [str(k_)], sig_parts, inode, md, cd + 1, mc)
        elif isinstance(val, (int, float, bool, str)):
            sval = str(val)
            sval_sig = (sval[:27] + "...") if isinstance(val, str) and len(sval) > 30 else sval
            sig_parts.append(f"{ps}_V_{sval_sig}")
            raw.update({"type": val.__class__.__name__, "value": (sval[:97] + "...") if len(sval) > 100 else sval})
        elif val is None:
            sig_parts.append(f"{ps}_None")
            raw["type"] = "NoneType"
        else:
            tn = val.__class__.__name__
            sig_parts.append(f"{ps}_O_{tn}")
            raw["type"] = tn
            try:
                raw["value_str"] = str(val)[:100]
            except:
                raw["value_str"] = "Error_str_conversion"

    def _get_input_parts(self, mod: nn.Module, args: Tuple, kwargs: Dict, raw_in: Dict) -> List[str]:
        s_p = []
        raw_in["args"] = []
        for i, v_ in enumerate(args):
            node: Dict = {}
            raw_in["args"].append(node)
            self._serialize_value_recursive(v_, [f"arg{i}"], s_p, node)
        raw_in["kwargs"] = {}
        for k_, v_ in sorted(kwargs.items()):
            node: Dict = {}
            raw_in["kwargs"][k_] = node
            self._serialize_value_recursive(v_, [f"kw_{k_}"], s_p, node)
        return s_p

    def _get_weights_parts(self, mod: nn.Module, raw_w: Dict) -> List[str]:
        s_p = []
        raw_w.update({"parameters": {}, "buffers": {}})
        for n, p_ in sorted(mod.named_parameters(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, p_.shape)), str(p_.dtype).split(".")[-1]
            s_p.append(f"p_{n}_s{s}_dt{dt}")
            raw_w["parameters"][n] = {"shape": list(p_.shape), "dtype": str(p_.dtype)}
        for n, b_ in sorted(mod.named_buffers(recurse=True), key=lambda x: x[0]):
            s, dt = "_".join(map(str, b_.shape)), str(b_.dtype).split(".")[-1]
            s_p.append(f"b_{n}_s{s}_dt{dt}")
            raw_w["buffers"][n] = {"shape": list(b_.shape), "dtype": str(b_.dtype)}
        return s_p

    def generate_config_signature(
        self, mod: nn.Module, args: Tuple, kwargs: Dict, dtype: torch.dtype, cb=None
    ) -> Tuple[str, Dict[str, Any]]:
        raw: Dict[str, Any] = {
            "inputs": {},
            "module_structure": {"class_name": mod.__class__.__name__},
            "weights": {},
            "config": {},
        }
        s_p = [f"cls_{mod.__class__.__name__}"]
        s_p.extend(self._get_input_parts(mod, args, kwargs, raw["inputs"]) or ["inputs_empty"])
        w_parts = self._get_weights_parts(mod, raw["weights"])
        if w_parts:
            s_p.append(f"w_{hashlib.md5('_'.join(w_parts).encode()).hexdigest()[:16]}")
            raw["weights"]["hash"] = s_p[-1]
        else:
            s_p.append("w_empty")
        if cb:
            try:
                custom = cb(mod, args, kwargs)
                raw["config"]["custom_cb_out"] = custom
                if isinstance(custom, dict):
                    s_p.extend([f"cc_{k_}{v_}" for k_, v_ in sorted(custom.items())])
                else:
                    logger.warning("Custom sig cb no dict.")
            except Exception as e:
                logger.error(f"Custom sig cb fail: {e}", exc_info=True)
                raw["config"]["custom_cb_out"] = {"error": str(e)}
        s_p.append(f"moddt_{str(dtype).split('.')[-1]}")
        raw["config"]["mod_dtype"] = str(dtype)
        s_p.append(f"py_{sys.version_info.major}.{sys.version_info.minor}")
        raw["config"]["py_ver"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        s_p.append(f"pt_{torch.__version__}")
        raw["config"]["pt_ver"] = torch.__version__
        full_sig = "_".join(s_p)
        raw["full_sig_unhashed"] = full_sig
        hashed_sig = hashlib.md5(full_sig.encode()).hexdigest()
        raw["final_config_sig_hash"] = hashed_sig
        logger.debug(f"Config sig hash: {hashed_sig} (from: {full_sig[:100]}...)")
        return hashed_sig, raw

    def generate_plan_identifier(self, mem_bytes: Dict[str, int]) -> str:
        parts = [f"{k}{v / (1024**3):.1f}G" for k, v in sorted(mem_bytes.items())]
        mem_str = "_".join(parts) or "mem_auto_empty"
        plan_id = hashlib.md5(mem_str.encode()).hexdigest()[:16]
        logger.debug(f"Plan ID: {plan_id} (mem: {mem_str})")
        return plan_id


class InferenceOptimizerHook(ModelHook):
    def __init__(
        self,
        cache_dir="opt_cache",
        num_prefetch_streams=1,
        no_split_module_classes=None,
        custom_signature_callback=None,
        default_offload_policy="cpu",
        force_profiling=False,
        run_profiling_if_needed=True,
        max_memory_gb=None,
    ):
        super().__init__()
        self.base_cache_dir = cache_dir
        os.makedirs(self.base_cache_dir, exist_ok=True)
        self.num_prefetch_streams = max(1, num_prefetch_streams)
        self.no_split_module_classes = no_split_module_classes or []
        self.custom_signature_callback = custom_signature_callback
        self.default_offload_policy = default_offload_policy
        self.force_profiling = force_profiling  # User's initial setting
        self._force_profiling_active = force_profiling  # Internal flag, reset after use
        self.run_profiling_if_needed = run_profiling_if_needed
        self.user_max_memory_gb = max_memory_gb

        self.sig_gen = ConfigSignatureGenerator()
        self.current_plan: Optional[OptimizationPlan] = None
        self.current_config_sig_hash: Optional[str] = None
        self.current_max_memory_bytes: Optional[Dict[str, int]] = None
        self.current_plan_id: Optional[str] = None
        self.last_module_id_processed: Optional[int] = None
        self.active_pf_ctx: Optional[PrefetchContext] = None
        self.hooked_module_instance: Optional[nn.Module] = None
        self.current_module_dtype: Optional[torch.dtype] = None
        self.is_first_forward = True
        self.module: Optional[nn.Module] = None
        logger.info(f"IOHook init. Cache: {self.base_cache_dir}, Prefetch: {self.num_prefetch_streams}")

    def _get_max_mem_bytes(self) -> Dict[str, int]:  # keys are str
        mem_map: Dict[str, int] = {}
        if self.user_max_memory_gb:
            for k, v in self.user_max_memory_gb.items():
                mem_map[str(k)] = int(v * (1024**3))
            logger.info(f"User max_mem: {{k:f'{v / (1024**3):.1f}GB' for k,v in mem_map.items()}}")
        else:
            logger.info("Auto-balancing memory.")
            if not self.hooked_module_instance:
                logger.error("Cannot auto-balance: no module.")
                return {"cpu": 64 * (1024**3)}
            try:
                balanced = get_balanced_memory(
                    self.hooked_module_instance, self.current_module_dtype, False, self.no_split_module_classes
                )
                mem_map = {str(k): v for k, v in balanced.items()}
                logger.info(f"Auto-balanced max_mem: {{k:f'{v / (1024**3):.1f}GB' for k,v in mem_map.items()}}")
            except Exception as e:
                logger.error(f"Auto-balance fail: {e}", exc_info=True)
                mem_map = {"cpu": 64 * (1024**3)}
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        if str(i) not in mem_map:
                            mem_map[str(i)] = 1 * (1024**3)
        return mem_map

    def _get_sig_dir_path(self) -> str:  # For current_config_sig_hash
        if not self.current_config_sig_hash:
            logger.error("Config sig hash None. Cannot get dir.")
            return os.path.join(self.base_cache_dir, "_ERR_NO_CONF_SIG")
        d = os.path.join(self.base_cache_dir, self.current_config_sig_hash)
        os.makedirs(d, exist_ok=True)
        return d

    def _get_plan_file_path(self) -> Optional[str]:  # For current_plan_id
        if not self.current_plan_id:
            logger.error("Plan ID None. Cannot get path.")
            return None
        d = os.path.join(self._get_sig_dir_path(), "plans")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.current_plan_id}.json")

    def _run_profiling(self, args, kwargs, max_mem_prof, conf_sig_hash, raw_sig_details) -> Optional[ProfilingData]:
        logger.info("=" * 20 + " Profiling Session Start " + "=" * 20)
        if not self.hooked_module_instance or not self.current_module_dtype:
            logger.error("Cannot profile: state missing.")
            return None

        mod_to_prof = self.hooked_module_instance
        prof_data = ProfilingData()
        sig_dir_for_save = os.path.join(self.base_cache_dir, conf_sig_hash)
        os.makedirs(sig_dir_for_save, exist_ok=True)
        prof_data_path = os.path.join(sig_dir_for_save, "profiling_data.json")
        raw_details_path = os.path.join(sig_dir_for_save, "raw_signature_details.json")
        try:
            save_to_json_file(raw_sig_details, raw_details_path)
            logger.info(f"Raw sig details saved to {raw_details_path}")
        except Exception as e:
            logger.error(f"Failed to save raw sig details: {e}")

        remove_hook_from_module(mod_to_prof, recurse=True)
        try:
            logger.info(f"Preparing '{mod_to_prof.__class__.__name__}' for profiling.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            init_map = infer_fine_grained_device_map(
                mod_to_prof, None, self.no_split_module_classes, logger.isEnabledFor(logging.DEBUG)
            )
            if not init_map and any(mod_to_prof.parameters()):
                raise RuntimeError("Failed to create init map for profiling.")

            main_dev_prof = (
                torch.device("cuda:0") if torch.cuda.is_available() and "0" in max_mem_prof else torch.device("cpu")
            )
            dispatch_model(mod_to_prof, init_map, main_dev_prof, force_hooks=True)

            ph_count = 0
            for name, sub_mod in mod_to_prof.named_modules():
                if hasattr(sub_mod, "_hf_hook"):
                    add_hook_to_module(sub_mod, ProfilerHook(name), True)
                    ph_count += 1
            logger.info(f"Registered {ph_count} ProfilerHooks.")

            p_args, p_kwargs = copy.deepcopy(args), copy.deepcopy(kwargs)
            logger.info("Warm-up profiling inference...")
            with torch.no_grad():
                _ = mod_to_prof(*p_args, **p_kwargs)
            clear_device_cache()
            with _profile_run_context(prof_data), torch.no_grad():
                logger.info("Main profiling inference...")
                _ = mod_to_prof(*p_args, **p_kwargs)
            prof_data.save(prof_data_path)
        except Exception as e:
            logger.error(f"Error in profiling: {e}", exc_info=True)
            if prof_data.module_stats:
                prof_data.save(os.path.join(sig_dir_for_save, "profiling_data.error.json"))
            return None
        finally:
            logger.info("Cleaning up post-profiling...")
            remove_hook_from_module(mod_to_prof, recurse=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("=" * 20 + " Profiling Session End " + "=" * 20)
        return prof_data

    def _gen_opt_plan(self, prof_data: ProfilingData, max_mem_plan: Dict[str, int]) -> Optional[OptimizationPlan]:
        logger.info("Generating optimization plan...")
        if not self.hooked_module_instance:
            logger.error("Cannot gen plan: no module.")
            return None
        opt = HeuristicOptimizer(prof_data, max_mem_plan)
        return opt.optimize()

    def _setup_module_with_plan(self, mod_to_opt: nn.Module, plan: OptimizationPlan) -> Optional[PrefetchContext]:
        logger.info(f"Preparing/dispatching '{mod_to_opt.__class__.__name__}' with plan...")
        offload = self.default_offload_policy
        if any(d.type == "cpu" for d in plan.optimized_device_map.values()) and offload != "cpu":
            offload = "cpu"

        pf_ctx = PrefetchContext(plan, mod_to_opt, self.num_prefetch_streams, offload)
        remove_hook_from_module(mod_to_opt, recurse=True)  # Remove IOH and any prior hooks

        # state_dict for dispatch if module on meta (pinned CPU version is ideal)
        # For simplicity, this example relies on AlignDevicesHook's internal weights_map being properly populated
        # (e.g., from from_pretrained loading to CPU, then dispatch pinning).
        # If using init_empty_weights without explicit state_dict loading to CPU first, this might be suboptimal.
        # The prefetch hook now tries to pin from weights_map.
        self.cpu_state_dict = mod_to_opt.state_dict()

        main_dev_dispatch = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        map_for_dispatch = {
            k: ("cpu" if v.type == "cpu" else f"{v.type}:{v.index}" if v.index is not None else v.type)
            for k, v in plan.optimized_device_map.items()
        }
        dispatch_model(
            mod_to_opt, map_for_dispatch, main_dev_dispatch, force_hooks=True, state_dict=self.cpu_state_dict
        )

        trig_mods = {i.trigger_module for i in plan.prefetch_schedule}
        pf_mods = {i.module_to_prefetch for i in plan.prefetch_schedule}
        waits, trigs = 0, 0
        for name, sub_mod in mod_to_opt.named_modules():
            if not hasattr(sub_mod, "_hf_hook"):
                continue  # Only process dispatched submodules
            sub_mod_exec_dev = plan.optimized_device_map.get(name, main_dev_dispatch)
            if name in pf_mods:
                add_hook_to_module(sub_mod, PrefetchingWaitHook(pf_ctx, name, sub_mod, sub_mod_exec_dev), True)
                waits += 1
            if name in trig_mods:
                add_hook_to_module(sub_mod, PrefetchingHook(pf_ctx, name, sub_mod), True)
                trigs += 1

        add_hook_to_module(mod_to_opt, self, append=True)
        logger.info(f"Registered {waits} WaitHooks and {trigs} TriggerHooks.")
        return pf_ctx

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        nvtx.range_push("IOHook.pre_forward")
        if self.is_first_forward:
            self.hooked_module_instance = module
            logger.info(f"Hook first fwd: {module.__class__.__name__} (id:{id(module)})")
            try:
                p_dt = next((p.dtype for p in module.parameters(False)), None)
                b_dt = next((b.dtype for b in module.buffers(False)), None)
                self.current_module_dtype = getattr(module, "dtype", p_dt or b_dt or torch.get_default_dtype())
            except Exception as e:
                self.current_module_dtype = torch.get_default_dtype()
                logger.warning(f"Dtype infer fail: {e}. Using {self.current_module_dtype}")
            logger.info(f"Module dtype: {self.current_module_dtype}")
            self.current_max_memory_bytes = self._get_max_mem_bytes()
            self.is_first_forward = False

        if not all([self.hooked_module_instance, self.current_module_dtype, self.current_max_memory_bytes]):
            logger.error("Hook state not init.")
            nvtx.range_pop()
            return args, kwargs

        new_conf_hash, raw_details = self.sig_gen.generate_config_signature(
            self.hooked_module_instance, args, kwargs, self.current_module_dtype, self.custom_signature_callback
        )
        new_plan_id = self.sig_gen.generate_plan_identifier(self.current_max_memory_bytes)

        needs_update = (
            id(self.hooked_module_instance) != self.last_module_id_processed
            or new_conf_hash != self.current_config_sig_hash
            or new_plan_id != self.current_plan_id
            or self.current_plan is None
            or self._force_profiling_active
        )

        if needs_update:
            logger.info(
                f"Plan update triggered. ModID:{id(self.hooked_module_instance) != self.last_module_id_processed}, "
                f"ConfHash:{new_conf_hash != self.current_config_sig_hash}, PlanID:{new_plan_id != self.current_plan_id}, "
                f"NoPlan:{self.current_plan is None}, ForceProf:{self._force_profiling_active}"
            )
            self.current_config_sig_hash, self.current_plan_id = new_conf_hash, new_plan_id

            sig_dir = self._get_sig_dir_path()
            prof_data_path = os.path.join(sig_dir, "profiling_data.json")
            plan_path = self._get_plan_file_path()

            prof_data: Optional[ProfilingData] = None
            if not self._force_profiling_active:
                prof_data = ProfilingData.load(prof_data_path)

            if prof_data is None:  # Need to profile
                if self.run_profiling_if_needed or self._force_profiling_active:
                    logger.info(
                        f"Profiling for conf {self.current_config_sig_hash}. Force={self._force_profiling_active}"
                    )
                    prof_data = self._run_profiling(
                        args, kwargs, self.current_max_memory_bytes, self.current_config_sig_hash, raw_details
                    )
                    if not prof_data:
                        logger.error("Profiling fail.")
                        self.current_plan = None
                        self.active_pf_ctx = None
                        nvtx.range_pop()
                        return args, kwargs
                else:
                    logger.error(f"Prof data missing for {self.current_config_sig_hash}, not run.")
                    self.current_plan = None
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:  # Used existing profiling data
                logger.info(f"Using existing prof data from {prof_data_path}")
                raw_det_path = os.path.join(sig_dir, "raw_signature_details.json")
                if not os.path.exists(raw_det_path):  # Save raw details if missing (e.g. prof skipped)
                    try:
                        save_to_json_file(raw_details, raw_det_path)
                        logger.info(f"Raw sig details {raw_det_path} (prof skipped).")
                    except Exception as e:
                        logger.error(f"Fail save raw sig details (prof skipped): {e}")

            self.current_plan = None  # Reset
            if plan_path:
                self.current_plan = OptimizationPlan.load(plan_path)

            if self.current_plan is None:  # Need to generate plan
                logger.info(
                    f"Generating plan for plan_id {self.current_plan_id} (config {self.current_config_sig_hash})"
                )
                self.current_plan = self._gen_opt_plan(prof_data, self.current_max_memory_bytes)
                if self.current_plan and plan_path:
                    self.current_plan.save(plan_path)
                elif not self.current_plan:
                    logger.error("Plan gen fail.")
                    self.active_pf_ctx = None
                    nvtx.range_pop()
                    return args, kwargs
            else:
                logger.info(f"Loaded existing plan from {plan_path}")

            if self.active_pf_ctx:
                self.active_pf_ctx.clear_all_module_prefetch_streams()
            self.active_pf_ctx = self._setup_module_with_plan(self.hooked_module_instance, self.current_plan)
            if not self.active_pf_ctx:
                logger.error("Failed to prep/dispatch module.")
                self.current_plan = None
                nvtx.range_pop()
                return args, kwargs

            self.last_module_id_processed = id(self.hooked_module_instance)
            self._force_profiling_active = False  # Reset internal flag after one use

            # Arg alignment by the module's current hook (set up by _setup_module_with_plan)
            current_mod_hook = getattr(self.hooked_module_instance, "_hf_hook", None).hooks[0]
            if current_mod_hook:
                logger.debug(f"Calling pre_forward of module's own hook: {type(current_mod_hook)}")
                args, kwargs = current_mod_hook.pre_forward(self.hooked_module_instance, *args, **kwargs)
            else:
                logger.warning(f"No _hf_hook on {self.hooked_module_instance.__class__.__name__} post-setup.")

        nvtx.range_pop()
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        return output


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s"
    )
    # For more detail during development/debugging of the hook itself:
    # logging.getLogger("your_module_name_if_different").setLevel(logging.DEBUG) # Replace with actual module name if not __main__
    # logging.getLogger(__name__).setLevel(logging.DEBUG) # If this script is run directly

    example_logger = logging.getLogger("SDXL_Example")
    example_logger.setLevel(logging.INFO)

    optimizer_hook = InferenceOptimizerHook(
        cache_dir="example_optim_cache_sdxl",
        max_memory_gb={0: 8, "cpu": 24} if torch.cuda.is_available() and torch.cuda.device_count() > 0 else {"cpu": 24},
        force_profiling=False,  # True to re-profile once per config
        num_prefetch_streams=1,
    )

    model_path = os.getenv("CI_TEST_MODEL_PATH", "playground-v2.5-1024px-aesthetic.fp16.safetensors")
    if not os.path.exists(model_path):
        example_logger.warning(f"Model '{model_path}' not found. Skipping SDXL example.")
    else:
        example_logger.info(f"Loading SDXL pipeline from: {model_path}")
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=torch.float16, local_files_only=True, use_safetensors=True
            )
            example_logger.info(f"Pipeline loaded. UNet dtype: {pipe.unet.dtype}")
        except Exception as e:
            example_logger.error(f"Failed to load pipeline: {e}", exc_info=True)
            sys.exit(1)

        add_hook_to_module(pipe.unet, optimizer_hook, append=False)
        example_logger.info(f"IOHook attached to UNet ({pipe.unet.__class__.__name__}).")

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            example_logger.info("Moving VAE & TextEncoders to CPU for UNet GPU memory.")
            pipe.vae.to("cpu")
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
        else:
            example_logger.info("No CUDA GPUs. Running on CPU.")

        prompt = "A majestic dragon soaring through a vibrant sunset sky, fantasy art, highly detailed"
        num_steps = 3
        for i in range(2):
            example_logger.info(f"\n--- Inference pass {i + 1} --- (Prompt: '{prompt}', Steps: {num_steps})")
            try:
                with torch.no_grad():
                    latents = pipe(prompt, num_inference_steps=num_steps, output_type="latent").images
                example_logger.info(
                    f"Pass {i + 1} complete. Latents shape: {latents.shape if latents is not None else 'None'}"
                )
            except Exception as e:
                example_logger.error(f"Pass {i + 1} error: {e}", exc_info=True)
        example_logger.info("\nExample finished.")
