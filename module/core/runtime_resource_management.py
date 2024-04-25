import gc
import sys
from typing import Any, Optional

import diffusers
import psutil
import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device
import accelerate

resources_device = torch.device


class AutoManage:
    def __init__(
        self,
        obj,
        runtime_device=None,
        offload_device=resources_device("cpu"),
        inference_memory_size=1024 * 1024 * 1024,
    ) -> None:
        if runtime_device is None:
            device_strategy = FixDeviceStrategy()
        else:
            device_strategy = FixDeviceStrategy(runtime_device, offload_device)
        if not isinstance(obj, ResourcesUser):
            user = ResourcesManagement.find_user(obj)
            if user is None:
                if isinstance(obj, torch.nn.Module):
                    user = TorchModuleWrapper(obj, device_strategy)
                elif isinstance(obj, diffusers.DiffusionPipeline):
                    user = DiffusersPipelineWrapper(obj, device_strategy)
                else:
                    raise NotImplementedError()
                ResourcesManagement.add_user(user)
            obj = user
        assert isinstance(obj, ResourcesUser)
        obj.inference_memory_size = inference_memory_size
        self.user = obj

    def load(self):
        self.user.set_keep_status()
        self.user.load()

    def offload(self):
        self.user.remove_keep_status()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.offload()

    def get_device(self):
        return self.user.device_strategy.get_runtime_device()


class AutoManageHook(ModelHook):
    def __init__(self, am: AutoManage):
        self.am = am

    @property
    def execution_device(self):
        return self.am.get_device()

    def pre_forward(self, module, *args, **kwargs):
        self.am.load()
        return send_to_device(args, self.am.get_device()), send_to_device(kwargs, self.am.get_device())

    def post_forward(self, module, output):
        self.am.offload()
        return output


class DeviceStrategy:
    def free_runtime_device(self):
        raise NotImplementedError()

    def free_offload_device(self):
        raise NotImplementedError()

    def get_runtime_device(self):
        raise NotImplementedError()

    def get_offload_device(self):
        raise NotImplementedError()

    def is_manage_by(self, management):
        raise NotImplementedError()


class FixDeviceStrategy(DeviceStrategy):
    def __init__(self, runtime_device=resources_device("cuda", 0), offload_device=resources_device("cpu")) -> None:
        self.runtime_management = get_management(runtime_device)
        self.offload_management = get_management(offload_device)

    def free_runtime_device(self, size: int):
        return self.runtime_management.free(size)

    def free_offload_device(self, size: int):
        return self.offload_management.free(size)

    def get_runtime_device(self):
        return self.runtime_management.device

    def get_offload_device(self):
        return self.offload_management.device

    def is_manage_by(self, management):
        return management is self.runtime_management


class ResourcesUser:
    def __init__(self) -> None:
        self.keep_in_load = False
        self.device_strategy = DeviceStrategy()
        self.inference_memory_size = 0

    @property
    def manage_object(self):
        raise NotImplementedError()

    def load(self, device: Optional[resources_device] = None):
        raise NotImplementedError()

    def offload(self, device: Optional[resources_device] = None):
        raise NotImplementedError()

    def is_keep_status(self):
        return self.keep_in_load

    def set_keep_status(self):
        self.keep_in_load = True

    def remove_keep_status(self):
        self.keep_in_load = False


class TorchStateDictWrapper(ResourcesUser):
    pass


class TorchModuleWrapper(ResourcesUser):
    def __init__(self, torch_model: torch.nn.Module, device_strategy: DeviceStrategy) -> None:
        super().__init__()
        self.torch_model = torch_model
        self.device_strategy = device_strategy

    @property
    def manage_object(self):
        return self.torch_model

    def load(self, device: Optional[resources_device] = None):
        origin_keep_in_load = self.keep_in_load
        self.keep_in_load = True
        try:
            try:
                self.torch_model.to(device=self.device_strategy.get_runtime_device())
            except Exception:
                self.device_strategy.free_runtime_device(self.module_size(self.torch_model))
                self.torch_model.to(device=self.device_strategy.get_runtime_device())

            self.device_strategy.free_runtime_device(self.inference_memory_size)
        finally:
            self.keep_in_load = origin_keep_in_load

    def offload(self, device: Optional[resources_device] = None):
        self.torch_model.to(device=self.device_strategy.get_offload_device())

    @staticmethod
    def module_size(module: torch.nn.Module):
        module_mem = 0
        sd = module.state_dict()
        for k in sd:
            t = sd[k]
            module_mem += t.nelement() * t.element_size()
        return module_mem


class DiffusersPipelineWrapper(ResourcesUser):
    def __init__(self, pipeline: diffusers.DiffusionPipeline, device_strategy: DeviceStrategy) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.device_strategy = device_strategy
        self.all_module_hooks_map = {}

        all_model_components = {k: v for k, v in self.pipeline.components.items() if isinstance(v, torch.nn.Module)}

        for name, model in all_model_components.items():
            if name in self.all_module_hooks_map:
                continue
            if not isinstance(model, torch.nn.Module):
                continue

            hook = AutoManageHook(AutoManage(model))
            add_hook_to_module(model, hook, append=True)
            self.all_module_hooks_map[name] = hook

    @property
    def manage_object(self):
        return self.pipeline

    def load(self, device: Optional[resources_device] = None):
        pass

    def offload(self, device: Optional[resources_device] = None):
        pass

    def remove_keep_status(self):
        super().remove_keep_status()
        for name, hook in self.all_module_hooks_map.items():
            hook.post_forward(None, None)

    @staticmethod
    def pipeline_size(pipeline: diffusers.DiffusionPipeline):
        pipe_mem = 0
        for comp_name, comp in pipeline.components.items():
            if isinstance(comp, torch.nn.Module):
                pipe_mem += TorchModuleWrapper.module_size(comp)
        return pipe_mem


class ResourcesManagement:
    resources_users: list[ResourcesUser] = []

    def __init__(self, device: resources_device) -> None:
        self.device = device

    def get_free(self) -> int:
        raise NotImplementedError()

    def free(self, size: int) -> bool:
        if self.get_free() > size:
            return

        for _ in range(2):
            self.clean_user()
            if self.get_free() > size:
                return

        for user in self.resources_users:
            if user.is_keep_status():
                continue
            if user.device_strategy.is_manage_by(self):
                user.offload()
            if self.get_free() > size:
                return

        try:
            import comfy.model_management

            comfy.model_management.unload_all_models()
        except Exception:
            pass

        if self.get_free() < size:
            print("The required ram is not satisfied.")

    @classmethod
    def find_user(cls, manage_object):
        for user in cls.resources_users:
            if manage_object is user.manage_object:
                return user

    @classmethod
    def add_user(cls, user):
        cls.resources_users.append(user)

    @classmethod
    def remove_user(cls, user):
        cls.resources_users.remove(user)

    @classmethod
    def clean_user(cls):
        for user in cls.resources_users[:]:
            if sys.getrefcount(user.manage_object) <= 2:
                cls.remove_user(user)
        gc.collect()


class MemoryManagementCPU(ResourcesManagement):
    def __init__(self) -> None:
        super().__init__(resources_device("cpu"))

    def get_free(self):
        return psutil.virtual_memory().available


class MemoryManagementCUDA(ResourcesManagement):
    def __init__(self, device: resources_device) -> None:
        assert device.type == "cuda"
        super().__init__(device)

    def get_free(self):
        stats = torch.cuda.memory_stats(self.device)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]
        mem_free_cuda, _ = torch.cuda.mem_get_info(self.device)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total


MANAGEMENT_INSTANCE_MAP: dict[resources_device, ResourcesManagement] = {}


def get_management(device: resources_device) -> ResourcesManagement:
    if device.index is None:
        device = resources_device(device.type, index=0)
    if device not in MANAGEMENT_INSTANCE_MAP:
        if device.type == "cpu":
            MANAGEMENT_INSTANCE_MAP[device] = MemoryManagementCPU()
        elif device.type == "cuda":
            MANAGEMENT_INSTANCE_MAP[device] = MemoryManagementCUDA(device)

    return MANAGEMENT_INSTANCE_MAP[device]
