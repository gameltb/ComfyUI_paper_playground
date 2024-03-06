import gc
import sys
from typing import Any, Optional

import psutil
import torch
import diffusers

resources_device = torch.device


class AutoManage:
    def __init__(
        self,
        obj,
        runtime_device,
        offload_device=resources_device("cpu"),
        inference_memory_size=1024 * 1024 * 1024,
    ) -> None:
        if not isinstance(obj, ResourcesUser):
            user = ResourcesManagement.find_user(obj)
            if user is None:
                if isinstance(obj, torch.nn.Module):
                    user = TorchModuleWrapper(obj, runtime_device, offload_device)
                elif isinstance(obj, diffusers.DiffusionPipeline):
                    user = DiffusersPipelineWrapper(obj, runtime_device, offload_device)
                else:
                    raise NotImplementedError()
                ResourcesManagement.add_user(user)
            obj = user
        assert isinstance(obj, ResourcesUser)
        obj.inference_memory_size = inference_memory_size
        self.user = obj

    def __enter__(self):
        self.user.keep_in_load = True
        self.user.load()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        self.user.keep_in_load = False


class ResourcesUser:
    def __init__(self) -> None:
        self.keep_in_load = False
        self.runtime_management = None
        self.offload_management = None
        self.inference_memory_size = 0

    @property
    def manage_object(self):
        raise NotImplementedError()

    def load(self, device: Optional[resources_device] = None):
        raise NotImplementedError()

    def offload(self, device: Optional[resources_device] = None):
        raise NotImplementedError()


class TorchStateDictWrapper(ResourcesUser):
    pass


class TorchModuleWrapper(ResourcesUser):
    def __init__(self, torch_model: torch.nn.Module, runtime_device, offload_device) -> None:
        super().__init__()
        self.torch_model = torch_model
        self.runtime_management = get_management(runtime_device)
        self.offload_management = get_management(offload_device)

    @property
    def manage_object(self):
        return self.torch_model

    def load(self, device: Optional[resources_device] = None):
        try:
            self.torch_model.to(device=self.runtime_management.device)
        except Exception:
            origin_keep_in_load = self.keep_in_load
            self.keep_in_load = True
            self.runtime_management.free(self.module_size(self.torch_model) + self.inference_memory_size)
            self.keep_in_load = origin_keep_in_load

            self.torch_model.to(device=self.runtime_management.device)

    def offload(self, device: Optional[resources_device] = None):
        self.torch_model.to(device=self.offload_management.device)

    @staticmethod
    def module_size(module: torch.nn.Module):
        module_mem = 0
        sd = module.state_dict()
        for k in sd:
            t = sd[k]
            module_mem += t.nelement() * t.element_size()
        return module_mem


class DiffusersPipelineWrapper(ResourcesUser):
    def __init__(self, pipeline: diffusers.DiffusionPipeline, runtime_device, offload_device) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.runtime_management = get_management(runtime_device)
        self.offload_management = get_management(offload_device)

    @property
    def manage_object(self):
        return self.pipeline

    def load(self, device: Optional[resources_device] = None):
        origin_keep_in_load = self.keep_in_load
        self.keep_in_load = True
        self.runtime_management.free(self.pipeline_size(self.pipeline) + self.inference_memory_size)
        self.keep_in_load = origin_keep_in_load

        if not hasattr(self.pipeline, "_all_hooks") or len(self.pipeline._all_hooks) == 0:
            self.pipeline.enable_model_cpu_offload(device=self.runtime_management.device)

    def offload(self, device: Optional[resources_device] = None):
        self.pipeline.maybe_free_model_hooks()

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

        self.clean_user()

        for user in self.resources_users:
            if user.keep_in_load:
                continue
            if user.runtime_management is self:
                user.offload()
            if self.get_free() > size:
                return

        try:
            import comfy.model_management

            comfy.model_management.unload_all_models()
        except Exception:
            pass

        if self.get_free() > size:
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
