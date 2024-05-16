# wrapper to module have pacth module, make it run and can be export to onnx.
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple

import torch


# module can be inject to other network and get ext input it self.
class PatchModuleKwargsHook:
    """"""

    def __init__(self) -> None:
        self.ext_kwargs = {}

    def __call__(self, module, args, kwargs) -> Tuple[Any, Dict[str, torch.Tensor]]:
        kwargs.update(self.ext_kwargs)
        return (args, kwargs)


class PatchAbleModuleWrapper(torch.nn.Module):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.replaced_module_kwargs_hook_map: OrderedDict[str, PatchModuleKwargsHook] = OrderedDict()

    def forward(self, /, **kwargs):
        for k, v in self.replaced_module_kwargs_hook_map.items():
            hook_kwarg_prefix = f"{k}_"
            for arg_name in list(kwargs.keys()):
                if arg_name.startswith(hook_kwarg_prefix):
                    v.ext_kwargs[arg_name.removeprefix(hook_kwarg_prefix)] = kwargs.pop(arg_name)

        return self.module(**kwargs)

    def register_forward_ext_kwargs_hook(self, hook_id):
        if hook_id in self.replaced_module_kwargs_hook_map:
            raise Exception(f"hook_id {hook_id} already registered")
        hook = PatchModuleKwargsHook()
        self.replaced_module_kwargs_hook_map[hook_id] = hook

    def apply_forward_ext_kwargs_hook(self, module: torch.nn.Module, hook_id):
        ext_kwargs_hook = self.replaced_module_kwargs_hook_map.get(hook_id, None)
        if ext_kwargs_hook is None:
            raise Exception(f"module_ext_kwargs_hook_id {hook_id} not registered")
        module.register_forward_pre_hook(ext_kwargs_hook, with_kwargs=True)
