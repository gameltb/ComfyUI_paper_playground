# module witch controlflow can be patch
from dataclasses import dataclass
from enum import IntEnum
from collections import OrderedDict


class PatchType(IntEnum):
    EXCLUSIVE = 0
    RANDOM_ORDER = 1


@dataclass
class PatchDefine:
    patch_type: PatchType = PatchType.EXCLUSIVE


class ControlFlowPatchModuleMixin:
    def get_patch_name(self) -> str:
        """

        Returns:
            str: control flow patch name.
        """
        raise NotImplementedError()


class ControlFlowPatchAbleModuleMixin:
    """fail fast."""

    supported_patch: dict[str, PatchDefine] = {}

    def __init__(self) -> None:
        self.init_patch()

    def init_patch(self) -> None:
        self.patch_module_map = OrderedDict()

    def get_patch_define(self, path_type: str):
        patch_define = self.supported_patch.get(path_type, None)
        if patch_define is None:
            raise Exception(f"patch_define {path_type} not registered")
        return patch_define

    def add_patch(self, path_type: str, patch_object: ControlFlowPatchModuleMixin):
        patch_define = self.get_patch_define(path_type)

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            if path_type in self.patch_module_map:
                raise Exception(f"EXCLUSIVE patch {path_type} has been applied.")

        patch_name = patch_object.get_patch_name()

        if hasattr(self, patch_name):
            raise Exception(f"patch {path_type} has a conflicting name {patch_name}.")

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            self.patch_module_map[path_type] = patch_object
        elif patch_define.patch_type == PatchType.RANDOM_ORDER:
            if path_type not in self.patch_module_map:
                self.patch_module_map[path_type] = []
            self.patch_module_map[path_type].append(patch_object)

        self.__setattr__(patch_name, patch_object)

    def get_patch(self, path_type: str):
        patch_define = self.get_patch_define(path_type)
        default_value = None
        if patch_define.patch_type == PatchType.RANDOM_ORDER:
            default_value = []
        return self.patch_module_map.get(path_type, default_value)
