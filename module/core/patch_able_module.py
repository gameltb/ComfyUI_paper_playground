# module witch controlflow can be patch
import typing
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, TypeVar


class PatchType(IntEnum):
    EXCLUSIVE = 0
    RANDOM_ORDER = 1


@dataclass
class PatchDefine:
    patch_type: PatchType = PatchType.EXCLUSIVE

    @staticmethod
    def from_type_hint(patch_type_hint):
        if typing.get_origin(patch_type_hint) == list:
            patch_define = PatchDefine(PatchType.RANDOM_ORDER)
        else:
            patch_define = PatchDefine()

        return patch_define


class ControlFlowPatchModuleMixin:
    def get_patch_module_name(self) -> str:
        """

        Returns:
            str: control flow patch name.
        """
        raise NotImplementedError()


T = TypeVar("T")


class ControlFlowPatchAbleModuleMixin(Generic[T]):
    """fail fast."""

    def __init__(self) -> None:
        self.patch_module_init()

    def patch_module_init(self) -> None:
        self.patch_module_map: T = OrderedDict()
        for patch_name, patch_type_hint in self._get_patch_module_type_hints().items():
            patch_define = PatchDefine.from_type_hint(patch_type_hint)
            if patch_define.patch_type == PatchType.RANDOM_ORDER:
                self.patch_module_map[patch_name] = []
            else:
                self.patch_module_map[patch_name] = None

    def _get_patch_module_type_hints(self):
        type_hints = None

        patch_module_map_type = None
        for type_ in type(self).__orig_bases__:
            if typing.get_origin(type_) is ControlFlowPatchAbleModuleMixin:
                patch_module_map_type = typing.get_args(type_)[0]

        if patch_module_map_type is not None:
            type_hints = typing.get_type_hints(patch_module_map_type)

        return type_hints

    def _get_patch_module_define(self, path_type: str):
        patch_define = None

        type_hints = self._get_patch_module_type_hints()
        if type_hints is not None:
            type_hint = type_hints.get(path_type, None)
            if type_hint is not None:
                patch_define = PatchDefine.from_type_hint(type_hint)

        if patch_define is None:
            raise Exception(f"patch_define {path_type} not registered")
        return patch_define

    def add_patch_module(self, path_type: str, patch_object: ControlFlowPatchModuleMixin):
        patch_define = self._get_patch_module_define(path_type)

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            if path_type in self.patch_module_map:
                raise Exception(f"EXCLUSIVE patch {path_type} has been applied.")

        patch_name = patch_object.get_patch_module_name()

        if hasattr(self, patch_name):
            raise Exception(f"patch {path_type} has a conflicting name {patch_name}.")

        if patch_define.patch_type == PatchType.EXCLUSIVE:
            self.patch_module_map[path_type] = patch_object
        elif patch_define.patch_type == PatchType.RANDOM_ORDER:
            if path_type not in self.patch_module_map:
                self.patch_module_map[path_type] = []
            self.patch_module_map[path_type].append(patch_object)

        self.__setattr__(patch_name, patch_object)

    @property
    def patch_module(self):
        return self.patch_module_map
