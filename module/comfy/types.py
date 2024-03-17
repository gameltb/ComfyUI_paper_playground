import typing

import comfy.clip_vision
import comfy.controlnet
import comfy.model_patcher
import comfy.sd
import torch
from pydantic import BaseModel

REGISTERED_TYPES = set()


class ComfyWidgetType(BaseModel):
    """Base type for ComfyUI types that have options controlling how they are displayed."""

    TYPE: typing.ClassVar
    required: bool = True
    forceInput: bool = True

    def opts(self):
        return self.model_dump(mode="python", exclude_none=True)

    @property
    def type(self):
        return self.TYPE

    def __getitem__(self, item):
        return item

    def __init_subclass__(cls):
        assert cls.TYPE not in REGISTERED_TYPES
        REGISTERED_TYPES.add(cls.TYPE)


def find_comfy_widget_type_annotation(
    tp: typing.Union[typing.Annotated, ComfyWidgetType],
) -> typing.Union[ComfyWidgetType, None]:
    if isinstance(tp, ComfyWidgetType):
        return tp
    elif hasattr(tp, "__metadata__"):
        for meta in reversed(tp.__metadata__):
            if isinstance(meta, ComfyWidgetType):
                return meta
    return None


def gen_simple_new_type(
    cls,
    type_str: str,
    is_required: bool = True,
    is_forceInput: bool = True,
):
    class Widget(ComfyWidgetType):
        TYPE = type_str
        required: bool = is_required
        forceInput: bool = is_forceInput

    return typing.Annotated[cls, Widget()]


def new_widget(tp, is_required: bool = True, is_forceInput: bool = True, **kwargs):
    return find_comfy_widget_type_annotation(tp).model_copy(
        update={"required": is_required, "forceInput": is_forceInput, **kwargs}
    )


class IntWidget(ComfyWidgetType):
    TYPE = "INT"
    forceInput: bool = False
    min: int = None
    max: int = None
    step: int = None
    display: typing.Literal["number", "slider"] = None


IntType = typing.Annotated[int, IntWidget()]
IntSeedType = typing.Annotated[int, IntWidget(min=0, max=0xFFFFFFFFFFFFFFFF)]
IntStepsType = typing.Annotated[int, IntWidget(min=1, max=10000)]


class FloatWidget(ComfyWidgetType):
    TYPE = "FLOAT"
    forceInput: bool = False
    min: float = None
    max: float = None
    step: float = None
    round: float = None
    display: typing.Literal["number", "slider"] = None


FloatType = typing.Annotated[float, FloatWidget()]
FloatCFGType = typing.Annotated[float, FloatWidget(min=0.0, max=100.0, step=0.1, round=0.01)]
FloatPercentageType = typing.Annotated[float, FloatWidget(min=0.0, max=1.0, step=0.01)]
"""0 to 1, step 0.01."""


class StringWidget(ComfyWidgetType):
    TYPE = "STRING"
    forceInput: bool = False
    multiline: bool = None


StringType = typing.Annotated[str, StringWidget()]
StringMultilineType = typing.Annotated[str, StringWidget(multiline=True)]


class BoolWidget(ComfyWidgetType):
    TYPE = "BOOLEAN"
    forceInput: bool = False
    label_on: str = None
    label_off: str = None


BoolType = typing.Annotated[bool, BoolWidget()]

MaskType = gen_simple_new_type(torch.Tensor, "MASK")
"""Tensor [B,H,W]"""

ImageType = gen_simple_new_type(torch.Tensor, "IMAGE")
"""Tensor [B,H,W,C] float32 cpu"""

LatentType = gen_simple_new_type(dict[str, torch.Tensor], "LATENT")
"""samples : Tensor [B,H,W,C]"""

VaeType = gen_simple_new_type(comfy.sd.VAE, "VAE")

ModelType = gen_simple_new_type(comfy.model_patcher.ModelPatcher, "MODEL")

ClipType = gen_simple_new_type(comfy.sd.CLIP, "CLIP")

ConditioningType = gen_simple_new_type(list[tuple[torch.Tensor, dict[str, torch.Tensor]]], "CONDITIONING")

ClipVisionType = gen_simple_new_type(comfy.clip_vision.ClipVisionModel, "CLIP_VISION")

ControlNetType = gen_simple_new_type(
    typing.Union[comfy.controlnet.ControlLora, comfy.controlnet.ControlNet, comfy.controlnet.T2IAdapter], "CONTROL_NET"
)


class ComboWidget(ComfyWidgetType):
    TYPE = "COMBO"
    forceInput: bool = False
    choices: typing.Union[
        typing.Mapping[str, typing.Any],
        typing.List[str],
        typing.Callable[[], typing.Union[typing.Mapping[str, typing.Any], typing.List[str]]],
    ]
    ext_none_choice: str = None
    choices_cache: typing.Mapping[str, typing.Any] | typing.List[str] = None

    def opts(self):
        return self.model_dump(
            mode="python", exclude_none=True, exclude={"choices", "ext_none_choice", "choices_cache"}
        )

    @property
    def type(self):
        choices = self.choices
        if callable(choices):
            choices = choices()

        self.choices_cache = choices
        if isinstance(choices, dict):
            typs = list(choices.keys())
        elif isinstance(choices, list):
            typs = choices[:]

        if self.ext_none_choice is not None:
            typs.append(self.ext_none_choice)
        return typs

    def __getitem__(self, item):
        if item == self.ext_none_choice:
            return None

        choices = self.choices_cache
        if isinstance(choices, list):
            return item
        elif isinstance(choices, dict):
            return choices[item]


# Workaround ComfyUI #257
AnyType = typing.Annotated[typing.Any, type("AnyType", (str,), {"__ne__": lambda self, value: False})("*")]


class ReturnType(BaseModel):
    ui: dict[str, list[typing.Any]] = dict()
    result: tuple = tuple()
