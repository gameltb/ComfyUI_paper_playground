import typing
import torch

from pydantic import BaseModel


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


class IntWidget(ComfyWidgetType):
    TYPE = "INT"
    forceInput: bool = False
    min: int = None
    max: int = None
    step: int = None
    display: typing.Literal["number", "slider"] = None


IntType = typing.Annotated[int, IntWidget()]


class FloatWidget(ComfyWidgetType):
    TYPE = "FLOAT"
    forceInput: bool = False
    min: float = None
    max: float = None
    step: float = None
    round: float = None
    display: typing.Literal["number", "slider"] = None


FloatType = typing.Annotated[float, FloatWidget()]


class StringWidget(ComfyWidgetType):
    TYPE = "STRING"
    forceInput: bool = False
    multiline: bool = None


StringType = typing.Annotated[str, StringWidget()]


class BoolWidget(ComfyWidgetType):
    TYPE = "BOOLEAN"
    forceInput: bool = False
    label_on: str = None
    label_off: str = None


BoolType = typing.Annotated[bool, BoolWidget()]


class ColorWidget(ComfyWidgetType):
    """Widget only available if you have MTB node pack"""

    TYPE = "COLOR"


ColorType = typing.Annotated[torch.Tensor, ColorWidget()]


class ImageWidget(ComfyWidgetType):
    """Tensor [B,H,W,C]"""

    TYPE = "IMAGE"


ImageType = typing.Annotated[torch.Tensor, ImageWidget()]


class LatentWidget(ComfyWidgetType):
    TYPE = "LATENT"


LatentType = typing.Annotated[torch.Tensor, LatentWidget()]


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


class ReturnUI(BaseModel):
    ui: dict = dict()
    result: tuple = tuple()


__all__ = [
    "IntWidget",
    "FloatWidget",
    "StringWidget",
    "BoolWidget",
    "ColorWidget",
    "ImageWidget",
    "LatentWidget",
    "ComboWidget",
    "AnyType",
    "IntType",
    "FloatType",
    "StringType",
    "BoolType",
    "ColorType",
    "ImageType",
    "LatentType",
    "ReturnUI",
]
