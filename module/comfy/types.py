import typing

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


class Int(ComfyWidgetType):
    TYPE = "INT"
    forceInput: bool = False
    min: int = None
    max: int = None
    step: int = None
    display: typing.Literal["number", "slider"] = None


class Float(ComfyWidgetType):
    TYPE = "FLOAT"
    forceInput: bool = False
    min: float = None
    max: float = None
    step: float = None
    round: float = None
    display: typing.Literal["number", "slider"] = None


class String(ComfyWidgetType):
    TYPE = "STRING"
    forceInput: bool = False
    multiline: bool = None


class Bool(ComfyWidgetType):
    TYPE = "BOOLEAN"
    forceInput: bool = False
    label_on: str = None
    label_off: str = None


class Color(ComfyWidgetType):
    """Widget only available if you have MTB node pack"""

    TYPE = "COLOR"


class IMAGE(ComfyWidgetType):
    """Tensor [B,H,W,C]"""

    TYPE = "IMAGE"


class LATENT(ComfyWidgetType):
    TYPE = "LATENT"


class Combo(ComfyWidgetType):
    TYPE = "COMBO"
    forceInput: bool = False
    choices: typing.Mapping[str, typing.Any] | typing.List[str] | typing.Callable[
        [], typing.Mapping[str, typing.Any] | typing.List[str]
    ]
    ext_none_choice: str = None
    choices_cache: typing.Mapping[str, typing.Any] | typing.List[str]

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
Any = type("AnyType", (str,), {"__ne__": lambda self, value: False})("*")


__all__ = ["Int", "Float", "String", "Bool", "Color", "IMAGE", "LATENT", "Combo", "Any"]
