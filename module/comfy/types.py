import typing

from pydantic import BaseModel


class ComfyWidgetType(BaseModel):
    """Base type for ComfyUI types that have options controlling how they are displayed."""

    TYPE: typing.ClassVar
    required: bool = True

    def opts(self):
        return self.model_dump(mode="python", exclude_none=True)

    @property
    def type(self):
        return self.TYPE

    def __getitem__(self, item):
        return item


class Int(ComfyWidgetType):
    TYPE = "INT"
    min: int = None
    max: int = None
    step: int = None
    display: typing.Literal["number", "slider"] = None


class Float(ComfyWidgetType):
    TYPE = "FLOAT"
    min: float = None
    max: float = None
    step: float = None
    round: float = None
    display: typing.Literal["number", "slider"] = None


class String(ComfyWidgetType):
    TYPE = "STRING"
    multiline: bool = None


class Bool(ComfyWidgetType):
    TYPE = "BOOLEAN"
    label_on: str = None
    label_off: str = None


class Color(ComfyWidgetType):
    """Widget only available if you have MTB node pack"""

    TYPE = "COLOR"


class IMAGE(ComfyWidgetType):
    TYPE = "IMAGE"


class LATENT(ComfyWidgetType):
    TYPE = "LATENT"


class Combo(ComfyWidgetType):
    TYPE = "COMBO"
    choices: typing.Mapping[str, typing.Any]
    ext_none_choice: str = None

    def opts(self):
        return self.model_dump(mode="python", exclude_none=True, exclude={"choices"})

    @property
    def type(self):
        typs = list(self.choices.keys())
        if self.ext_none_choice is not None:
            typs.append(self.ext_none_choice)
        return typs

    def __getitem__(self, item):
        if item == self.ext_none_choice:
            return None
        return self.choices[item]


# Workaround ComfyUI #257
Any = type("AnyType", (str,), {"__ne__": lambda self, value: False})("*")


__all__ = ["Int", "Float", "String", "Bool", "Color", "IMAGE", "LATENT", "Combo", "Any"]
