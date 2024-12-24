from enum import StrEnum
from typing import Annotated, Any, Callable, ClassVar, List, Literal, Mapping, Optional, Union

from ...core.annotated_model import AnnotatedBaseModel, find_annotated_model

REGISTERED_TYPES = set()


class ComfyWidgetInputType(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    HIDDEN = "hidden"


class ComfyWidget(AnnotatedBaseModel):
    """Base type for ComfyUI types that have options controlling how they are displayed."""

    TYPE: ClassVar
    input_type: ComfyWidgetInputType = ComfyWidgetInputType.REQUIRED
    forceInput: bool = True
    lazy: bool = False
    tooltip: Optional[str] = None
    default: Optional[object] = None

    def opts(self):
        return self.model_dump(mode="python", exclude_none=True)

    @property
    def type(self):
        return self.TYPE

    def __getitem__(self, item):
        return item

    def __init_subclass__(cls):
        assert cls.TYPE not in REGISTERED_TYPES, f"Registered type {cls.TYPE}"
        REGISTERED_TYPES.add(cls.TYPE)


INPUT_TYPE = "input_type"
FORCE_INPUT = "forceInput"
LAZY = "lazy"
TOOLTIP = "tooltip"


def find_comfy_widget_type_annotation(annotation: Union[type, ComfyWidget]) -> Optional[ComfyWidget]:
    return find_annotated_model(annotation, model_type=ComfyWidget)


def make_widget(
    type_str: str,
    is_required: bool = True,
    is_forceInput: bool = True,
    widget_input_type: Optional[ComfyWidgetInputType] = None,
):
    """make simple widget."""

    class Widget(ComfyWidget):
        TYPE = type_str
        input_type: ComfyWidgetInputType = (
            widget_input_type
            if widget_input_type is not None
            else (ComfyWidgetInputType.REQUIRED if is_required else ComfyWidgetInputType.OPTIONAL)
        )
        forceInput: bool = is_forceInput

    return Widget()


def new_widget(tp, is_required: bool = True, is_forceInput: bool = True, **kwargs):
    return find_comfy_widget_type_annotation(tp).model_copy(
        update={
            "input_type": ComfyWidgetInputType.REQUIRED if is_required else ComfyWidgetInputType.OPTIONAL,
            "forceInput": is_forceInput,
            **kwargs,
        }
    )


class IntWidget(ComfyWidget):
    TYPE = "INT"
    forceInput: bool = False
    min: int = None
    max: int = None
    step: int = None
    display: Literal["number", "slider"] = None


IntType = Annotated[int, IntWidget()]
IntSeedType = Annotated[int, IntWidget(min=0, max=0xFFFFFFFFFFFFFFFF)]
IntStepsType = Annotated[int, IntWidget(min=1, max=10000)]


class FloatWidget(ComfyWidget):
    TYPE = "FLOAT"
    forceInput: bool = False
    min: float = None
    max: float = None
    step: float = None
    round: float = None
    display: Literal["number", "slider"] = None


FloatType = Annotated[float, FloatWidget()]
FloatCFGType = Annotated[float, FloatWidget(min=0.0, max=100.0, step=0.1, round=0.01)]
FloatPercentageType = Annotated[float, FloatWidget(min=0.0, max=1.0, step=0.01)]
"""0 to 1, step 0.01."""


class StringWidget(ComfyWidget):
    TYPE = "STRING"
    forceInput: bool = False
    multiline: bool = None
    dynamicPrompts: bool = None


StringType = Annotated[str, StringWidget()]
StringMultilineType = Annotated[str, StringWidget(multiline=True)]


class BoolWidget(ComfyWidget):
    TYPE = "BOOLEAN"
    forceInput: bool = False
    label_on: str = None
    label_off: str = None


BoolType = Annotated[bool, BoolWidget()]


class ComboWidget(ComfyWidget):
    TYPE = "COMBO"
    forceInput: bool = False
    choices: Union[Mapping[str, Any], List[str], Callable[[], Union[Mapping[str, Any], List[str]]]]
    ext_none_choice: str = None
    choices_cache: Mapping[str, Any] | List[str] = None

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
AnyType = Annotated[Any, type("AnyType", (str,), {"__ne__": lambda self, value: False})("*")]
