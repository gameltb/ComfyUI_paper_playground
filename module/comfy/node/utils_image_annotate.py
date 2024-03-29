import json
from dataclasses import dataclass
from typing import Annotated, Union

from ..registry import register_node
from ..types import ComfyWidgetType, ImageType


@dataclass
class ImageAnnotateCanvasSelect:
    label: str = None
    coor: Union[list[tuple[int, int]], tuple[int, int]] = None
    active: bool = False
    creating: bool = False
    dragging: bool = False
    uuid: str = None
    index: int = None
    labelFillStyle: str = None
    textFillStyle: str = None
    fillStyle: str = None
    type: int = None


class IMAGE_ANNOTATE_Widget(ComfyWidgetType):
    TYPE = "IMAGE_ANNOTATE"
    required: bool = True
    forceInput: bool = False
    image_input_name: str = "image"

    def __getitem__(self, item):
        if isinstance(item,str):
            return [ImageAnnotateCanvasSelect(**a) for a in json.loads(item)]
        return item


ImageAnnotateType = Annotated[list[ImageAnnotateCanvasSelect], IMAGE_ANNOTATE_Widget()]


@register_node(category="utils")
def image_annotate_input(image: ImageType, annotate: ImageAnnotateType) -> tuple[ImageAnnotateType]:
    print(annotate)
    return (annotate,)
