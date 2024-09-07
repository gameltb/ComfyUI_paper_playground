from typing import Annotated

from plyfile import PlyData

from ...common.path_tool import get_output_path
from ..registry import register_node
from ..types import ReturnType, StringType, make_widget

PlyDataType = Annotated[PlyData, make_widget("PLY_DATA")]


@register_node(category="utils/plyfile", output=True)
def save_ply(ply_data: PlyDataType, save_path: StringType = "output.ply"):
    save_path = get_output_path(save_path)
    ply_data.write(save_path)
    return ReturnType(ui={"save_path": [save_path]})
