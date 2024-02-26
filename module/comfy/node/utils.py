from typing import Annotated

from ..registry import register_node
from ..types import StringType, ReturnUI, new_widget


@register_node(identifier="ShowString", category="utils", output=True)
def show_string(string: Annotated[StringType, new_widget(StringType, is_forceInput=True)]):
    return ReturnUI(ui={"string": [string]})
