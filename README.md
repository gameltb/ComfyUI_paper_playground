# ComfyUI paper playground

Evaluate some papers in ComfyUI, just playground.

# Play

First of all, this is a very experimental repo, API or nodes may change in the future.

## Create your node

This repo use decorator to create and register the node.  
You can put the node file under the `module/comfy/node/` folder , paper repo to `module/paper/`.  
Than import you node at [module/comfy/\_\_init\_\_.py](module/comfy/__init__.py).  
example node

```python
import os
from typing import Annotated, Optional, NamedTuple

from ..registry import register_node
from ..types import BoolType, ComboWidget, ComfyWidget, make_widget, FORCE_INPUT  # some base comfyui type


class ExampleWidget(ComfyWidget):
    TYPE = "EXAMPLE"  # identifier of comfyui type.


ExampleType = Annotated[bool, ExampleWidget()]  # TypeAlias , use as bool

SimpleType = Annotated[str, make_widget("SIMPLE")]  # Simple TypeAlias without definition Widget, use as str
"""Simple type doc"""


class ExampleReturn(NamedTuple):
    output1: ExampleType
    output2: SimpleType


@register_node(
    identifier="node_identifier",  # identifier of node. If not provided, it will be generated from the function name.
    display_name="display_name",  # display_name show to user. it will be generated from the identifier.
    category="loaders",  # category under playground.
)
def example(
    input1: ExampleType,  # use Annotated type
    input2: BoolType = True,  # you can set default value.
    op_input1: Annotated[BoolType, FORCE_INPUT:True] = True,  # change widget property,
    combo_input0: Annotated[
        str, ComboWidget(choices=["int"])
    ] = None,  # use Combo widget to select an item from the list.
    combo_input1: Annotated[
        type, ComboWidget(choices={"int": int})
    ] = None,  # with dict choices Combo widget will automatically convert  key to value .
    combo_input2: Annotated[
        str, ComboWidget(choices=lambda: os.listdir("."))
    ] = None,  # with lambda Combo widget will update list when web page refresh.
    combo_input3: Annotated[
        Optional[type], ComboWidget(choices=lambda: {"int": int}, ext_none_choice="none")
    ] = None,  # you can set an extra None option name, which is converted to None when passed in.
) -> ExampleReturn:  # return NamedTuple or use tuple[ExampleType, SimpleType] without name
    """example node description.

    Args:
        input1 (ExampleType): input1 tooltip
        input2 (BoolType): input2 bool tooltip
    """
    # do what you want.
    # Note: The node caches all input parameters for other or future node executions, so these parameters should be read-only, there are currently no constraints to guarantee this, accidentally modifying the parameter object may result in unexpected behavior, in this case please use the copy or copy method of the object.
    pass
```
