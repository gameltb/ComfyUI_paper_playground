import pathlib
from typing import Annotated

from ...core.runtime_resource_management import get_management, resources_device
from ..registry import register_node
from ..types import ReturnType, StringType, new_widget


@register_node(identifier="ShowString", category="utils", output=True)
def show_string(string: Annotated[StringType, new_widget(StringType, is_forceInput=True)]):
    return ReturnType(ui={"string": [string]})


@register_node(identifier="GeneratePlaygroundNodeDocumentation", category="utils", output=True)
def generate_playground_node_documentation():
    """Generate playground node documentation for user. Save to the root of the repo."""
    from .. import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    node_doc_path = pathlib.Path(__file__).parent.parent.parent.parent / "node.md"
    with open(node_doc_path, "w") as f:
        f.write("# Playground Nodes\n")
        for node_id, node_cls in NODE_CLASS_MAPPINGS.items():
            node_display_name = NODE_DISPLAY_NAME_MAPPINGS[node_id] if node_id in NODE_DISPLAY_NAME_MAPPINGS else None
            node_category = node_cls.CATEGORY
            node_description = node_cls.DESCRIPTION if hasattr(node_cls, "DESCRIPTION") else "No description."
            f.write(
                f"""
## {node_display_name}

ID : {node_id}  
CATEGORY : {node_category}  
DESCRIPTION : {node_description}  
"""
            )


@register_node(category="utils", output=True)
def free_vram():
    get_management(resources_device("cuda")).free(1e30)
