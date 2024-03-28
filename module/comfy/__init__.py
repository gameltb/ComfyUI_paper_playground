import importlib
import os
import traceback

from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, set_pack_options

set_pack_options("paper_playground", "playground")

from .node import diffusers, plyfile, utils, utils_image_annotate

BASE_DIR = os.path.dirname(__file__)

for node_type in ["arxiv", "github"]:
    node_dir = os.path.join(BASE_DIR, "node", "paper", node_type)

    for file_name in os.listdir(node_dir):
        try:
            base_name, ext = os.path.splitext(file_name)
            if ext == ".py":
                importlib.import_module(f".node.paper.{node_type}.{base_name}", package=__package__)
        except Exception:
            traceback.print_exc()
try:
    from .node import rmbg
except Exception:
    traceback.print_exc()
