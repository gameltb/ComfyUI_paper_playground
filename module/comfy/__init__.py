import importlib
import os
import traceback

from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, set_pack_options

set_pack_options("paper_playground", "playground")

from .node import diffusers

BASE_DIR = os.path.dirname(__file__)
ARXIV_NODE_DIR = os.path.join(BASE_DIR, "node", "paper", "arxiv")

for file_name in os.listdir(ARXIV_NODE_DIR):
    try:
        base_name, ext = os.path.splitext(file_name)
        if ext == ".py":
            importlib.import_module(f".node.paper.arxiv.{base_name}", package=__package__)
    except Exception as e:
        traceback.print_exc()
        print(f"Cannot import {file_name} module :", e)
