import sys
import traceback

WEB_DIRECTORY = "web"

try:
    from .module.comfy.node import diffusers
    from .module.comfy.node.paper.arxiv import abs2312_02145, abs2312_13964
except Exception as e:
    raise e

from .module.comfy.utils import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
