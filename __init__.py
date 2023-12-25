import traceback 
import sys 

WEB_DIRECTORY = "web"

NODE_CLASS_MAPPINGS = {}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {}

from .module.comfy.node import diffusers
NODE_CLASS_MAPPINGS.update(diffusers.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(diffusers.NODE_DISPLAY_NAME_MAPPINGS)

from .module.comfy.node.paper.arxiv import abs2312_13964
NODE_CLASS_MAPPINGS.update(abs2312_13964.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(abs2312_13964.NODE_DISPLAY_NAME_MAPPINGS)

if len(NODE_CLASS_MAPPINGS) == 0:
    raise Exception("import failed")