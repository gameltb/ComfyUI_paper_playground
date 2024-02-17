from .registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, set_pack_options

set_pack_options('paper_playground', 'Playground Nodes')

from .node import diffusers
from .node.paper.arxiv import abs2312_02145, abs2312_13964