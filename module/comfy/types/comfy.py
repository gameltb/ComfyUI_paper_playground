from typing import Annotated, Any, Optional, TypedDict, Union

import comfy.clip_vision
import comfy.controlnet
import comfy.model_patcher
import comfy.sd
import comfy_execution.graph
import torch
from pydantic import BaseModel

from .common import ComfyWidgetInputType, gen_widget

MaskType = Annotated[torch.Tensor, gen_widget("MASK")]
"""Tensor [B,H,W] 0~1"""

ImageType = Annotated[torch.Tensor, gen_widget("IMAGE")]
"""Tensor [B,H,W,C] 0~1"""


class LatentTypeDict(TypedDict, total=False):
    samples: torch.Tensor
    """[B,C,H,W] 0~1"""
    batch_index: torch.Tensor
    """[B]"""
    noise_mask: torch.Tensor
    """[B,C,H,W] 0~1"""


LatentType = Annotated[LatentTypeDict, gen_widget("LATENT")]
"""samples batch_index noise_mask"""

VaeType = Annotated[comfy.sd.VAE, gen_widget("VAE")]

ModelType = Annotated[comfy.model_patcher.ModelPatcher, gen_widget("MODEL")]

SigmasType = Annotated[torch.Tensor, gen_widget("SIGMAS")]

ClipType = Annotated[comfy.sd.CLIP, gen_widget("CLIP")]


class ConditioningAttrTypeDict(TypedDict, total=False):
    pooled_output: torch.Tensor
    area: Union[tuple[int, int, int, int], tuple[str, int, int, int, int]]
    strength: float
    set_area_to_bounds: bool
    mask: MaskType
    mask_strength: float
    start_percent: float
    end_percent: float
    concat_latent_image: torch.Tensor
    """[B,C,H,W] 0~1"""
    concat_mask: MaskType
    prompt_type: str


ConditioningType = Annotated[list[tuple[torch.Tensor, ConditioningAttrTypeDict]], gen_widget("CONDITIONING")]
"""list of [cond, cond_attr_dict] """

ClipVisionType = Annotated[comfy.clip_vision.ClipVisionModel, gen_widget("CLIP_VISION")]

ControlNetType = Annotated[
    Union[comfy.controlnet.ControlLora, comfy.controlnet.ControlNet, comfy.controlnet.T2IAdapter],
    gen_widget("CONTROL_NET"),
]

PromptType = Annotated[
    dict,
    gen_widget("PROMPT", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
DynPromptType = Annotated[
    comfy_execution.graph.DynamicPrompt,
    gen_widget("DYNPROMPT", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
ExtraPnginfoType = Annotated[
    Optional[dict],
    gen_widget("EXTRA_PNGINFO", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
UniqueIdType = Annotated[
    str,
    gen_widget("UNIQUE_ID", widget_input_type=ComfyWidgetInputType.HIDDEN),
]


class ReturnType(BaseModel, arbitrary_types_allowed=True):
    ui: Optional[dict[str, list[Any]]] = None
    expand: Optional[dict] = None
    result: Union[tuple, comfy_execution.graph.ExecutionBlocker] = tuple()
