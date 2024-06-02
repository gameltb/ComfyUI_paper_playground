from typing import Annotated, Any, Callable, ClassVar, List, Literal, Mapping, Optional, Union

import comfy.clip_vision
import comfy.controlnet
import comfy.model_patcher
import comfy.sd
import torch
from pydantic import BaseModel

from .common import gen_widget

MaskType = Annotated[torch.Tensor, gen_widget("MASK")]
"""Tensor [B,H,W]"""

ImageType = Annotated[torch.Tensor, gen_widget("IMAGE")]
"""Tensor [B,H,W,C] float32 cpu"""

LatentType = Annotated[dict[str, torch.Tensor], gen_widget("LATENT")]
"""samples : Tensor [B,H,W,C]"""

VaeType = Annotated[comfy.sd.VAE, gen_widget("VAE")]

ModelType = Annotated[comfy.model_patcher.ModelPatcher, gen_widget("MODEL")]

SigmasType = Annotated[torch.Tensor, gen_widget("SIGMAS")]

ClipType = Annotated[comfy.sd.CLIP, gen_widget("CLIP")]

ConditioningType = Annotated[list[tuple[torch.Tensor, dict[str, torch.Tensor]]], gen_widget("CONDITIONING")]

ClipVisionType = Annotated[comfy.clip_vision.ClipVisionModel, gen_widget("CLIP_VISION")]

ControlNetType = Annotated[
    Union[comfy.controlnet.ControlLora, comfy.controlnet.ControlNet, comfy.controlnet.T2IAdapter],
    gen_widget("CONTROL_NET"),
]


class ReturnType(BaseModel):
    ui: dict[str, list[Any]] = dict()
    result: tuple = tuple()
