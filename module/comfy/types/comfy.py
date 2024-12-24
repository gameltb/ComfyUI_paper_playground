import typing
from typing import Annotated, Any, Optional, TypedDict, Union

import torch
from pydantic import BaseModel

from .common import ComfyWidgetInputType, make_widget

COMFYUI = True

try:
    import comfy
except ModuleNotFoundError:
    COMFYUI = False

if typing.TYPE_CHECKING or COMFYUI:
    import comfy.clip_vision
    import comfy.controlnet
    import comfy.model_patcher
    import comfy.samplers
    import comfy.sd
    import comfy_execution.graph
else:

    class comfy:
        class clip_vision:
            class ClipVisionModel: ...

        class controlnet:
            class ControlLora: ...

            class ControlNet: ...

            class T2IAdapter: ...

        class model_patcher:
            class ModelPatcher: ...

        class sd:
            class VAE: ...

            class CLIP: ...

        class samplers:
            class KSAMPLER: ...

    class comfy_execution:
        class graph:
            class ExecutionBlocker: ...

            class DynamicPrompt: ...


MaskType = Annotated[torch.Tensor, make_widget("MASK")]
"""Tensor [B,H,W] 0~1"""

ImageType = Annotated[torch.Tensor, make_widget("IMAGE")]
"""Tensor [B,H,W,C] 0~1"""


class LatentTypeDict(TypedDict, total=False):
    samples: torch.Tensor
    """[B,C,H,W] 0~1"""
    batch_index: torch.Tensor
    """[B]"""
    noise_mask: torch.Tensor
    """[B,C,H,W] 0~1"""


LatentType = Annotated[LatentTypeDict, make_widget("LATENT")]
"""samples batch_index noise_mask"""

VaeType = Annotated[comfy.sd.VAE, make_widget("VAE")]

ModelType = Annotated[comfy.model_patcher.ModelPatcher, make_widget("MODEL")]

SigmasType = Annotated[torch.Tensor, make_widget("SIGMAS")]

SamplerType = Annotated[comfy.samplers.KSAMPLER, make_widget("SAMPLER")]

ClipType = Annotated[comfy.sd.CLIP, make_widget("CLIP")]


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


ConditioningType = Annotated[list[tuple[torch.Tensor, ConditioningAttrTypeDict]], make_widget("CONDITIONING")]
"""list of [cond, cond_attr_dict] """

ClipVisionType = Annotated[comfy.clip_vision.ClipVisionModel, make_widget("CLIP_VISION")]

ControlNetType = Annotated[
    Union[comfy.controlnet.ControlLora, comfy.controlnet.ControlNet, comfy.controlnet.T2IAdapter],
    make_widget("CONTROL_NET"),
]

PromptType = Annotated[
    dict,
    make_widget("PROMPT", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
DynPromptType = Annotated[
    comfy_execution.graph.DynamicPrompt,
    make_widget("DYNPROMPT", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
ExtraPnginfoType = Annotated[
    Optional[dict],
    make_widget("EXTRA_PNGINFO", widget_input_type=ComfyWidgetInputType.HIDDEN),
]
UniqueIdType = Annotated[
    str,
    make_widget("UNIQUE_ID", widget_input_type=ComfyWidgetInputType.HIDDEN),
]


class ReturnType(BaseModel, arbitrary_types_allowed=True):
    ui: Optional[dict[str, list[Any]]] = None
    expand: Optional[dict] = None
    result: Union[tuple, comfy_execution.graph.ExecutionBlocker] = tuple()
