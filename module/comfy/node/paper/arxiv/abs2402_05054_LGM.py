import os
from typing import Annotated

import comfy.utils
import numpy as np
import torch

from .....common import file_get_tool, path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2402_05054.core.models import LGM
from .....paper.arxiv.abs2402_05054.core.options import config_defaults
from .....paper.arxiv.abs2402_05054.mvdream.pipeline_mvdream import MVDreamPipeline
from .....pipelines.abs2402_05054 import LGMPipeline
from ....registry import register_node
from ....types import (
    ComboWidget,
    FloatCFGType,
    ImageType,
    IntSeedType,
    IntStepsType,
    IntType,
    MaskType,
    StringMultilineType,
    gen_widget,
)
from ...plyfile import PlyDataType

DEFAULT_CATEGORY = path_tool.gen_default_category_path_by_module_name(__name__)

LGMPipelineType = Annotated[LGMPipeline, gen_widget("LGM_PIPELINE")]


@register_node(category=DEFAULT_CATEGORY)
def load_lgm(
    lgb_config: Annotated[str, ComboWidget(choices=["big", "default", "small", "tiny"])] = "big",
) -> tuple[LGMPipelineType]:
    model_path = file_get_tool.find_or_download_huggingface_repo(
        [
            file_get_tool.FileSource(
                loacal_folder=path_tool.get_data_path(__name__),
            ),
            file_get_tool.FileSourceHuggingface(repo_id="ashawkey/LGM"),
        ]
    )
    ckpt_path = os.path.join(model_path, "model_fixrot.safetensors")

    lgm_model = LGM(config_defaults[lgb_config])

    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    lgm_model.load_state_dict(ckpt, strict=False)

    # device
    lgm_model = lgm_model.half()
    lgm_model.eval()

    return (LGMPipeline(lgm_model),)


@register_node(category=DEFAULT_CATEGORY)
def run_lgm(
    lgm_pipeline: LGMPipelineType,
    image: ImageType,
) -> tuple[PlyDataType]:
    input_image = image.cpu().float().numpy()
    return (lgm_pipeline(input_image),)


LGMMvdreamPipelineType = Annotated[MVDreamPipeline, gen_widget("LGM_MVDREAM_PIPELINE")]


@register_node(category=DEFAULT_CATEGORY)
def load_lgm_mvdream() -> tuple[LGMMvdreamPipelineType]:
    model_path = file_get_tool.find_or_download_huggingface_repo(
        [
            file_get_tool.FileSource(
                loacal_folder=path_tool.get_data_path(__name__),
            ),
            file_get_tool.FileSourceHuggingface(repo_id="ashawkey/imagedream-ipmv-diffusers"),
        ]
    )

    lgm_mvdream_model = MVDreamPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    return (lgm_mvdream_model,)


@register_node(category=DEFAULT_CATEGORY)
def run_lgm_mvdream(
    lgm_mvdream_pipeline: LGMMvdreamPipelineType,
    image: ImageType,
    mask: MaskType,
    prompt: StringMultilineType,
    prompt_neg: StringMultilineType = "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate",
    seed: IntSeedType = 0,
    mv_guidance_scale: FloatCFGType = 5,
    num_inference_steps: IntStepsType = 30,
    elevation: IntType = 0,
) -> tuple[ImageType]:
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)

    generator = torch.manual_seed(seed)

    mask = mask.unsqueeze(2)
    # give the white background to reference_image
    reference_image = (image * mask + (1 - mask)).detach().cpu().numpy()

    # generate multi-view images
    with AutoManage(lgm_mvdream_pipeline) as am:
        mv_images = lgm_mvdream_pipeline(
            prompt,
            reference_image,
            generator=generator,
            negative_prompt=prompt_neg,
            guidance_scale=mv_guidance_scale,
            num_inference_steps=num_inference_steps,
            elevation=elevation,
        )
    mv_images = torch.from_numpy(
        np.stack([mv_images[1], mv_images[2], mv_images[3], mv_images[0]], axis=0)
    ).float()  # [4, H, W, 3], float32

    return (mv_images,)
