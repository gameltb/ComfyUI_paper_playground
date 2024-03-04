from typing import Annotated

import comfy.utils
import torch
import comfy.model_management

from .....common import path_tool
from .....paper.arxiv.abs2402_05054.core.models import LGM
from .....paper.arxiv.abs2402_05054.core.options import config_defaults
from .....pipelines.abs2402_05054 import LGMPipeline
from ....registry import register_node
from ....types import ComboWidget, ImageType, gen_simple_new_type
from ...plyfile import PlyDataType

LGMPipelineType = gen_simple_new_type(LGMPipeline, "LGM_PIPELINE")


@register_node(category="arxiv/abs2402_05054")
def load_lgm(
    ckpt_path: Annotated[str, ComboWidget(choices=lambda: path_tool.get_model_filename_list(__name__, ""))],
    lgb_config: Annotated[str, ComboWidget(choices=["big", "default", "small", "tiny"])] = "big",
) -> tuple[LGMPipelineType]:
    ckpt_path = path_tool.get_model_full_path(__name__, "", ckpt_path)

    lgm_model = LGM(config_defaults[lgb_config])

    ckpt = comfy.utils.load_torch_file(ckpt_path)

    lgm_model.load_state_dict(ckpt, strict=False)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lgm_model = lgm_model.half().to(device)
    lgm_model.eval()

    return (LGMPipeline(lgm_model),)


@register_node(category="arxiv/abs2402_05054")
def run_lgm(
    lgm_pipeline: LGMPipelineType,
    image: ImageType,
) -> tuple[PlyDataType]:
    comfy.model_management.free_memory(6 * 1024 * 1024 * 1024, torch.device("cuda", 0))
    input_image = image.cpu().float().numpy()
    return (lgm_pipeline(input_image),)
