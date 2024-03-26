from typing import Annotated

import comfy.utils

from .....common import path_tool
from .....paper.arxiv.abs2402_05054.core.models import LGM
from .....paper.arxiv.abs2402_05054.core.options import config_defaults
from .....pipelines.abs2402_05054 import LGMPipeline
from ....registry import register_node
from ....types import ComboWidget, ImageType, gen_widget
from ...plyfile import PlyDataType

LGMPipelineType = Annotated[LGMPipeline, gen_widget("LGM_PIPELINE")]


@register_node(category="arxiv/abs2402_05054")
def load_lgm(
    ckpt_path: Annotated[str, ComboWidget(choices=lambda: path_tool.get_model_filename_list(__name__, ""))],
    lgb_config: Annotated[str, ComboWidget(choices=["big", "default", "small", "tiny"])] = "big",
) -> tuple[LGMPipelineType]:
    ckpt_path = path_tool.get_model_full_path(__name__, "", ckpt_path)

    lgm_model = LGM(config_defaults[lgb_config])

    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    lgm_model.load_state_dict(ckpt, strict=False)

    # device
    lgm_model = lgm_model.half()
    lgm_model.eval()

    return (LGMPipeline(lgm_model),)


@register_node(category="arxiv/abs2402_05054")
def run_lgm(
    lgm_pipeline: LGMPipelineType,
    image: ImageType,
) -> tuple[PlyDataType]:
    input_image = image.cpu().float().numpy()
    return (lgm_pipeline(input_image),)
