import os
from typing import Annotated

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from .....common import path_tool
from .....paper.arxiv.abs2304_05690.hybrik.models import builder
from .....paper.arxiv.abs2304_05690.hybrik.utils.config import update_config
from .....paper.arxiv.abs2304_05690.hybrik.utils.easydict import EasyDict
from .....paper.arxiv.abs2304_05690.hybrik.utils.presets import SimpleTransform3DSMPLX
from .....pipelines.abs2304_05690 import HybrikXPipeline
from .....utils.json import np_dumps
from ....registry import register_node
from ....types import ComboWidget, ImageType, StringType, gen_widget

SMPLX_CONFIG_PATH = os.path.join(path_tool.get_paper_repo_path(__name__), "configs/smplx")
SMPLX_CONFIG_FILES = {filename: os.path.join(SMPLX_CONFIG_PATH, filename) for filename in os.listdir(SMPLX_CONFIG_PATH)}


HybrikxPipelineType = Annotated[HybrikXPipeline, gen_widget("HYBRIKX_PIPELINE")]
HybrikxFrameType = Annotated[dict, gen_widget("HYBRIKX_FRAME")]


@register_node(category="arxiv/abs2304_05690")
def load_hybrikx(
    cfg_file_path: Annotated[str, ComboWidget(choices=SMPLX_CONFIG_FILES)],
    ckpt_path: Annotated[str, ComboWidget(choices=lambda: path_tool.get_model_filename_list(__name__, "hybrikx"))],
) -> tuple[HybrikxPipelineType]:
    ckpt_path = path_tool.get_model_full_path(__name__, "hybrikx", ckpt_path)

    cfg = update_config(cfg_file_path)

    cfg["MODEL"]["EXTRA"]["USE_KID"] = cfg["DATASET"].get("USE_KID", False)
    cfg["LOSS"]["ELEMENTS"]["USE_KID"] = cfg["DATASET"].get("USE_KID", False)

    bbox_3d_shape = cfg["MODEL"].get("BBOX_3D_SHAPE", (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummpy_set = EasyDict(
        {
            "joint_pairs_17": None,
            "joint_pairs_24": None,
            "joint_pairs_29": None,
            "bbox_3d_shape": bbox_3d_shape,
        }
    )

    transformation = SimpleTransform3DSMPLX(
        dummpy_set,
        scale_factor=cfg["DATASET"]["SCALE_FACTOR"],
        color_factor=cfg["DATASET"]["COLOR_FACTOR"],
        occlusion=cfg["DATASET"]["OCCLUSION"],
        input_size=cfg["MODEL"]["IMAGE_SIZE"],
        output_size=cfg["MODEL"]["HEATMAP_SIZE"],
        depth_dim=cfg["MODEL"]["EXTRA"]["DEPTH_DIM"],
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg["DATASET"]["ROT_FACTOR"],
        sigma=cfg["MODEL"]["EXTRA"]["SIGMA"],
        train=False,
        add_dpg=False,
        loss_type=cfg["LOSS"]["TYPE"],
    )

    det_model = fasterrcnn_resnet50_fpn(pretrained=True)

    hybrik_model = builder.build_sppe(cfg["MODEL"])

    print(f"Loading model from {ckpt_path}...")
    save_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if type(save_dict) == dict:
        model_dict = save_dict["model"]
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    det_model.eval()
    hybrik_model.eval()

    return (HybrikXPipeline(transformation, det_model, hybrik_model),)


@register_node(category="arxiv/abs2304_05690")
def run_hybrikx(
    hybrikx_pipeline: HybrikxPipelineType,
    image: ImageType,
) -> tuple[HybrikxFrameType]:
    input_image = image.cpu().float().numpy()[0]
    return (hybrikx_pipeline(input_image),)


@register_node(category="arxiv/abs2304_05690")
def hybrikx_to_string(
    hybrikx_frame: HybrikxFrameType,
) -> tuple[StringType]:
    return (np_dumps(hybrikx_frame),)
