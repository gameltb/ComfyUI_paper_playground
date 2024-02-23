import os

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from .....common import path_tool
from .....paper.arxiv.abs2304_05690.hybrik.models import builder
from .....paper.arxiv.abs2304_05690.hybrik.utils.config import update_config
from .....paper.arxiv.abs2304_05690.hybrik.utils.presets import SimpleTransform3DSMPLX
from ....registry import register_node
from ....types import Combo, Bool, String, ComfyWidgetType, IMAGE, Int, Float, LATENT
from .....pipelines.abs2304_05690 import HybrikXPipeline
from .....paper.arxiv.abs2304_05690.hybrik.utils.easydict import EasyDict
from .....utils.json import np_dump


SMPLX_CONFIG_PATH = os.path.join(path_tool.get_paper_repo_path(__name__), "configs/smplx")
SMPLX_CONFIG_FILES = {filename: os.path.join(SMPLX_CONFIG_PATH, filename) for filename in os.listdir(SMPLX_CONFIG_PATH)}

HYBRIKX_MODEL_PATH = path_tool.get_model_filename_list(__name__, "hybrikx")


class HYBRIKX_PIPELINE(ComfyWidgetType):
    TYPE = "HYBRIKX_PIPELINE"


class HYBRIKX_FRAME(ComfyWidgetType):
    TYPE = "HYBRIKX_FRAME"


@register_node(category="arxiv/abs2304_05690")
def load_hybrikx(
    cfg_file_path: Combo(choices=SMPLX_CONFIG_FILES),
    ckpt_path: Combo(
        choices=lambda: {
            p: path_tool.get_model_full_path(__name__, "hybrikx", p)
            for p in path_tool.get_model_filename_list(__name__, "hybrikx")
        }
    ),
) -> (HYBRIKX_PIPELINE(),):
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

    det_model.cuda()
    hybrik_model.cuda()
    det_model.eval()
    hybrik_model.eval()

    return (HybrikXPipeline(transformation, det_model, hybrik_model),)


@register_node(category="arxiv/abs2304_05690")
def run_hybrikx(
    hybrikx_pipeline: HYBRIKX_PIPELINE(),
    image: IMAGE(),
) -> (HYBRIKX_FRAME(),):
    input_image = image.cpu().float().numpy()[0]
    return (hybrikx_pipeline(input_image),)


@register_node(category="arxiv/abs2304_05690", output=True)
def save_hybrikx(
    hybrikx_frame: HYBRIKX_FRAME(),
    path: String() = "hybrikx_frame.json",
) -> tuple():
    with open(path, "w") as f:
        np_dump(hybrikx_frame, f)
    return {}
