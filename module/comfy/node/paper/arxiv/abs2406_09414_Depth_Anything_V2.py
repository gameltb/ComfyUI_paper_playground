import os
from typing import Annotated

import comfy.utils
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from .....common import file_get_tool, path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2406_09414_Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from .....paper.arxiv.abs2406_09414_Depth_Anything_V2.depth_anything_v2.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from .....paper.arxiv.abs2406_09414_Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import (
    DepthAnythingV2 as DepthAnythingV2Metric,
)
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import (
    ComboWidget,
    ImageType,
    make_widget,
)

DEFAULT_CATEGORY = path_tool.gen_default_category_path_by_module_name(__name__)


class DepthAnythingV2Pipeline(PlaygroundPipeline):
    def __init__(self, model: DepthAnythingV2) -> None:
        super().__init__()
        self.register_modules(
            model=model,
        )
        self.model = model

    @torch.no_grad()
    def __call__(self, image, input_size=518):
        with AutoManage(self.model) as am:
            device = am.get_device()

            transform = Compose(
                [
                    Resize(
                        width=input_size,
                        height=input_size,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ]
            )

            h, w = image.shape[:2]

            image = transform({"image": image})["image"]
            image = torch.from_numpy(image).unsqueeze(0)

            image = image.to(device)

            depth = self.model(image)

            depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

        return depth


DepthAnythingV2PipelineType = Annotated[DepthAnythingV2Pipeline, make_widget("DEPTH_ANYTHING_V2_PIPELINE")]
DepthAnythingV2DepthType = Annotated[torch.Tensor, make_widget("DEPTH_ANYTHING_V2_DEPTH")]
"""Tensor [B,H,W]"""

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

repo_config_map = {
    "depth-anything/Depth-Anything-V2-Large": {
        "model_config": "vitl",
        "model_name": "depth_anything_v2_vitl.pth",
        "cls": DepthAnythingV2,
        "ext_configs": {},
    },
    "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large": {
        "model_config": "vitl",
        "model_name": "depth_anything_v2_metric_hypersim_vitl.pth",
        "cls": DepthAnythingV2Metric,
        "ext_configs": {"max_depth": 20},
    },
    "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large": {
        "model_config": "vitl",
        "model_name": "depth_anything_v2_metric_vkitti_vitl.pth",
        "cls": DepthAnythingV2Metric,
        "ext_configs": {"max_depth": 80},
    },
}


@register_node(category=DEFAULT_CATEGORY)
def load_DepthAnythingV2(
    repo_id: Annotated[str, ComboWidget(choices=list(repo_config_map.keys()))],
) -> tuple[DepthAnythingV2PipelineType]:
    model_path = file_get_tool.find_or_download_huggingface_repo(
        [
            file_get_tool.FileSourceHuggingface(repo_id=repo_id),
        ]
    )
    repo_config = repo_config_map[repo_id]
    ckpt_path = os.path.join(model_path, repo_config["model_name"])

    model = repo_config["cls"](**model_configs[repo_config["model_config"]], **repo_config["ext_configs"])

    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model.load_state_dict(ckpt)
    model.eval()

    return (DepthAnythingV2Pipeline(model),)


@register_node(category=DEFAULT_CATEGORY)
def run_DepthAnythingV2(
    DepthAnythingV2_pipeline: DepthAnythingV2PipelineType,
    image: ImageType,
) -> tuple[ImageType, DepthAnythingV2DepthType]:
    depth = DepthAnythingV2_pipeline(image[0].numpy())
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_image = (
        depth_norm.reshape((-1, 1, depth_norm.shape[-2], depth_norm.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    )
    return (depth_image, depth.unsqueeze(0))
