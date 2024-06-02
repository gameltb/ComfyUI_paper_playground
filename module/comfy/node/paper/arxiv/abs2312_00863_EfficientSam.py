import os
from typing import Annotated

import comfy.utils
import torch

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2312_00863.efficient_sam.build_efficient_sam import build_efficient_sam
from .....paper.arxiv.abs2312_00863.efficient_sam.efficient_sam import EfficientSam
from ...utils_image_annotate import ImageAnnotateType
from ....registry import register_node
from ....types import ComboWidget, ImageType, MaskType, gen_widget

EfficientSamModelType = Annotated[EfficientSam, gen_widget("EFFICIENT_SAM_MODEL")]

EFFICIENT_SAM_CONFIG = {
    "efficient_sam_vitt": {
        "encoder_patch_embed_dim": 192,
        "encoder_num_heads": 3,
        "checkpoint": "weights/efficient_sam_vitt.pt",
    },
    "efficient_sam_vits": {
        "encoder_patch_embed_dim": 384,
        "encoder_num_heads": 6,
        "checkpoint": "weights/efficient_sam_vits.pt",
    },
}


def parse_annotate(annotates: ImageAnnotateType):
    input_points = []
    input_labels = []
    for annotate in annotates:
        if annotate.type == 1:
            input_points.append(annotate.coor[0])
            input_labels.append(2)
            input_points.append(annotate.coor[1])
            input_labels.append(3)
        elif annotate.type == 3:
            input_points.append(annotate.coor)
            input_labels.append(1)

    return input_points, input_labels


@register_node(category="arxiv/abs2312_00863")
def load_efficient_sam(
    config: Annotated[dict, ComboWidget(choices=EFFICIENT_SAM_CONFIG)] = "efficient_sam_vitt",
) -> tuple[EfficientSamModelType]:
    config = config.copy()
    ckpt_path = path_tool.get_paper_repo_path(__name__, config.pop("checkpoint"))

    model = build_efficient_sam(**config).eval()

    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model.load_state_dict(ckpt["model"])

    return (model,)


@register_node(category="arxiv/abs2312_00863")
def run_efficient_sam(
    efficient_sam_model: EfficientSamModelType,
    image: ImageType,
    annotate: ImageAnnotateType,
) -> tuple[MaskType]:
    with AutoManage(efficient_sam_model) as am:
        pts_sampled, pts_labels = parse_annotate(annotate)
        input_points = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2]).to(am.get_device())
        input_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1]).to(am.get_device())

        predicted_logits, predicted_iou = efficient_sam_model.forward(
            image.permute(0, 3, 1, 2).to(am.get_device()),
            input_points,
            input_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(predicted_logits, sorted_ids[..., None, None], dim=2)
    # The masks are already sorted by their predicted IOUs.
    # The first dimension is the batch size (we have a single image. so it is 1).
    # The second dimension is the number of masks we want to generate (in this case, it is only 1)
    # The third dimension is the number of candidate masks output by the model.
    # For this demo we use the first mask.
    return (predicted_logits[0, 0],)
