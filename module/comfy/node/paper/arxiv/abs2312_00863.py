import os
from typing import Annotated

import torch

import comfy.utils

from .....common import path_tool
from .....paper.arxiv.abs2312_00863.efficient_sam.build_efficient_sam import build_efficient_sam
from .....paper.arxiv.abs2312_00863.efficient_sam.efficient_sam import EfficientSam

from ....registry import register_node
from ....types import ComboWidget, ImageType, gen_simple_new_type
from .....core.runtime_resource_management import AutoManage
from ....registry import register_node
from ....types import ImageType, MaskType, gen_simple_new_type

EfficientSamModelType = gen_simple_new_type(EfficientSam, "EFFICIENT_SAM_MODEL")

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


@register_node(category="arxiv/abs2312_00863")
def load_efficient_sam(
    config: Annotated[dict, ComboWidget(choices=EFFICIENT_SAM_CONFIG)] = "efficient_sam_vitt",
) -> tuple[EfficientSamModelType]:
    config = config.copy()
    ckpt_path = os.path.join(path_tool.get_paper_repo_path(__name__), config.pop("checkpoint"))

    model = build_efficient_sam(**config).eval()

    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model.load_state_dict(ckpt["model"])

    return (model,)


@register_node(category="arxiv/abs2312_00863")
def run_efficient_sam(
    efficient_sam_model: EfficientSamModelType,
    image: ImageType,
) -> tuple[MaskType]:
    with AutoManage(efficient_sam_model) as am:
        input_points = torch.tensor([[[[580, 350], [650, 350]]]]).to(am.get_device())
        input_labels = torch.tensor([[[1, 1]]]).to(am.get_device())

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
