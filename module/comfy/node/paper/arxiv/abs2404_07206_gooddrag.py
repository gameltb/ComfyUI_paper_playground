import os
from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2404_07206.pipeline import GoodDragger
from ....registry import register_node
from ....types import (
    BoolType,
    FloatPercentageType,
    ImageType,
    IntStepsType,
    IntType,
    MaskType,
    StringMultilineType,
    StringType,
)
from ...diffusers import DiffusersPipelineType
from ...utils_image_annotate import ImageAnnotateType

DEFAULT_CATEGORY = path_tool.gen_default_category_path_by_module_name(__name__)


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def get_original_points(
    handle_points: List[torch.Tensor],
    full_h: int,
    full_w: int,
    sup_res_w,
    sup_res_h,
) -> List[torch.Tensor]:
    """
    Convert local handle points and target points back to their original UI coordinates.

    Args:
        sup_res_h: Half original height of the UI canvas.
        sup_res_w: Half original width of the UI canvas.
        handle_points: List of handle points in local coordinates.
        full_h: Original height of the UI canvas.
        full_w: Original width of the UI canvas.

    Returns:
        original_handle_points: List of handle points in original UI coordinates.
    """
    original_handle_points = []

    for cur_point in handle_points:
        original_point = torch.round(
            torch.tensor([cur_point[1] * full_w / sup_res_w, cur_point[0] * full_h / sup_res_h])
        )
        original_handle_points.append(original_point)

    return original_handle_points


@register_node(category=DEFAULT_CATEGORY)
def run_gooddrag(
    diffusers_pipeline: DiffusersPipelineType,
    image: ImageType,
    annotate: ImageAnnotateType,
    mask: MaskType,
    prompt: StringMultilineType = "",
    inversion_strength: FloatPercentageType = 0.75,
    lam: FloatPercentageType = 0.1,
    latent_lr: FloatPercentageType = 0.02,
    lora_path: StringType = "",
    drag_end_step: IntStepsType = 7,
    track_per_step: IntStepsType = 10,
    compare_mode: BoolType = False,
    r1: IntType = 4,
    r2: IntType = 12,
    d: IntType = 4,
    max_drag_per_track: IntType = 3,
    drag_loss_threshold: IntType = 0,
    once_drag: BoolType = False,
    max_track_no_change: IntType = 5,
    feature_idx: IntType = 3,
) -> tuple[ImageType]:
    height, width = image.shape[1:3]
    n_inference_step = 50
    guidance_scale = 1.0

    with torch.inference_mode(False):
        diffusers_pipeline = diffusers_pipeline.to("cuda")

        dragger = GoodDragger(
            diffusers_pipeline,
            prompt,
            height,
            width,
            inversion_strength,
            r1,
            r2,
            d,
            drag_end_step,
            track_per_step,
            lam,
            latent_lr,
            n_inference_step,
            guidance_scale,
            feature_idx,
            compare_mode,
            lora_path,
            max_drag_per_track,
            drag_loss_threshold,
            once_drag,
            max_track_no_change,
        )

        # source_image = preprocess_image(source_image, device)
        return_intermediate_images = False
        save_intermedia = False
        points = [[147, 67], [202, 46], [410, 88], [336, 54]]

        with AutoManage(dragger.model) as am:
            gen_image, intermediate_features, new_points_handle, intermediate_images = dragger.good_drag(
                image.permute(0, 3, 1, 2), points, mask[0], return_intermediate_images=return_intermediate_images
            )

    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename="drag_image")

    gen_image = F.interpolate(gen_image, (height, width), mode="bilinear")

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f"points {points}")
    print(f"new points {new_points}")

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode="bilinear")
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)

    return gen_image.cpu().permute(0, 2, 3, 1), new_points
