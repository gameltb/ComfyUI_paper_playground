import os

import cv2
import numpy as np
import torch
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import ImageType, IntType, MaskType, gen_simple_new_type


def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated == 255] = 128
    trimap[eroded == 255] = 255
    return trimap


class VitMattePipeline(PlaygroundPipeline):
    def __init__(self, processor: VitMatteImageProcessor, model: VitMatteForImageMatting) -> None:
        super().__init__()

        self.register_modules(
            processor=processor,
            model=model,
        )
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def __call__(self, image, trimap):
        with AutoManage(self.model) as am:
            device = am.get_device()
            inputs = self.processor(images=image, trimaps=trimap, return_tensors="pt")
            alphas = self.model(**{k: v.to(device) for k, v in inputs.items()}).alphas
        return alphas


VitMattePipelineType = gen_simple_new_type(VitMattePipeline, "VIT_MATTE_PIPELINE")


@register_node(category="arxiv/abs2305_15272")
def load_vit_matte_pipeline() -> tuple[VitMattePipelineType]:
    repo_id = "hustvl/vitmatte-small-composition-1k"
    local_path = path_tool.get_local_huggingface_path(repo_id)
    if os.path.exists(local_path):
        processor = VitMatteImageProcessor.from_pretrained(local_path, local_files_only=True)
        model = VitMatteForImageMatting.from_pretrained(local_path, local_files_only=True)
    else:
        processor = VitMatteImageProcessor.from_pretrained(repo_id)
        model = VitMatteForImageMatting.from_pretrained(repo_id)

    return (VitMattePipeline(processor, model),)


@register_node(category="arxiv/abs2305_15272")
def run_vit_matte_pipeline(
    vit_matte_pipeline: VitMattePipelineType,
    image: ImageType,
    mask: MaskType,
    trimap_erode: IntType = 10,
    trimap_dilate: IntType = 10,
) -> tuple[MaskType]:
    np_mask = (torch.ge(mask[0], 0.5).cpu().detach().numpy() * 255).astype(np.uint8)
    trimap: np.ndarray = generate_trimap(np_mask, trimap_erode, trimap_dilate)

    alphas = vit_matte_pipeline(image.cpu().detach() * 255, trimap)

    return (alphas.squeeze(1)[:, : image.shape[1], : image.shape[2]],)
