import torch

import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .....paper.arxiv.abs2312_02145.marigold import MarigoldPipeline
from ....registry import register_node
from ...diffusers import DiffusersPipelineFromPretrained


def resize_max_res(img, max_edge_resolution: int):
    B, C, original_height, original_width = img.shape
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = comfy.utils.common_upscale(
        img, new_width, new_height, "nearest-exact", ""
    )
    return resized_img

@register_node(display_name="Abs 2312.02145 Diffusers Pipeline From Pretrained")
class Abs2312_02145_DiffusersPipelineFromPretrained(DiffusersPipelineFromPretrained):
    @classmethod
    def INPUT_TYPES(s):
        input_types = super().INPUT_TYPES()
        input_types["required"].pop("pipeline_type")
        return input_types

    FUNCTION = "abs2312_02145_from_pretrained"

    CATEGORY = "playground/arxiv/abs2312_02145"

    def abs2312_02145_from_pretrained(
        self, local_files_only, directory=None, model_id=None
    ):
        pipeline_cls = MarigoldPipeline

        return self.from_pretrained(
            pipeline_cls, local_files_only, directory=directory, model_id=model_id
        )

@register_node(display_name="Abs 2312.02145 Diffusers Pipeline Sampler")
class Abs2312_02145_DiffusersPipelineSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "max_resolution": ("INT", {"default": 768, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_pipeline"

    CATEGORY = "playground/arxiv/abs2312_02145"

    def do_pipeline(
        self,
        diffusers_pipeline,
        image,
        seed,
        steps,
        max_resolution,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: MarigoldPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )

        comfy.model_management.load_models_gpu([pipeline_comfy_model_patcher_wrapper])

        image = image.permute(0, 3, 1, 2)
        image = resize_max_res(image, max_resolution).to(
            device=diffusers_pipeline.device, dtype=diffusers_pipeline.dtype
        )

        generator = torch.Generator(device=diffusers_pipeline.device)
        generator.manual_seed(seed)

        pbar = comfy.utils.ProgressBar(steps)

        def callback_on_step_end(i, t, callback_kwargs):
            pbar.update(i)
            return {}

        # Predict depth
        depth_pred = diffusers_pipeline.single_infer(
            rgb_in=image, num_inference_steps=steps, show_pbar=True, generator=generator
        )

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        return (depth_pred[0],)
