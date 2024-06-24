import comfy.utils
import torch

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2312_02145.marigold import MarigoldPipeline
from ....registry import register_node
from ....types import ComboWidget
from ...diffusers import DiffusersPipelineType, diffusers_from_pretrained_cls, get_diffusers_folder_paths

DEFAULT_CATEGORY = path_tool.gen_default_category_path_by_module_name(__name__)


def resize_max_res(img, max_edge_resolution: int):
    B, C, original_height, original_width = img.shape
    downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = comfy.utils.common_upscale(img, new_width, new_height, "nearest-exact", "")
    return resized_img


@register_node(
    identifier="Abs2312_02145_DiffusersPipelineFromPretrained",
    display_name="Abs 2312.02145 Diffusers Pipeline From Pretrained (Marigold)",
    category=DEFAULT_CATEGORY,
)
def abs2312_02145_from_pretrained(
    directory: ComboWidget(choices=lambda: get_diffusers_folder_paths()) = None,
) -> tuple[DiffusersPipelineType]:
    pipeline_cls = MarigoldPipeline
    return diffusers_from_pretrained_cls(pipeline_cls, True, directory=directory, model_id=None)


@register_node(display_name="Abs 2312.02145 Diffusers Pipeline Sampler (Marigold)")
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

    CATEGORY = f"playground/{DEFAULT_CATEGORY}"

    def do_pipeline(
        self,
        diffusers_pipeline,
        image,
        seed,
        steps,
        max_resolution,
    ):
        with AutoManage(diffusers_pipeline) as am:
            image = image.permute(0, 3, 1, 2)
            image = resize_max_res(image, max_resolution).to(device=am.get_device(), dtype=diffusers_pipeline.dtype)

            generator = torch.Generator(device=am.get_device())
            generator.manual_seed(seed)

            pbar = comfy.utils.ProgressBar(steps)

            def callback_on_step_end(i, t, callback_kwargs):
                pbar.update_absolute(i)
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
