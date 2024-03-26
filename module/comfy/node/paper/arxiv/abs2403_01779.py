import copy
from typing import Annotated

import comfy.utils
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, UniPCMultistepScheduler

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2403_01779 import pipelines
from .....paper.arxiv.abs2403_01779.garment_seg.process import generate_mask, load_seg_model
from .....paper.arxiv.abs2403_01779.utils.utils import is_torch2_available, prepare_image, prepare_mask
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import (
    ComboWidget,
    FloatCFGType,
    FloatPercentageType,
    ImageType,
    IntSeedType,
    IntStepsType,
    IntType,
    MaskType,
    StringMultilineType,
    gen_widget,
)
from ...diffusers import (
    DIFFUSERS_SCHEDULER_CLASS_MAP,
    DiffusersPipelineType,
)

if is_torch2_available():
    from .....paper.arxiv.abs2403_01779.garment_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
    from .....paper.arxiv.abs2403_01779.garment_adapter.attention_processor import (
        REFAnimateDiffAttnProcessor2_0 as REFAnimateDiffAttnProcessor,
    )
    from .....paper.arxiv.abs2403_01779.garment_adapter.attention_processor import (
        REFAttnProcessor2_0 as REFAttnProcessor,
    )
else:
    from .....paper.arxiv.abs2403_01779.garment_adapter.attention_processor import AttnProcessor, REFAttnProcessor


class OmsDiffusionPipeline(PlaygroundPipeline):
    def __init__(self, base_pipeline: StableDiffusionPipeline, ref_unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.attn_store = {}

        self.register_modules(base_pipeline=base_pipeline, ref_unet=ref_unet)
        self.base_pipeline = base_pipeline
        self.ref_unet = ref_unet

    @classmethod
    def from_base_pipeline(cls, sd_pipe: StableDiffusionPipeline, ref_path):
        sd_pipe_unet: UNet2DConditionModel = sd_pipe.components.get("unet")
        cls.set_adapter(sd_pipe_unet, "write")

        config = copy.deepcopy(sd_pipe_unet.config)
        ref_unet = sd_pipe_unet.__class__.from_config(config)

        if ref_unet.config.in_channels == 9:
            ref_unet.conv_in = torch.nn.Conv2d(
                4, 320, ref_unet.conv_in.kernel_size, ref_unet.conv_in.stride, ref_unet.conv_in.padding
            )
            ref_unet.register_to_config(in_channels=4)

        state_dict = comfy.utils.load_torch_file(ref_path, safe_load=True)
        ref_unet.load_state_dict(state_dict, strict=False)
        ref_unet.to(dtype=sd_pipe.dtype)

        cls.set_adapter(ref_unet, "read")

        return OmsDiffusionPipeline(pipelines.OmsDiffusionPipeline(**sd_pipe.components), ref_unet)

    @classmethod
    def set_adapter(cls, unet: UNet2DConditionModel, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def generate(
        self,
        cloth_image,
        cloth_mask_image,
        prompt=None,
        a_prompt="best quality, high quality",
        num_images_per_prompt=4,
        negative_prompt=None,
        seed=-1,
        guidance_scale=7.5,
        cloth_guidance_scale=2.5,
        num_inference_steps=20,
        height=512,
        width=384,
        **kwargs,
    ):
        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with AutoManage(self.base_pipeline) as am:
            cloth = (cloth.to(am.get_device()) * cloth_mask.to(am.get_device())).to(dtype=torch.float16)
            with torch.inference_mode():
                prompt_embeds, negative_prompt_embeds = self.base_pipeline.encode_prompt(
                    prompt,
                    device=self.device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                prompt_embeds_null = self.base_pipeline.encode_prompt(
                    [""],
                    device=self.device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=False,
                )[0]
                cloth_embeds = (
                    self.base_pipeline.vae.encode(cloth).latent_dist.mode()
                    * self.base_pipeline.vae.config.scaling_factor
                )
                with AutoManage(self.ref_unet) as ref_am:
                    prompt_embeds_null = prompt_embeds_null.to(ref_am.get_device())
                    self.ref_unet(
                        torch.cat([cloth_embeds] * num_images_per_prompt),
                        0,
                        prompt_embeds_null,
                        cross_attention_kwargs={
                            "attn_store": self.attn_store,
                        },
                    )

            generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
            images = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                cloth_guidance_scale=cloth_guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={
                    "attn_store": self.attn_store,
                    "do_classifier_free_guidance": guidance_scale > 1.0,
                    "enable_cloth_guidance": True,
                },
                **kwargs,
            ).images

        return images, cloth_mask_image


OmsDiffusionPipelineType = Annotated[OmsDiffusionPipeline, gen_widget("OMS_DIFFUSION_PIPELINE")]


@register_node(category="arxiv/abs2403_01779")
def load_oms_diffusion(
    diffusers_pipeline: DiffusersPipelineType,
    ckpt_path: Annotated[str, ComboWidget(choices=lambda: path_tool.get_model_filename_list(__name__, ""))],
) -> tuple[OmsDiffusionPipelineType]:
    ckpt_path = path_tool.get_model_full_path(__name__, "", ckpt_path)

    return (OmsDiffusionPipeline.from_base_pipeline(diffusers_pipeline, ckpt_path),)


@register_node(category="arxiv/abs2403_01779")
def run_oms_diffusers(
    oms_pipeline: OmsDiffusionPipelineType,
    seed: IntSeedType,
    steps: IntStepsType = 20,
    cfg: FloatCFGType = 7.5,
    cloth_guidance_scale: FloatCFGType = 2.5,
    scheduler: ComboWidget(
        choices=DIFFUSERS_SCHEDULER_CLASS_MAP, ext_none_choice="PIPELINE_DEFAULT"
    ) = "PIPELINE_DEFAULT",
    denoise: FloatPercentageType = 1.0,
    positive_prompt: StringMultilineType = "",
    negative_prompt: StringMultilineType = "",
    batch: IntType = 1,
    cloth_image: ImageType = None,
    cloth_mask: MaskType = None,
) -> tuple[ImageType]:
    _, height, width, _ = cloth_image.shape

    if scheduler is not None:
        oms_pipeline.base_pipeline.scheduler = scheduler.from_config(oms_pipeline.base_pipeline.scheduler.config)
    else:
        oms_pipeline.base_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            oms_pipeline.base_pipeline.scheduler.config
        )

    cloth_mask_image = (
        cloth_mask.reshape((-1, 1, cloth_mask.shape[-2], cloth_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    )

    pbar = comfy.utils.ProgressBar(steps)

    def callback_on_step_end(self, i, t, callback_kwargs):
        pbar.update_absolute(i)
        return {}

    output = oms_pipeline.generate(
        cloth_image=cloth_image.permute(0, 3, 1, 2),
        cloth_mask_image=cloth_mask_image.permute(0, 3, 1, 2),
        prompt=positive_prompt if len(positive_prompt) > 0 else None,
        negative_prompt=negative_prompt if len(negative_prompt) > 0 else None,
        seed=seed,
        cloth_guidance_scale=cloth_guidance_scale,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=cfg,
        num_images_per_prompt=batch,
        output_type="pt",
        callback_on_step_end=callback_on_step_end,
    )

    return (output[0].permute(0, 2, 3, 1),)
