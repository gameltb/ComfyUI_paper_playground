import copy

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .....common import path_tool
from .....paper.arxiv.abs2312_13964.animatediff.models.resnet import InflatedConv3d
from .....paper.arxiv.abs2312_13964.animatediff.models.unet import UNet3DConditionModel
from .....paper.arxiv.abs2312_13964.animatediff.pipelines import I2VPipeline
from ....registry import register_node
from ...diffusers import DiffusersComfyModelPatcherWrapper


def UNet3DConditionModel_from_unet_2d(
    unet_2d: UNet2DConditionModel, unet_additional_kwargs
):
    config = copy.deepcopy(unet_2d.config)
    config["_class_name"] = UNet3DConditionModel.__name__
    config["down_block_types"] = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D",
    ]
    config["up_block_types"] = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
    ]

    model = UNet3DConditionModel.from_config(config, **unet_additional_kwargs)
    m, u = model.load_state_dict(unet_2d.state_dict(), strict=False)
    print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

    params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
    print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")

    return model


@register_node(display_name="Abs 2312.13964 Diffusers Pipeline Build (PIA)")
class Abs2312_13964_DiffusersPipelineBuild:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_pipeline": ("DIFFUSERS_PIPELINE",),
                "pia_unet_name": (path_tool.get_model_filename_list(__name__, "pia"),),
            },
            "optional": {
                "dreambooth_pipeline": ("DIFFUSERS_PIPELINE",),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "build_pipeline"

    CATEGORY = "playground/arxiv/abs2312_13964"

    def build_pipeline(self, base_pipeline, pia_unet_name, dreambooth_pipeline=None):
        pia_unet_path = path_tool.get_model_full_path(__name__, "pia", pia_unet_name)

        pipeline_comfy_model_patcher_wrapper = base_pipeline
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )
        if dreambooth_pipeline is not None:
            dreambooth_pipeline_comfy_model_patcher_wrapper = dreambooth_pipeline
            dreambooth_pipeline: StableDiffusionPipeline = (
                dreambooth_pipeline_comfy_model_patcher_wrapper.model
            )

        unet_additional_kwargs = {
            "use_motion_module": True,
            "motion_module_resolutions": [1, 2, 4, 8],
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
                "zero_initialize": True,
            },
        }
        unet = UNet3DConditionModel_from_unet_2d(
            diffusers_pipeline.components.get("unet"),
            unet_additional_kwargs=unet_additional_kwargs,
        )

        old_weights = unet.conv_in.weight
        old_bias = unet.conv_in.bias
        new_conv1 = InflatedConv3d(
            9,
            old_weights.shape[0],
            kernel_size=unet.conv_in.kernel_size,
            stride=unet.conv_in.stride,
            padding=unet.conv_in.padding,
            bias=True if old_bias is not None else False,
        )
        param = torch.zeros((320, 5, 3, 3), requires_grad=True)
        new_conv1.weight = torch.nn.Parameter(torch.cat((old_weights, param), dim=1))
        if old_bias is not None:
            new_conv1.bias = old_bias
        unet.conv_in = new_conv1
        unet.config["in_channels"] = 9

        unet_ckpt = torch.load(pia_unet_path, map_location="cpu")
        unet.load_state_dict(unet_ckpt, strict=False)

        vae = diffusers_pipeline.components.get("vae")
        tokenizer = diffusers_pipeline.components.get("tokenizer")
        text_encoder = diffusers_pipeline.components.get("text_encoder")

        if dreambooth_pipeline is not None:
            # load unet
            converted_unet_checkpoint = dreambooth_pipeline.components.get(
                "unet"
            ).state_dict()

            old_value = converted_unet_checkpoint["conv_in.weight"]
            new_param = unet_ckpt["conv_in.weight"][:, 4:, :, :].clone().cpu()
            new_value = torch.nn.Parameter(torch.cat((old_value, new_param), dim=1))
            converted_unet_checkpoint["conv_in.weight"] = new_value
            unet.load_state_dict(converted_unet_checkpoint, strict=False)

            vae = dreambooth_pipeline.components.get("vae")
            text_encoder = dreambooth_pipeline.components.get("text_encoder")

        noise_scheduler_kwargs = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "steps_offset": 1,
            "clip_sample": False,
        }
        noise_scheduler = DDIMScheduler(**noise_scheduler_kwargs)

        pia_pipeline = I2VPipeline(
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=noise_scheduler,
        ).to(
            device=comfy.model_management.unet_offload_device(),
            dtype=comfy.model_management.unet_dtype(),
        )

        pipeline_comfy_model_patcher_wrapper = DiffusersComfyModelPatcherWrapper(
            pia_pipeline,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            size=1,
        )

        return (pipeline_comfy_model_patcher_wrapper,)


@register_node(display_name="Abs 2312.13964 Diffusers Pipeline Sampler (PIA)")
class Abs2312_13964_DiffusersPipelineSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "video_length": ("INT", {"default": 16, "min": 1, "max": 32}),
                "magnitude": ("INT", {"default": 0, "min": -3, "max": 3}),
                "loop": ("BOOLEAN", {"default": True}),
                "style_transfer": ("BOOLEAN", {"default": True}),
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_pipeline"

    CATEGORY = "playground/arxiv/abs2312_13964"

    def do_pipeline(
        self,
        diffusers_pipeline,
        image,
        seed,
        steps,
        video_length,
        magnitude,
        loop,
        style_transfer,
        positive_prompt,
        negative_prompt,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: I2VPipeline = pipeline_comfy_model_patcher_wrapper.model

        comfy.model_management.load_models_gpu([pipeline_comfy_model_patcher_wrapper])

        if magnitude is not None:
            mask_sim_range = [magnitude]

        if style_transfer:
            mask_sim_range = [
                -1 * magnitude - 1 if magnitude >= 0 else magnitude
                for magnitude in mask_sim_range
            ]
        elif loop:
            mask_sim_range = [
                magnitude + 3 if magnitude >= 0 else magnitude
                for magnitude in mask_sim_range
            ]

        generator = torch.Generator(
            device=pipeline_comfy_model_patcher_wrapper.load_device
        )
        generator.manual_seed(seed)
        # seed_everything(config.generate.global_seed)

        sim_ranges = mask_sim_range
        if isinstance(sim_ranges, int):
            sim_ranges = [sim_ranges]

        B, H, W, C = image.shape
        sample_height = H
        sample_width = W
        image = image[0].numpy() * 255

        pbar = comfy.utils.ProgressBar(steps)

        def callback_on_step_end(i, t, callback_kwargs):
            pbar.update(i)
            return {}

        for sim_range in sim_ranges:
            print(f"using sim_range : {sim_range}")
            mask_sim_range = sim_range
            sample = diffusers_pipeline(
                image=image,
                prompt=positive_prompt,
                generator=generator,
                video_length=video_length,
                height=sample_height,
                width=sample_width,
                negative_prompt=negative_prompt,
                mask_sim_template_idx=mask_sim_range,
                num_inference_steps=steps,
                cond_frame=0,
                callback=callback_on_step_end,
                callback_steps=1,
            ).videos

        return (sample[0].permute(1, 2, 3, 0),)
