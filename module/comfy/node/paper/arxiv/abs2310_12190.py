import os
from typing import Annotated

import comfy.utils
import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
from omegaconf import OmegaConf

from .....common import path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2310_12190.lvdm.models.ddpm3d import LatentVisualDiffusion
from .....paper.arxiv.abs2310_12190.lvdm.models.samplers.ddim import DDIMSampler
from .....paper.arxiv.abs2310_12190.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from .....paper.arxiv.abs2310_12190.utils.utils import instantiate_from_config
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import (
    BoolType,
    ComboWidget,
    FloatCFGType,
    FloatPercentageType,
    FloatType,
    ImageType,
    IntSeedType,
    IntStepsType,
    IntType,
    IntWidget,
    StringMultilineType,
    gen_simple_new_type,
    new_widget,
)


class DynamiCrafterPipeline(PlaygroundPipeline):
    def __init__(self, model: LatentVisualDiffusion) -> None:
        super().__init__()

        self.register_modules(
            model=model,
        )
        self.model = model

    @torch.no_grad()
    def __call__(
        self,
        input_image_start: torch.Tensor,  # [C,H,W]
        interp: bool,
        prompts,
        video_frames,
        video_size,
        n_samples=1,
        ddim_steps=50,
        ddim_eta=1.0,
        unconditional_guidance_scale=1.0,
        cfg_img=None,
        frame_stride=None,
        text_input=False,
        multiple_cond_cfg=False,
        loop=False,
        timestep_spacing="uniform",
        guidance_rescale=0.0,
        input_image_end: torch.Tensor = None,
    ):
        _, height, width = input_image_start.shape
        ## run over data
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        ## latent noise shape
        h, w = height // 8, width // 8
        channels = self.model.model.diffusion_model.out_channels
        noise_shape = [1, channels, video_frames, h, w]

        transform = transforms.Compose(
            [
                # transforms.Resize(min(video_size)),
                # transforms.CenterCrop(video_size),
                # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        if interp:
            image_tensor1 = transform(input_image_start).unsqueeze(1)  # [c,1,h,w]
            image_tensor2 = transform(input_image_end).unsqueeze(1)  # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, "c t h w -> c (repeat t) h w", repeat=video_frames // 2)
            frame_tensor2 = repeat(image_tensor2, "c t h w -> c (repeat t) h w", repeat=video_frames // 2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
        else:
            image_tensor = transform(input_image_start).unsqueeze(1)  # [c,1,h,w]
            frame_tensor = repeat(image_tensor, "c t h w -> c (repeat t) h w", repeat=video_frames)

        with torch.no_grad(), torch.cuda.amp.autocast():
            videos = frame_tensor.unsqueeze(0)

            batch_samples = self.image_guided_synthesis(
                prompts,
                videos,
                noise_shape,
                n_samples,
                ddim_steps,
                ddim_eta,
                unconditional_guidance_scale,
                cfg_img,
                frame_stride,
                text_input,
                multiple_cond_cfg,
                loop,
                interp,
                timestep_spacing,
                guidance_rescale,
            )
        return batch_samples

    @torch.no_grad()
    def image_guided_synthesis(
        self,
        prompts,
        videos,
        noise_shape,
        n_samples=1,
        ddim_steps=50,
        ddim_eta=1.0,
        unconditional_guidance_scale=1.0,
        cfg_img=None,
        frame_stride=3,
        text_input=False,
        multiple_cond_cfg=False,
        loop=False,
        interp=False,
        timestep_spacing="uniform",
        guidance_rescale=0.0,
        **kwargs,
    ):
        ddim_sampler = DDIMSampler(self.model) if not multiple_cond_cfg else DDIMSampler_multicond(self.model)
        batch_size = noise_shape[0]
        frame_stride = torch.tensor([frame_stride] * batch_size, dtype=torch.long)

        if not text_input:
            prompts = [""] * batch_size

        img = videos[:, :, 0]  # bchw
        with AutoManage(self.model.embedder) as am:
            img = img.to(am.get_device())
            img_emb = self.model.embedder(img)  ## blc
        with AutoManage(self.model.image_proj_model) as am:
            img_emb = img_emb.to(am.get_device())
            img_emb = self.model.image_proj_model(img_emb)

        with AutoManage(self.model.cond_stage_model) as am:
            cond_emb = self.model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
        if self.model.model.conditioning_key == "hybrid":
            z = self.get_latent_z(videos)  # b c t h w
            if loop or interp:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                img_cat_cond = z[:, :, :1, :, :]
                img_cat_cond = repeat(img_cat_cond, "b c t h w -> b c (repeat t) h w", repeat=z.shape[2])
            cond["c_concat"] = [img_cat_cond]  # b c 1 h w

        if unconditional_guidance_scale != 1.0:
            if self.model.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                with AutoManage(self.model.cond_stage_model) as am:
                    uc_emb = self.model.get_learned_conditioning(prompts)
            elif self.model.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            with AutoManage(self.model.embedder) as am:
                uc_img_emb = self.model.embedder(torch.zeros_like(img))  ## b l c
            with AutoManage(self.model.image_proj_model) as am:
                uc_img_emb = self.model.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
            if self.model.model.conditioning_key == "hybrid":
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        ## we need one more unconditioning image=yes, text=""
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
            if self.model.model.conditioning_key == "hybrid":
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        z0 = None
        cond_mask = None

        batch_variants = []
        for _ in range(n_samples):
            if z0 is not None:
                cond_z0 = z0.clone()
                kwargs.update({"clean_cond": True})
            else:
                cond_z0 = None
            if ddim_sampler is not None:
                with AutoManage(self.model.model) as am:
                    frame_stride = frame_stride.to(am.get_device())
                    self.model.betas = self.model.betas.to(am.get_device())
                    self.model.sqrt_alphas_cumprod = self.model.sqrt_alphas_cumprod.to(am.get_device())
                    self.model.sqrt_one_minus_alphas_cumprod = self.model.sqrt_one_minus_alphas_cumprod.to(
                        am.get_device()
                    )
                    samples, _ = ddim_sampler.sample(
                        S=ddim_steps,
                        conditioning=cond,
                        batch_size=batch_size,
                        shape=noise_shape[1:],
                        verbose=False,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        cfg_img=cfg_img,
                        mask=cond_mask,
                        x0=cond_z0,
                        fs=frame_stride,
                        timestep_spacing=timestep_spacing,
                        guidance_rescale=guidance_rescale,
                        **kwargs,
                    )

            ## reconstruct from latent to pixel space
            with AutoManage(self.model.first_stage_model) as am:
                batch_images = self.model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants)
        return batch_variants.permute(1, 0, 2, 3, 4, 5)

    def get_latent_z(self, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, "b c t h w -> (b t) c h w")
        with AutoManage(self.model.first_stage_model) as am:
            x = x.to(am.get_device())
            z = self.model.encode_first_stage(x)
        z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
        return z


DynamiCrafterPipelineType = gen_simple_new_type(DynamiCrafterPipeline, "DYNAMI_CRAFTER_PIPELINE")

DYNAMI_CRAFTER_CONFIG_PATH = os.path.join(path_tool.get_paper_repo_path(__name__), "configs")
DYNAMI_CRAFTER_CONFIG_FILES = {
    filename: os.path.join(DYNAMI_CRAFTER_CONFIG_PATH, filename) for filename in os.listdir(DYNAMI_CRAFTER_CONFIG_PATH)
}


@register_node(category="arxiv/abs2310_12190")
def load_dynami_crafter(
    ckpt_path: Annotated[str, ComboWidget(choices=lambda: path_tool.get_model_filename_list(__name__, ""))],
    dynami_crafter_config: Annotated[str, ComboWidget(choices=DYNAMI_CRAFTER_CONFIG_FILES)] = "inference_512_v1.0.yaml",
) -> tuple[DynamiCrafterPipelineType]:
    ckpt_path = path_tool.get_model_full_path(__name__, "", ckpt_path)

    config = OmegaConf.load(dynami_crafter_config)
    model_config = config.pop("model", OmegaConf.create())

    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
    model: LatentVisualDiffusion = instantiate_from_config(model_config)
    ckpt = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model.model = model.model.to(torch.float16)

    return (DynamiCrafterPipeline(model),)


@register_node(category="arxiv/abs2310_12190")
def run_dynami_crafter(
    dynami_crafter_pipeline: DynamiCrafterPipelineType,
    interp: BoolType,
    seed: IntSeedType,
    video_frames: IntType = 16,
    ddim_steps: IntStepsType = 50,
    ddim_eta: FloatPercentageType = 1.0,
    unconditional_guidance_scale: FloatCFGType = 7.5,
    cfg_img: FloatCFGType = 1.0,
    frame_stride: Annotated[int, IntWidget(min=5, max=30)] = 5,
    multiple_cond_cfg: BoolType = False,
    loop: BoolType = False,
    timestep_spacing: Annotated[str, ComboWidget(choices=["uniform", "uniform_trailing", "quad"])] = "uniform",
    guidance_rescale: FloatType = 0.0,
    image_start: ImageType = None,
    image_end: Annotated[ImageType, new_widget(ImageType, is_required=False)] = None,
    prompt: StringMultilineType = "",
) -> tuple[ImageType]:
    videos = dynami_crafter_pipeline.__call__(
        image_start[0].permute(2, 0, 1),
        interp,
        prompt,
        video_frames,
        None,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        unconditional_guidance_scale=unconditional_guidance_scale,
        cfg_img=cfg_img,
        frame_stride=frame_stride,
        text_input=len(prompt) > 0,
        multiple_cond_cfg=multiple_cond_cfg,
        loop=loop,
        timestep_spacing=timestep_spacing,
        guidance_rescale=guidance_rescale,
        input_image_end=image_end[0].permute(2, 0, 1),
    )
    return (videos[0][0].permute(1, 2, 3, 0),)
