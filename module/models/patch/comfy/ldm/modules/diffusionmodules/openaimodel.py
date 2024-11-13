from typing import Callable, TypedDict

import torch
from comfy.ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    apply_control,
    forward_timestep_embed,
    timestep_embedding,
)

from ........module.core.patch_able_module import ControlFlowPatchAbleModuleMixin


class PatchModuleMapType(TypedDict, total=False):
    comfyui_input_block_patch: list[Callable[[torch.Tensor, dict], torch.Tensor]]
    comfyui_input_block_patch_after_skip: list[Callable[[torch.Tensor, dict], torch.Tensor]]
    comfyui_middle_block_patch: list[Callable[[torch.Tensor, dict], torch.Tensor]]
    comfyui_output_block_patch: list[Callable[[torch.Tensor, torch.Tensor, dict], tuple[torch.Tensor, torch.Tensor]]]


class UNetModel(UNetModel, ControlFlowPatchAbleModuleMixin[PatchModuleMapType]):
    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
            h = apply_control(h, control, "input")
            for p in self.patcher_module.get("comfyui_input_block_patch"):
                h = p(h, transformer_options)

            hs.append(h)

            for p in self.patcher_module.get("comfyui_input_block_patch_after_skip"):
                h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed(
                self.middle_block,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = apply_control(h, control, "middle")
        for p in self.patcher_module.get("comfyui_middle_block_patch"):
            h = p(h, transformer_options)

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, "output")
            for p in self.patcher_module.get("comfyui_output_block_patch"):
                h, hsp = p(h, hsp, transformer_options)

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                output_shape,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator,
            )
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
