import os

import diffusers
import torch
from diffusers import StableDiffusionPipeline

import comfy.model_management
import folder_paths

DIFFUSERS_PIPELINE_CLASS_MAP = {}
DIFFUSERS_MODEL_CLASS_MAP = {}
DIFFUSERS_SCHEDULER_CLASS_MAP = {}

import inspect

clsmembers = inspect.getmembers(diffusers, inspect.isclass)

for cls_name, cls in clsmembers:
    if issubclass(cls, diffusers.DiffusionPipeline):
        DIFFUSERS_PIPELINE_CLASS_MAP[cls_name] = cls
    elif issubclass(cls, diffusers.ModelMixin):
        DIFFUSERS_MODEL_CLASS_MAP[cls_name] = cls
    elif issubclass(cls, diffusers.SchedulerMixin):
        DIFFUSERS_SCHEDULER_CLASS_MAP[cls_name] = cls
    # else:
    #     print("not register: ", cls)


def get_diffusers_folder_paths():
    paths = []
    for folder_path in folder_paths.get_folder_paths("diffusers"):
        for diffusers_folder_path in os.listdir(folder_path):
            full_folder_path = os.path.join(folder_path, diffusers_folder_path)
            if os.path.exists(os.path.join(full_folder_path, "model_index.json")):
                paths.append(diffusers_folder_path)
    return paths


def find_full_diffusers_folder_path(diffusers_folder_path):
    for folder_path in folder_paths.get_folder_paths("diffusers"):
        if diffusers_folder_path in os.listdir(folder_path):
            return os.path.join(folder_path, diffusers_folder_path)


class DiffusersPipelineFromPretrained:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_type": (
                    list(DIFFUSERS_PIPELINE_CLASS_MAP.keys()),
                    {"default": StableDiffusionPipeline.__name__},
                ),
                "local_files_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "directory": (get_diffusers_folder_paths(),),
                "model_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "from_pretrained"

    CATEGORY = "playground/loaders"

    def from_pretrained(
        self, pipeline_type, local_files_only, directory=None, model_id=None
    ):
        pretrained_model_name_or_path: str = model_id
        if (
            pretrained_model_name_or_path == None
            or len(pretrained_model_name_or_path.strip()) == 0
        ):
            pretrained_model_name_or_path = find_full_diffusers_folder_path(directory)

        pipeline_cls = DIFFUSERS_PIPELINE_CLASS_MAP[pipeline_type]

        pipeline = pipeline_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            safety_checker=None,
            torch_dtype=comfy.model_management.unet_dtype(),
            local_files_only=local_files_only,
        ).to(device="cpu")

        return (pipeline,)


class DiffusersPipelineSamplerBase:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "do_pipeline"

    CATEGORY = "playground/sampling"

    def do_pipeline(
        self,
        diffusers_pipeline,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        latent_image,
        denoise,
    ):
        latent = latent_image["samples"]
        batch, _, height, width = latent.shape
        generator = torch.Generator(device="cuda").manual_seed(seed)
        output_type = "pt"
        if False:
            output_type = "latent"

        diffusers_pipeline.to(device="cuda")

        output = diffusers_pipeline(
            prompt=positive_prompt,
            height=height * 8,
            width=width * 8,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch,
            generator=generator,
            output_type=output_type,
        )

        diffusers_pipeline.to(device="cpu")
        return (output.images.permute(0, 2, 3, 1),)


# class DiffusersComponent:


NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineFromPretrained": DiffusersPipelineFromPretrained,
    "DiffusersPipelineSamplerBase": DiffusersPipelineSamplerBase,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineFromPretrained": "Diffusers Pipeline From Pretrained",
    "DiffusersPipelineSamplerBase": "Diffusers Pipeline Sampler Base",
}
