import copy
import inspect
import json
import os
from typing import Annotated, Union

import comfy.model_management
import comfy.utils
import diffusers
import folder_paths
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline

from ...core.runtime_resource_management import AutoManage
from ..registry import register_node
from ..types import (
    BoolType,
    ComboWidget,
    FloatCFGType,
    FloatPercentageType,
    ImageType,
    IntSeedType,
    IntStepsType,
    LatentType,
    ModelType,
    SigmasType,
    StringMultilineType,
    StringType,
    gen_widget,
)

DIFFUSERS_PIPELINE_CLASS_MAP = {}
DIFFUSERS_MODEL_CLASS_MAP = {}
DIFFUSERS_SCHEDULER_CLASS_MAP = {}


clsmembers = inspect.getmembers(diffusers, inspect.isclass)

for cls_name, cls in clsmembers:
    if issubclass(cls, diffusers.DiffusionPipeline):
        DIFFUSERS_PIPELINE_CLASS_MAP[cls_name] = cls
    elif issubclass(cls, diffusers.ModelMixin):
        if cls != diffusers.ModelMixin:
            DIFFUSERS_MODEL_CLASS_MAP[cls_name] = cls
    elif issubclass(cls, diffusers.SchedulerMixin):
        if cls != diffusers.SchedulerMixin:
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
        full_path = os.path.join(folder_path, diffusers_folder_path)
        if os.path.exists(full_path):
            return full_path


def get_diffusers_component_folder_paths():
    paths = []
    for folder_path in folder_paths.get_folder_paths("diffusers"):
        for diffusers_folder_path in os.listdir(folder_path):
            full_folder_path = os.path.join(folder_path, diffusers_folder_path)
            if os.path.exists(os.path.join(full_folder_path, "model_index.json")):
                with open(os.path.join(full_folder_path, "model_index.json"), "r") as f:
                    model_map = json.load(f)
                for key in model_map:
                    if not key.startswith("_"):
                        paths.append(os.path.join(diffusers_folder_path, key))
            elif os.path.exists(os.path.join(full_folder_path, "config.json")):
                paths.append(diffusers_folder_path)
    return paths


def get_diffusers_ip_adapter_paths():
    paths = []
    for folder_path in folder_paths.get_folder_paths("diffusers"):
        ip_adapter_dir_path = os.path.join(folder_path, "IP-Adapter")
        if os.path.exists(ip_adapter_dir_path):
            for root, subdir, file in os.walk(ip_adapter_dir_path, followlinks=True):
                for filename in file:
                    full_path = os.path.join(root, filename)
                    if filename.startswith("ip-adapter") and os.path.isfile(full_path):
                        paths.append(full_path.removeprefix(folder_path).removeprefix(os.sep))
    return paths


DiffusersPipelineType = Annotated[DiffusionPipeline, gen_widget("DIFFUSERS_PIPELINE")]
DiffusersComponentType = Annotated[
    Union[diffusers.ModelMixin, diffusers.SchedulerMixin], gen_widget("DIFFUSERS_COMPONENT")
]
DiffusersLoRAType = Annotated[dict[str, torch.Tensor], gen_widget("DIFFUSERS_LORA")]
MetaDiffusersPipelineType = Annotated[DiffusionPipeline, gen_widget("META_DIFFUSERS_PIPELINE")]
"""Diffusers Pipeline without weights, make it deepcopy able."""


@register_node(identifier="DiffusersPipelineFromPretrained", category="diffusers/loaders")
def diffusers_from_pretrained(
    pipeline_type: ComboWidget(choices=DIFFUSERS_PIPELINE_CLASS_MAP) = DiffusionPipeline.__name__,
    local_files_only: BoolType = True,
    directory: ComboWidget(choices=lambda: get_diffusers_folder_paths()) = None,
    model_id: StringType = "",
) -> tuple[DiffusersPipelineType]:
    """Instantiate a PyTorch diffusion pipeline from pretrained pipeline weights."""
    pipeline_cls = pipeline_type
    return diffusers_from_pretrained_cls(pipeline_cls, local_files_only, directory=directory, model_id=model_id)


def diffusers_from_pretrained_cls(pipeline_cls, local_files_only=True, directory=None, model_id=None):
    pretrained_model_name_or_path: str = model_id
    if pretrained_model_name_or_path is None or len(pretrained_model_name_or_path.strip()) == 0:
        pretrained_model_name_or_path = find_full_diffusers_folder_path(directory)

    pipeline = pipeline_cls.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=comfy.model_management.unet_dtype(),
        local_files_only=local_files_only,
    ).to(device=comfy.model_management.unet_offload_device())

    return (pipeline,)


SINGLE_FILE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "diffusers_config")
SINGLE_FILE_CONFIG_FILES = {
    "v1": os.path.join(SINGLE_FILE_CONFIG_PATH, "v1-inference.yaml"),
    "v2": os.path.join(SINGLE_FILE_CONFIG_PATH, "v2-inference-v.yaml"),
    "xl": os.path.join(SINGLE_FILE_CONFIG_PATH, "sd_xl_base.yaml"),
    "xl_refiner": os.path.join(SINGLE_FILE_CONFIG_PATH, "sd_xl_refiner.yaml"),
}


@register_node(identifier="DiffusersPipelineFromSingleFile", category="diffusers/loaders")
def diffusers_from_single_file(
    pipeline_type: ComboWidget(choices=DIFFUSERS_PIPELINE_CLASS_MAP) = StableDiffusionPipeline.__name__,
    ckpt_name: ComboWidget(choices=lambda: folder_paths.get_filename_list("checkpoints")) = None,
    single_file_config_file: ComboWidget(choices=SINGLE_FILE_CONFIG_FILES) = None,
    inference_mode: BoolType = True,
) -> tuple[DiffusersPipelineType]:
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    with torch.inference_mode(inference_mode):
        pipeline = pipeline_type.from_single_file(
            ckpt_path,
            load_safety_checker=None,
            torch_dtype=comfy.model_management.unet_dtype(),
            local_files_only=True,
            original_config_file=single_file_config_file,
        ).to(device=comfy.model_management.unet_offload_device())

    return (pipeline,)


@register_node(identifier="DiffusersPipelineSamplerBase", category="diffusers/sampling")
def diffusers_sampler_base(
    diffusers_pipeline: DiffusersPipelineType,
    seed: IntSeedType,
    steps: IntStepsType = 20,
    cfg: FloatCFGType = 8.0,
    scheduler: ComboWidget(
        choices=DIFFUSERS_SCHEDULER_CLASS_MAP, ext_none_choice="PIPELINE_DEFAULT"
    ) = "PIPELINE_DEFAULT",
    latent_image: LatentType = None,
    denoise: FloatPercentageType = 1.0,
    positive_prompt: StringMultilineType = "",
    negative_prompt: StringMultilineType = "",
) -> tuple[ImageType]:
    latent = latent_image["samples"]
    batch, _, height, width = latent.shape
    output_type = "pt"
    if False:
        output_type = "latent"

    if scheduler is not None:
        diffusers_pipeline.scheduler = scheduler.from_config(diffusers_pipeline.scheduler.config)

    pbar = comfy.utils.ProgressBar(steps)

    def callback_on_step_end(self, i, t, callback_kwargs):
        pbar.update_absolute(i)
        return {}

    with AutoManage(diffusers_pipeline) as am:
        generator = torch.Generator(device=am.get_device()).manual_seed(seed)

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
            callback_on_step_end=callback_on_step_end,
        )

    return (output.images.permute(0, 2, 3, 1),)


@register_node(identifier="DiffusersComponentFromPretrained", category="diffusers/loaders")
def diffusers_component_from_pretrained(
    component_type: Annotated[
        type[DiffusersComponentType],
        ComboWidget(choices=DIFFUSERS_MODEL_CLASS_MAP | DIFFUSERS_SCHEDULER_CLASS_MAP, ext_none_choice="AUTO"),
    ] = "AUTO",
    local_files_only: BoolType = True,
    directory: Annotated[str, ComboWidget(choices=lambda: get_diffusers_component_folder_paths())] = "",
    model_id: StringType = "",
    subfolder: StringType = "",
    variant: StringType = "",
) -> tuple[DiffusersComponentType]:
    pretrained_model_name_or_path: str = model_id
    if pretrained_model_name_or_path is None or len(pretrained_model_name_or_path.strip()) == 0:
        pretrained_model_name_or_path = find_full_diffusers_folder_path(directory)

    if subfolder is not None and len(subfolder.strip()) == 0:
        subfolder = None

    if variant is not None and len(variant.strip()) == 0:
        variant = None

    if component_type is None:
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            pipeline_class_name = json.load(f)["_class_name"]
        component_type: diffusers.ModelMixin = getattr(diffusers, pipeline_class_name)

    component = component_type.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        subfolder=subfolder,
        variant=variant,
        torch_dtype=comfy.model_management.unet_dtype(),
        local_files_only=local_files_only,
    ).to(device=comfy.model_management.unet_offload_device())

    return (component,)


@register_node(identifier="DiffusersPipelineComponentSet", category="diffusers/tool")
def set_component(
    diffusers_pipeline: DiffusersPipelineType,
    diffusers_component: DiffusersComponentType,
    component_key: StringType = "auto",
) -> tuple[DiffusersPipelineType]:
    diffusers_component.to(dtype=diffusers_pipeline.dtype)

    if component_key == "auto":
        for k, v in diffusers_pipeline.components.items():
            if type(v) == type(diffusers_component):
                component_key = k
                break

    new_component_map = copy.copy(diffusers_pipeline.components)
    new_component_map[component_key] = diffusers_component
    diffusers_pipeline = diffusers_pipeline.__class__(**new_component_map)

    return (diffusers_pipeline,)


@register_node(identifier="DiffusersPipelineComponentGet", category="diffusers/tool")
def get_component(
    diffusers_pipeline: DiffusersPipelineType, component_key: StringType = ""
) -> tuple[DiffusersComponentType]:
    return (diffusers_pipeline.components.get(component_key),)


@register_node(identifier="DiffusersPipelineComponentShow", category="diffusers/tool")
def show_component(diffusers_pipeline: DiffusersPipelineType) -> tuple[StringType]:
    components_map = {}
    for k, v in diffusers_pipeline.components.items():
        components_map[k] = str(type(v))

    return (json.dumps(components_map, indent=4),)


@register_node(identifier="DiffusersPipelineListAdapters", category="diffusers/tool")
def list_adapters(diffusers_pipeline: DiffusersPipelineType) -> tuple[StringType]:
    return (json.dumps(diffusers_pipeline.get_list_adapters(), indent=4),)


@register_node(identifier="DiffusersLoadLora", category="diffusers/loaders")
def load_lora(
    lora_name: Annotated[str, ComboWidget(choices=lambda: folder_paths.get_filename_list("loras"))],
) -> tuple[DiffusersLoRAType]:
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

    return (lora,)


@register_node(identifier="DiffusersPipelineLoadLoraWeights", category="diffusers/tool")
def load_lora_weights(
    diffusers_pipeline: DiffusersPipelineType, diffusers_lora: DiffusersLoRAType, adapter_name: StringType = ""
) -> tuple[DiffusersPipelineType]:
    diffusers_lora = copy.copy(diffusers_lora)
    adapter_name = adapter_name if adapter_name is not None and len(adapter_name.strip()) > 0 else None

    diffusers_pipeline.load_lora_weights(diffusers_lora, adapter_name=adapter_name)

    return (diffusers_pipeline,)


@register_node(identifier="DiffusersPipelineLoadIPAdapter", category="diffusers/tool")
def load_ip_adapter(
    diffusers_pipeline: DiffusersPipelineType,
    ip_adapter_name: Annotated[str, ComboWidget(choices=lambda: get_diffusers_ip_adapter_paths())],
    model_id: StringType = "",
    subfolder: StringType = "",
    weight_name: StringType = "",
) -> tuple[DiffusersPipelineType]:
    pretrained_model_name_or_path: str = model_id
    pretrained_model_full_path = model_id
    if pretrained_model_name_or_path is None or len(pretrained_model_name_or_path.strip()) == 0:
        pretrained_model_full_path = find_full_diffusers_folder_path(ip_adapter_name)
        pretrained_model_name_or_path = os.path.dirname(pretrained_model_full_path)
        weight_name = os.path.basename(pretrained_model_full_path)
        subfolder = ""

    diffusers_pipeline.load_ip_adapter(pretrained_model_name_or_path, weight_name=weight_name, subfolder=subfolder)

    return (diffusers_pipeline,)


@register_node(identifier="DiffusersPipelineSetIPAdapterScale", category="diffusers/tool")
def set_ip_adapter_scale(
    diffusers_pipeline: DiffusersPipelineType,
    ip_adapter_scale: FloatPercentageType = 0.5,
) -> tuple[DiffusersPipelineType]:
    diffusers_pipeline.set_ip_adapter_scale(ip_adapter_scale)

    return (diffusers_pipeline,)


@register_node(category="diffusers/tool")
def get_hype_sd_sigmas(model: ModelType, num_inference_steps: IntStepsType = 1) -> tuple[SigmasType]:
    scheduler = DDIMScheduler.from_config(
        {
            "_class_name": "PNDMScheduler",
            "_diffusers_version": "0.6.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None,
            "clip_sample": False,
        },
        timestep_spacing="trailing",
    )
    scheduler.set_timesteps(num_inference_steps)
    comfy.model_management.load_models_gpu([model])
    sigmas = model.model.model_sampling.sigma(scheduler.timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return (sigmas,)
