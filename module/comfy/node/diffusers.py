import copy
import inspect
import json
import os
import typing

import diffusers
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from ..registry import register_node
from ..types import (
    ComboWidget,
    BoolType,
    StringWidget,
    ImageType,
    IntWidget,
    FloatWidget,
    LatentType,
    gen_simple_new_type,
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


class DiffusersComfyModelPatcherWrapper(comfy.model_patcher.ModelPatcher):
    def __init__(self, *args, enable_sequential_cpu_offload=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload

    def patch_model(self, device_to=None):
        if device_to is not None:
            if not self.enable_sequential_cpu_offload:
                self.model.to(device=device_to)
            self.current_device = device_to

        return self.model

    def unpatch_model(self, device_to=None):
        if device_to is not None:
            if not self.enable_sequential_cpu_offload:
                self.model.to(device=device_to)
            self.current_device = device_to


DiffusersPipelineType = gen_simple_new_type(DiffusersComfyModelPatcherWrapper, "DIFFUSERS_PIPELINE")


@register_node(identifier="DiffusersPipelineFromPretrained", category="loaders")
def diffusers_from_pretrained(
    pipeline_type: ComboWidget(choices=DIFFUSERS_PIPELINE_CLASS_MAP) = DiffusionPipeline.__name__,
    local_files_only: BoolType = True,
    directory: ComboWidget(choices=lambda: get_diffusers_folder_paths()) = None,
    model_id: StringWidget() = "",
) -> tuple[DiffusersPipelineType]:
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

    pipeline_comfy_model_patcher_wrapper = DiffusersComfyModelPatcherWrapper(
        pipeline,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
        size=1,
    )

    return (pipeline_comfy_model_patcher_wrapper,)


SINGLE_FILE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "diffusers_config")
SINGLE_FILE_CONFIG_FILES = {
    "v1": os.path.join(SINGLE_FILE_CONFIG_PATH, "v1-inference.yaml"),
    "v2": os.path.join(SINGLE_FILE_CONFIG_PATH, "v2-inference-v.yaml"),
    "xl": os.path.join(SINGLE_FILE_CONFIG_PATH, "sd_xl_base.yaml"),
    "xl_refiner": os.path.join(SINGLE_FILE_CONFIG_PATH, "sd_xl_refiner.yaml"),
}


@register_node(identifier="DiffusersPipelineFromSingleFile", category="loaders")
def diffusers_from_single_file(
    pipeline_type: ComboWidget(choices=DIFFUSERS_PIPELINE_CLASS_MAP) = StableDiffusionPipeline.__name__,
    ckpt_name: ComboWidget(choices=lambda: folder_paths.get_filename_list("checkpoints")) = None,
    single_file_config_file: ComboWidget(choices=SINGLE_FILE_CONFIG_FILES) = None,
) -> tuple[DiffusersPipelineType]:
    pipeline_cls: StableDiffusionPipeline = pipeline_type
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

    pipeline = pipeline_cls.from_single_file(
        ckpt_path,
        load_safety_checker=None,
        torch_dtype=comfy.model_management.unet_dtype(),
        local_files_only=True,
        original_config_file=single_file_config_file,
    ).to(device=comfy.model_management.unet_offload_device())

    pipeline_comfy_model_patcher_wrapper = DiffusersComfyModelPatcherWrapper(
        pipeline,
        load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(),
        size=1,
    )

    return (pipeline_comfy_model_patcher_wrapper,)


@register_node(identifier="DiffusersPipelineSamplerBase", category="sampling")
def diffusers_sampler_base(
    diffusers_pipeline: DiffusersPipelineType,
    seed: IntWidget(min=0, max=0xFFFFFFFFFFFFFFFF) = 0,
    steps: IntWidget(min=1, max=10000) = 20,
    cfg: FloatWidget(min=0.0, max=100.0, step=0.1, round=0.01) = 8.0,
    scheduler: ComboWidget(choices=DIFFUSERS_SCHEDULER_CLASS_MAP, ext_none_choice="PIPELINE_DEFAULT") = None,
    latent_image: LatentType = None,
    denoise: FloatWidget(min=0.0, max=1.0, step=0.01) = 1.0,
    positive_prompt: StringWidget(multiline=True) = "",
    negative_prompt: StringWidget(multiline=True) = "",
) -> tuple[ImageType]:
    pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
    latent = latent_image["samples"]
    batch, _, height, width = latent.shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_type = "pt"
    if False:
        output_type = "latent"

    comfy.model_management.load_models_gpu([pipeline_comfy_model_patcher_wrapper])

    diffusers_pipeline = pipeline_comfy_model_patcher_wrapper.model

    if scheduler is not None:
        diffusers_pipeline.scheduler = scheduler.from_config(diffusers_pipeline.scheduler.config)

    pbar = comfy.utils.ProgressBar(steps)

    def callback_on_step_end(self, i, t, callback_kwargs):
        pbar.update(i)
        return {}

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


@register_node()
class DiffusersComponentFromPretrained:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "component_type": (
                    ["AUTO"] + list(DIFFUSERS_MODEL_CLASS_MAP.keys()) + list(DIFFUSERS_SCHEDULER_CLASS_MAP.keys()),
                ),
                "local_files_only": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "directory": (get_diffusers_component_folder_paths(),),
                "model_id": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_COMPONENT",)
    FUNCTION = "from_pretrained"

    CATEGORY = "playground/loaders"

    def from_pretrained(
        self,
        component_type,
        local_files_only,
        directory=None,
        model_id=None,
        subfolder=None,
    ):
        pretrained_model_name_or_path: str = model_id
        if pretrained_model_name_or_path == None or len(pretrained_model_name_or_path.strip()) == 0:
            pretrained_model_name_or_path = find_full_diffusers_folder_path(directory)

        if subfolder != None and len(subfolder.strip()) == 0:
            subfolder = None

        if component_type != "AUTO":
            component_cls = DIFFUSERS_MODEL_CLASS_MAP.get(
                component_type, DIFFUSERS_SCHEDULER_CLASS_MAP.get(component_type, None)
            )
        else:
            component_cls = diffusers.ModelMixin

        component = component_cls.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=comfy.model_management.unet_dtype(),
            local_files_only=local_files_only,
        ).to(device=comfy.model_management.unet_offload_device())

        return (component,)


@register_node()
class DiffusersPipelineComponentSet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "diffusers_component": ("DIFFUSERS_COMPONENT",),
                "component_key": ("STRING", {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "set_component"

    CATEGORY = "playground/tool"

    def set_component(self, diffusers_pipeline, diffusers_component, component_key):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model

        diffusers_component.to(dtype=diffusers_pipeline.dtype)

        if component_key == "auto":
            for k, v in diffusers_pipeline.components.items():
                if type(v) == type(diffusers_component):
                    component_key = k
                    break

        new_component_map = copy.copy(diffusers_pipeline.components)
        new_component_map[component_key] = diffusers_component
        diffusers_pipeline = diffusers_pipeline.__class__(**new_component_map)

        pipeline_comfy_model_patcher_wrapper = DiffusersComfyModelPatcherWrapper(
            diffusers_pipeline,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            size=1,
        )

        return (pipeline_comfy_model_patcher_wrapper,)


@register_node()
class DiffusersPipelineComponentGet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "component_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("DIFFUSERS_COMPONENT",)
    FUNCTION = "get_component"

    CATEGORY = "playground/tool"

    def get_component(self, diffusers_pipeline, component_key):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline = pipeline_comfy_model_patcher_wrapper.model

        return (diffusers_pipeline.components.get(component_key),)


@register_node()
class DiffusersPipelineComponentShow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show_component"

    OUTPUT_NODE = True

    CATEGORY = "playground/tool"

    def show_component(self, diffusers_pipeline):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline = pipeline_comfy_model_patcher_wrapper.model

        components_map = {}
        for k, v in diffusers_pipeline.components.items():
            components_map[k] = str(type(v))

        return {"ui": {"components_map": [json.dumps(components_map, indent=4)]}}


@register_node()
class DiffusersPipelineListAdapters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "list_adapters"

    OUTPUT_NODE = True

    CATEGORY = "playground/tool"

    def list_adapters(self, diffusers_pipeline):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model

        return {"ui": {"components_map": [json.dumps(diffusers_pipeline.get_list_adapters(), indent=4)]}}


@register_node()
class DiffusersPipelineOptimization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
                "enable_model_cpu_offload": ("BOOLEAN", {"default": False}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
                "enable_xformers_memory_efficient_attention": (
                    "BOOLEAN",
                    {"default": True},
                ),
            }
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "optimization"

    CATEGORY = "playground/tool"

    def optimization(
        self,
        diffusers_pipeline,
        enable_vae_slicing,
        enable_vae_tiling,
        enable_model_cpu_offload,
        enable_sequential_cpu_offload,
        enable_xformers_memory_efficient_attention,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model

        if enable_vae_slicing:
            diffusers_pipeline.enable_vae_slicing()

        if enable_vae_tiling:
            diffusers_pipeline.enable_vae_tiling()

        if enable_model_cpu_offload:
            pipeline_comfy_model_patcher_wrapper.enable_sequential_cpu_offload = True
            diffusers_pipeline.enable_model_cpu_offload()

        if enable_sequential_cpu_offload:
            pipeline_comfy_model_patcher_wrapper.enable_sequential_cpu_offload = True
            diffusers_pipeline.enable_sequential_cpu_offload()

        if enable_xformers_memory_efficient_attention and comfy.model_management.xformers_enabled():
            diffusers_pipeline.enable_xformers_memory_efficient_attention()

        return (pipeline_comfy_model_patcher_wrapper,)


@register_node()
class DiffusersLoadLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }

    RETURN_TYPES = ("DIFFUSERS_LORA",)
    FUNCTION = "load_lora"

    CATEGORY = "playground/loaders"

    def load_lora(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return (lora,)


@register_node()
class DiffusersPipelineLoadLoraWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "diffusers_lora": ("DIFFUSERS_LORA",),
            },
            "optional": {
                "adapter_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "load_lora_weights"

    CATEGORY = "playground/tool"

    def load_lora_weights(self, diffusers_pipeline, diffusers_lora, adapter_name):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model
        diffusers_lora = copy.copy(diffusers_lora)
        adapter_name = adapter_name if adapter_name != None and len(adapter_name.strip()) > 0 else None

        diffusers_pipeline.load_lora_weights(diffusers_lora, adapter_name=adapter_name)

        return (pipeline_comfy_model_patcher_wrapper,)


@register_node()
class DiffusersPipelineLoadIPAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
            },
            "optional": {
                "ip_adapter_name": (get_diffusers_ip_adapter_paths(),),
                "model_id": ("STRING", {"default": ""}),
                "subfolder": ("STRING", {"default": ""}),
                "weight_name": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "load_ip_adapter"

    CATEGORY = "playground/tool"

    def load_ip_adapter(
        self,
        diffusers_pipeline,
        ip_adapter_name=None,
        model_id=None,
        subfolder=None,
        weight_name=None,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model

        pretrained_model_name_or_path: str = model_id
        pretrained_model_full_path = model_id
        if pretrained_model_name_or_path == None or len(pretrained_model_name_or_path.strip()) == 0:
            pretrained_model_full_path = find_full_diffusers_folder_path(ip_adapter_name)
            pretrained_model_name_or_path = os.path.dirname(pretrained_model_full_path)
            weight_name = os.path.basename(pretrained_model_full_path)
            subfolder = ""

        diffusers_pipeline.load_ip_adapter(pretrained_model_name_or_path, weight_name=weight_name, subfolder=subfolder)

        return (pipeline_comfy_model_patcher_wrapper,)


@register_node()
class DiffusersPipelineSetIPAdapterScale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_pipeline": ("DIFFUSERS_PIPELINE",),
                "ip_adapter_scale": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "set_ip_adapter_scale"

    CATEGORY = "playground/tool"

    def set_ip_adapter_scale(
        self,
        diffusers_pipeline,
        ip_adapter_scale,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        diffusers_pipeline: StableDiffusionPipeline = pipeline_comfy_model_patcher_wrapper.model

        diffusers_pipeline.set_ip_adapter_scale(ip_adapter_scale)

        return (pipeline_comfy_model_patcher_wrapper,)
