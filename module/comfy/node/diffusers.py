import copy
import json
import os

import diffusers
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel

import comfy.model_management
import comfy.model_patcher
import comfy.utils
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
                        paths.append(
                            full_path.removeprefix(folder_path).removeprefix(os.sep)
                        )
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


class DiffusersPipelineFromPretrained:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_type": (
                    ["AUTO"] + list(DIFFUSERS_PIPELINE_CLASS_MAP.keys()),
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
    FUNCTION = "diffusers_from_pretrained"

    CATEGORY = "playground/loaders"

    def diffusers_from_pretrained(
        self, pipeline_type, local_files_only, directory=None, model_id=None
    ):
        if pipeline_type != "AUTO":
            pipeline_cls = DIFFUSERS_PIPELINE_CLASS_MAP[pipeline_type]
        else:
            pipeline_cls = DiffusionPipeline

        return self.from_pretrained(
            pipeline_cls, local_files_only, directory=directory, model_id=model_id
        )

    def from_pretrained(
        self, pipeline_cls, local_files_only, directory=None, model_id=None
    ):
        pretrained_model_name_or_path: str = model_id
        if (
            pretrained_model_name_or_path == None
            or len(pretrained_model_name_or_path.strip()) == 0
        ):
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


class DiffusersPipelineFromSingleFile:
    config_path = os.path.join(os.path.dirname(__file__), "diffusers_config")
    config_files = {
        "v1": os.path.join(config_path, "v1-inference.yaml"),
        "v2": os.path.join(config_path, "v2-inference-v.yaml"),
        "xl": os.path.join(config_path, "sd_xl_base.yaml"),
        "xl_refiner": os.path.join(config_path, "sd_xl_refiner.yaml"),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_type": (
                    ["AUTO"] + list(DIFFUSERS_PIPELINE_CLASS_MAP.keys()),
                    {"default": StableDiffusionPipeline.__name__},
                ),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
        }

    RETURN_TYPES = ("DIFFUSERS_PIPELINE",)
    FUNCTION = "from_single_file"

    CATEGORY = "playground/loaders"

    def from_single_file(self, pipeline_type, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        if pipeline_type != "AUTO":
            pipeline_cls = DIFFUSERS_PIPELINE_CLASS_MAP[pipeline_type]
        else:
            pipeline_cls = StableDiffusionPipeline

        pipeline = pipeline_cls.from_single_file(
            ckpt_path,
            load_safety_checker=None,
            torch_dtype=comfy.model_management.unet_dtype(),
            local_files_only=True,
            config_files=self.config_files,
        ).to(device=comfy.model_management.unet_offload_device())

        pipeline_comfy_model_patcher_wrapper = DiffusersComfyModelPatcherWrapper(
            pipeline,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            size=1,
        )

        return (pipeline_comfy_model_patcher_wrapper,)


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
                "scheduler": (
                    ["PIPELINE_DEFAULT"] + list(DIFFUSERS_SCHEDULER_CLASS_MAP.keys()),
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
        scheduler,
        latent_image,
        denoise,
    ):
        pipeline_comfy_model_patcher_wrapper = diffusers_pipeline
        latent = latent_image["samples"]
        batch, _, height, width = latent.shape
        generator = torch.Generator(device="cuda").manual_seed(seed)
        output_type = "pt"
        if False:
            output_type = "latent"

        comfy.model_management.load_models_gpu([pipeline_comfy_model_patcher_wrapper])

        diffusers_pipeline = pipeline_comfy_model_patcher_wrapper.model

        if scheduler != "PIPELINE_DEFAULT":
            diffusers_pipeline.scheduler = DIFFUSERS_SCHEDULER_CLASS_MAP[
                scheduler
            ].from_config(diffusers_pipeline.scheduler.config)

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


class DiffusersComponentFromPretrained:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "component_type": (
                    ["AUTO"]
                    + list(DIFFUSERS_MODEL_CLASS_MAP.keys())
                    + list(DIFFUSERS_SCHEDULER_CLASS_MAP.keys()),
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
        if (
            pretrained_model_name_or_path == None
            or len(pretrained_model_name_or_path.strip()) == 0
        ):
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
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )

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
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )

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

        if (
            enable_xformers_memory_efficient_attention
            and comfy.model_management.xformers_enabled()
        ):
            diffusers_pipeline.enable_xformers_memory_efficient_attention()

        return (pipeline_comfy_model_patcher_wrapper,)


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
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )
        diffusers_lora = copy.copy(diffusers_lora)
        adapter_name = (
            adapter_name
            if adapter_name != None and len(adapter_name.strip()) > 0
            else None
        )

        diffusers_pipeline.load_lora_weights(diffusers_lora, adapter_name=adapter_name)

        return (pipeline_comfy_model_patcher_wrapper,)


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
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )

        pretrained_model_name_or_path: str = model_id
        pretrained_model_full_path = model_id
        if (
            pretrained_model_name_or_path == None
            or len(pretrained_model_name_or_path.strip()) == 0
        ):
            pretrained_model_full_path = find_full_diffusers_folder_path(
                ip_adapter_name
            )
            pretrained_model_name_or_path = os.path.dirname(pretrained_model_full_path)
            weight_name = os.path.basename(pretrained_model_full_path)
            subfolder = ""

        diffusers_pipeline.load_ip_adapter(
            pretrained_model_name_or_path, weight_name=weight_name, subfolder=subfolder
        )

        return (pipeline_comfy_model_patcher_wrapper,)


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
        diffusers_pipeline: StableDiffusionPipeline = (
            pipeline_comfy_model_patcher_wrapper.model
        )

        diffusers_pipeline.set_ip_adapter_scale(ip_adapter_scale)

        return (pipeline_comfy_model_patcher_wrapper,)


NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineFromSingleFile": DiffusersPipelineFromSingleFile,
    "DiffusersPipelineFromPretrained": DiffusersPipelineFromPretrained,
    "DiffusersComponentFromPretrained": DiffusersComponentFromPretrained,
    "DiffusersPipelineSamplerBase": DiffusersPipelineSamplerBase,
    "DiffusersPipelineComponentSet": DiffusersPipelineComponentSet,
    "DiffusersPipelineComponentGet": DiffusersPipelineComponentGet,
    "DiffusersPipelineComponentShow": DiffusersPipelineComponentShow,
    "DiffusersPipelineOptimization": DiffusersPipelineOptimization,
    "DiffusersLoadLora": DiffusersLoadLora,
    "DiffusersPipelineLoadLoraWeights": DiffusersPipelineLoadLoraWeights,
    "DiffusersPipelineLoadIPAdapter": DiffusersPipelineLoadIPAdapter,
    "DiffusersPipelineSetIPAdapterScale": DiffusersPipelineSetIPAdapterScale,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineFromSingleFile": "Diffusers Pipeline From Single File",
    "DiffusersPipelineFromPretrained": "Diffusers Pipeline From Pretrained",
    "DiffusersComponentFromPretrained": "Diffusers Component From Pretrained",
    "DiffusersPipelineSamplerBase": "Diffusers Pipeline Sampler Base",
    "DiffusersPipelineComponentSet": "Diffusers Pipeline Component Set",
    "DiffusersPipelineComponentGet": "Diffusers Pipeline Component Get",
    "DiffusersPipelineComponentShow": "Diffusers Pipeline Component Show",
    "DiffusersPipelineOptimization": "Diffusers Pipeline Optimization",
    "DiffusersLoadLora": "Diffusers Load Lora",
    "DiffusersPipelineLoadLoraWeights": "Diffusers Pipeline Load Lora Weights",
    "DiffusersPipelineLoadIPAdapter": "Diffusers Pipeline Load IP Adapter",
    "DiffusersPipelineSetIPAdapterScale": "Diffusers Pipeline Set IP Adapter Scale",
}
