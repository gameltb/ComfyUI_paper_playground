import os

import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image

from .....common import file_get_tool, path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.github.joytag.Models import VisionModel
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import FloatPercentageType, ImageType, StringType, gen_simple_new_type


class JoytagPipeline(PlaygroundPipeline):
    def __init__(self, top_tags: list[str], model: VisionModel) -> None:
        super().__init__()

        self.register_modules(
            top_tags=top_tags,
            model=model,
        )

    @torch.no_grad()
    def __call__(self, input_image: Image.Image, threshold: int):
        image_tensor = prepare_image(input_image, self.model.image_size)

        with AutoManage(self.model) as am:
            device = am.get_device()
            batch = {
                "image": image_tensor.unsqueeze(0).to(device),
            }

            with torch.amp.autocast_mode.autocast(device.type, enabled=True):
                preds = self.model(batch)
                tag_preds = preds["tags"].sigmoid().cpu()

        scores = {self.top_tags[i]: tag_preds[0][i] for i in range(len(self.top_tags))}
        predicted_tags = [tag for tag, score in scores.items() if score > threshold]
        tag_string = ", ".join(predicted_tags)

        return tag_string, scores


JoytagPipelineType = gen_simple_new_type(JoytagPipeline, "JOYTAG_PIPELINE")


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(
        image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )

    return image_tensor


@register_node(category="github/joytag")
def load_joytag() -> tuple[JoytagPipelineType]:
    model_path = file_get_tool.find_or_download_huggingface_repo(
        [
            file_get_tool.FileSource(
                loacal_folder=path_tool.get_model_dir(__name__, ""),
            ),
            file_get_tool.FileSourceHuggingface(repo_id="fancyfeast/joytag"),
        ]
    )

    top_tags_config_path = os.path.join(model_path, "top_tags.txt")
    with open(top_tags_config_path, "r") as f:
        top_tags = [line.strip() for line in f.readlines() if line.strip()]

    model = VisionModel.load_model(model_path)
    model.eval()

    return (JoytagPipeline(top_tags=top_tags, model=model),)


@register_node(category="github/joytag")
def joytag_predict(
    joytag_pipeline: JoytagPipelineType,
    image: ImageType,
    threshold: FloatPercentageType = 0.4,
) -> tuple[StringType]:
    input_image = TVF.to_pil_image(image[0].cpu().numpy())
    tag_string, scores = joytag_pipeline(input_image, threshold)
    return (tag_string,)
