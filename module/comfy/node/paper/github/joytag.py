import os

import torch
import torch.amp.autocast_mode
import torchvision.transforms.functional as TVF
from PIL import Image

from .....common import path_tool
from .....paper.github.joytag.Models import VisionModel
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import FloatPercentageType, ImageType, StringType, gen_simple_new_type

MODEL_PATH = path_tool.get_model_dir(__name__, "")

TOP_TAGS_CONFIG_PATH = os.path.join(MODEL_PATH, "top_tags.txt")


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
        batch = {
            "image": image_tensor.unsqueeze(0).to("cuda"),
        }

        with torch.amp.autocast_mode.autocast("cuda", enabled=True):
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
    with open(TOP_TAGS_CONFIG_PATH, "r") as f:
        top_tags = [line.strip() for line in f.readlines() if line.strip()]

    model = VisionModel.load_model(MODEL_PATH)
    model.eval()
    model = model.to("cuda")

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