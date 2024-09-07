import random
from typing import Annotated

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from .....common import file_get_tool, path_tool
from .....core.runtime_resource_management import AutoManage
from .....pipelines.playground_pipeline import PlaygroundPipeline
from ....registry import register_node
from ....types import ComboWidget, ImageType, IntType, StringMultilineType, StringType, make_widget

DEFAULT_CATEGORY = path_tool.gen_default_category_path_by_module_name(__name__)


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data["bboxes"], data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color="white", fontsize=8, bbox=dict(facecolor="red", alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis("off")

    fig.canvas.draw()
    image = np.array(fig.canvas.renderer._renderer)
    plt.close()
    return image


colormap = [
    "blue",
    "orange",
    "green",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "red",
    "lime",
    "indigo",
    "violet",
    "aqua",
    "magenta",
    "coral",
    "gold",
    "tan",
    "skyblue",
]


def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)

    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction["polygons"], prediction["labels"]):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print("Invalid polygon:", _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    return image


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

    Parameters:
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

    Returns:
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    # Extract bounding boxes and labels
    bboxes = data.get("bboxes", [])
    labels = data.get("bboxes_labels", [])

    # Construct the output format
    od_results = {"bboxes": bboxes, "labels": labels}

    return od_results


def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction["quad_boxes"], prediction["labels"]
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2), "{}".format(label), align="right", fill=color)
    return image


class Florence2Pipeline(PlaygroundPipeline):
    def __init__(self, processor: AutoProcessor, model: AutoModelForCausalLM) -> None:
        super().__init__()

        self.register_modules(
            processor=processor,
            model=model,
        )
        self.processor = processor
        self.model = model

    @torch.no_grad()
    def __call__(self, image, task_prompt, text_input=None, max_new_tokens=1024):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        B, H, W, C = image.shape
        with AutoManage(self.model) as am:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs.to(am.get_device())

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(W, H))
        return parsed_answer


Florence2PipelineType = Annotated[Florence2Pipeline, make_widget("FLORENCE_2_PIPELINE")]

Florence2ParsedAnswerType = Annotated[dict, make_widget("FLORENCE_2_PARSED_ANSWER")]


@register_node(category=DEFAULT_CATEGORY)
def load_florence_2_pipeline(
    repoid: Annotated[
        str, ComboWidget(choices=["microsoft/Florence-2-large", "microsoft/Florence-2-large-ft"])
    ] = "microsoft/Florence-2-large-ft",
) -> tuple[Florence2PipelineType]:
    local_path = file_get_tool.find_or_download_huggingface_repo(
        [
            file_get_tool.FileSourceHuggingface(repo_id=repoid),
        ]
    )

    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)

    return (Florence2Pipeline(processor, model),)


tasks_answer_post_processing_type = {
    "<OCR>": "pure_text",
    "<OCR_WITH_REGION>": "ocr",
    "<CAPTION>": "pure_text",
    "<DETAILED_CAPTION>": "pure_text",
    "<MORE_DETAILED_CAPTION>": "pure_text",
    "<OD>": "description_with_bboxes",
    "<DENSE_REGION_CAPTION>": "description_with_bboxes",
    "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
    "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
    "<REGION_TO_SEGMENTATION>": "polygons",
    "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
    "<REGION_TO_CATEGORY>": "pure_text",
    "<REGION_TO_DESCRIPTION>": "pure_text",
    "<REGION_TO_OCR>": "pure_text",
    "<REGION_PROPOSAL>": "bboxes",
}


@register_node(category=DEFAULT_CATEGORY)
def run_Florence_2_pipeline(
    florence_2_pipeline: Florence2PipelineType,
    image: ImageType,
    task_prompt: Annotated[str, ComboWidget(choices={k.strip("<>"): k for k in tasks_answer_post_processing_type})],
    text_input: StringMultilineType,
    max_new_tokens: IntType = 1024,
) -> tuple[Florence2ParsedAnswerType]:
    text_input = text_input.strip()
    if len(text_input) <= 0:
        text_input = None
    parsed_answer = florence_2_pipeline(image * 255, task_prompt, text_input, max_new_tokens=max_new_tokens)
    return (parsed_answer,)


@register_node(category=DEFAULT_CATEGORY)
def florence_2_parsed_answer_to_string(
    florence_2_parsed_answer: Florence2ParsedAnswerType,
) -> tuple[StringType]:
    return (florence_2_parsed_answer,)


@register_node(category=DEFAULT_CATEGORY)
def florence_2_parsed_answer_plot(
    florence_2_parsed_answer: Florence2ParsedAnswerType, image: ImageType
) -> tuple[ImageType]:
    plot_images = []
    for key, data in florence_2_parsed_answer.items():
        if key in ("<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>", "<CAPTION_TO_PHRASE_GROUNDING>"):
            plot_image = plot_bbox(image[0], data)
            plot_images.append(plot_image)
        elif key == "<OPEN_VOCABULARY_DETECTION>":
            data = convert_to_od_format(data)
            plot_image = plot_bbox(image[0], data)
            plot_images.append(plot_image)
        elif key == "<OCR_WITH_REGION>":
            i = 255.0 * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            plot_image = draw_ocr_bboxes(img, data)
            plot_images.append(plot_image)
        elif key in ("<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>"):
            i = 255.0 * image[0].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            plot_image = draw_polygons(img, data, fill_mask=True)
            plot_images.append(plot_image)

    if len(plot_images) == 0:
        plot_images.append(np.zeros((16, 16, 3)))

    return (torch.Tensor(np.array(plot_images)) / 255,)
