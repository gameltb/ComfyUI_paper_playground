import os
from typing import Annotated

import comfy.utils
import torch
import torch.amp.autocast_mode
import torch.nn.functional as F

from ...common import file_get_tool, import_tool
from ...core.runtime_resource_management import AutoManage
from ..registry import register_node
from ..types import ImageType, MaskType, make_widget

REPO_PATH = file_get_tool.find_or_download_huggingface_repo(
    [
        file_get_tool.FileSourceHuggingface(repo_id="briaai/RMBG-1.4"),
    ]
)

briarmbg = import_tool.module_from_file(os.path.join(REPO_PATH, "briarmbg.py"))
utilities = import_tool.module_from_file(os.path.join(REPO_PATH, "utilities.py"))


BriaRMBGType = Annotated[briarmbg.BriaRMBG, make_widget("BRIA_RMBG")]


@register_node(category="utils/rmbg")
def load_Bria_RMBG() -> tuple[BriaRMBGType]:
    model_path = os.path.join(REPO_PATH, "pytorch_model.bin")
    net = briarmbg.BriaRMBG()
    ckpt = comfy.utils.load_torch_file(model_path, safe_load=True)
    net.load_state_dict(ckpt)
    net.eval()

    return (net,)


@register_node(category="utils/rmbg")
def Bria_RMBG_predict(
    BriaRMBG: BriaRMBGType,
    input_image: ImageType,
) -> tuple[MaskType]:
    orig_im = input_image[0] * 255
    with AutoManage(BriaRMBG) as am:
        device = am.get_device()

        # prepare input
        model_input_size = [1024, 1024]
        orig_im_size = orig_im.shape[0:2]
        image = utilities.preprocess_image(orig_im, model_input_size).to(device)

        # inference
        result = BriaRMBG(image)

    result = torch.squeeze(F.interpolate(result[0][0], size=orig_im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)

    return (result.cpu(),)
