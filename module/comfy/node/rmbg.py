import os

import comfy.utils
import torch
import torch.amp.autocast_mode
import torch.nn.functional as F

from ..registry import register_node
from ..types import ImageType, MaskType, gen_simple_new_type
from .briarmbg.briarmbg import BriaRMBG
from .briarmbg.utilities import preprocess_image
from ...core.runtime_resource_management import AutoManage

MODEL_PATH = os.path.join(os.path.dirname(__file__), "briarmbg", "pytorch_model.bin")


BriaRMBGType = gen_simple_new_type(BriaRMBG, "BRIA_RMBG")


@register_node(category="utils/rmbg")
def load_Bria_RMBG() -> tuple[BriaRMBGType]:
    net = BriaRMBG()
    ckpt = comfy.utils.load_torch_file(MODEL_PATH, safe_load=True)
    net.load_state_dict(ckpt)
    net.eval()

    return (net,)


@register_node(category="utils/rmbg")
def Bria_RMBG_predict(
    BriaRMBG: BriaRMBGType,
    input_image: ImageType,
) -> tuple[MaskType]:
    orig_im = input_image[0] * 255
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare input
    model_input_size = [1024, 1024]
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    with AutoManage(BriaRMBG, device):
        # inference
        result = BriaRMBG(image)

    result = torch.squeeze(F.interpolate(result[0][0], size=orig_im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)

    return (result.cpu(),)
