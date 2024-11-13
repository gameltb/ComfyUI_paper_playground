import shutil
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

from ..registry import register_node
from ..types import ImageType, MaskType, StringType


@register_node(category="utils")
def load_image_resource_url(url: StringType) -> tuple[ImageType, MaskType]:
    r = requests.get(
        url,
        stream=True,
        # headers=def_headers,
        # proxies=proxies,
    )
    if not r.ok:
        print("Get error code: " + str(r.status_code))
        print(r.text)
        raise Exception(r.text)

    r.raw.decode_content = True

    buff = BytesIO()
    shutil.copyfileobj(r.raw, buff)
    buff.seek(0)

    i = Image.open(buff)
    i = ImageOps.exif_transpose(i)
    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    if "A" in i.getbands():
        mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image, mask.unsqueeze(0))
