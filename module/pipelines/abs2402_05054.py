import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..paper.arxiv.abs2402_05054.core.models import LGM
from .playground_pipeline import PlaygroundPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class LGMPipeline(PlaygroundPipeline):
    def __init__(self, lgm: LGM) -> None:
        super().__init__()
        self.register_modules(
            lgm=lgm,
        )
        self.lgm = lgm

    @torch.no_grad()
    def __call__(self, mv_image, input_elevation=0):
        device = "cuda"
        input_image = np.stack(
            [mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0
        )  # [4, 256, 256, 3], float32
        input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device)  # [4, 3, 256, 256]
        input_image = F.interpolate(
            input_image, size=(self.lgm.opt.input_size, self.lgm.opt.input_size), mode="bilinear", align_corners=False
        )
        input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        rays_embeddings = self.lgm.prepare_default_rays(device, elevation=input_elevation)
        input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]

        with torch.autocast(device_type=device, dtype=torch.float16):
            # generate gaussians
            gaussians = self.lgm.forward_gaussians(input_image)

        # save gaussians
        return self.lgm.gs.to_ply(gaussians)
