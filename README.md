# ComfyUI paper playground

Evaluate some papers in ComfyUI, just playground.

## IP-Adapter

Put https://huggingface.co/h94/IP-Adapter to `ComfyUI/models/diffusers/IP-Adapter`.

# Paper

## [PIA：Personalized Image Animator](https://github.com/open-mmlab/PIA)

Put PIA Checkpoint to `ComfyUI/models/playground/paper/arxiv/abs2312_13964/pia/pia.ckpt`.  
Put https://huggingface.co/runwayml/stable-diffusion-v1-5 to `ComfyUI/models/diffusers/stable-diffusion-v1-5`.  
Put majicmixRealistic(https://civitai.com/models/43331) model to `ComfyUI/models/checkpoints/`.

### Example

![](assets/abs2312_13964.png)

## [Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation](https://github.com/prs-eth/Marigold)

Put https://huggingface.co/Bingxin/Marigold to `ComfyUI/models/diffusers/Marigold`.

### Example

![](assets/abs2312_02145.png)

## [HybrIK: Hybrid Analytical-Neural Inverse Kinematics for Body Mesh Recovery](https://github.com/Jeff-sjtu/HybrIK)

Refer to the README.

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Unzip model_files to directory `ComfyUI/models/playground/paper/arxiv/abs2304_05690/` .  
Put HybrIK-X rle model to `ComfyUI/models/playground/paper/arxiv/abs2304_05690/hybrikx/`.
