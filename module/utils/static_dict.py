import os

import safetensors
import safetensors.torch
import torch


def load_state_dict(file_path: str, device="cpu"):
    model = {}

    _, ext = os.path.splitext(file_path)

    if ext in (".safetensors", ".st"):
        model = safetensors.torch.load_file(file_path, device=device)
    else:
        model = torch.load(file_path, map_location=device, mmap=True, weights_only=True)

    return model
