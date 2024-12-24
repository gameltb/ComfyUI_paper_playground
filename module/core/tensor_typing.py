from typing import Annotated, Optional

import torch
from torch.export import Dim

from .annotated_model import AnnotatedBaseModel, find_annotated_model


class TensorModel(AnnotatedBaseModel):
    dims: Optional[list[type]] = None
    dtype: Optional[str] = None


Tensor = Annotated[torch.Tensor, TensorModel()]

DIMS = "dims"
DTYPE = "dtype"

BATCH_DIM = Dim("batch")

if __name__ in ("__main__", "<run_path>"):
    print(find_annotated_model(Annotated[Tensor, DIMS : [Dim("C", min=1)]], model_type=TensorModel))
