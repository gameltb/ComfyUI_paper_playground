from typing import Annotated, Optional

import torch
from annotated_model import AnnotatedBaseModel, find_annotated_model


class TensorModel(AnnotatedBaseModel):
    dims: Optional[list[type]] = None
    dtype: Optional[str] = None


Tensor = Annotated[torch.Tensor, TensorModel()]

DIMS = "dims"
DTYPE = "dtype"

if __name__ in ("__main__", "<run_path>"):
    from torch.export import Dim

    print(find_annotated_model(Annotated[Tensor, DIMS : [Dim("C", min=1)]], model_type=TensorModel))
