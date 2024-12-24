import logging
from typing import Optional, TypeVar, Union

from pydantic import BaseModel

_logger = logging.getLogger(__name__)


class AnnotatedBaseModel(BaseModel):
    pass


M = TypeVar("M", bound=AnnotatedBaseModel)


def find_annotated_model(annotation: Union[type, M], model_type: type[M] = AnnotatedBaseModel) -> Optional[M]:
    if isinstance(annotation, model_type):
        return annotation
    elif hasattr(annotation, "__metadata__"):
        annotated_model = None
        build_kwargs_list: list[slice] = []
        for meta in reversed(annotation.__metadata__):
            if isinstance(meta, model_type):
                annotated_model = meta
                break
            elif type(meta) is slice:
                build_kwargs_list.append(meta)
        if annotated_model is not None:
            build_kwargs = {}
            for build_kwarg in reversed(build_kwargs_list):
                build_kwargs[build_kwarg.start] = build_kwarg.stop

            build_kwargs_kset = set(build_kwargs.keys())
            model_fields_set = set(annotated_model.model_fields.keys())
            for k in build_kwargs_kset - model_fields_set:
                _logger.warning(f"Ignored annotated key {k}:{build_kwargs[k]}.")

            annotated_model = annotated_model.model_copy(update=build_kwargs)
        return annotated_model
    return None
