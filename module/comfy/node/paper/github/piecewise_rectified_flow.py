from diffusers import UNet2DConditionModel

from .....paper.github.piecewise_rectified_flow.src.scheduler_perflow import PeRFlowScheduler
from .....paper.github.piecewise_rectified_flow.src.utils_perflow import merge_delta_weights_into_unet
from ....registry import register_node
from ...diffusers import DiffusersComponentType, DiffusersPipelineType


@register_node(category="github/piecewise_rectified_flow")
def apply_piecewise_rectified_flow(
    diffusers_pipeline: DiffusersPipelineType,
    diffusers_component: DiffusersComponentType,
) -> tuple[DiffusersPipelineType]:
    """component from hansyan/perflow-sd15-delta-weights, variant="v0-1"."""
    assert isinstance(diffusers_component, UNet2DConditionModel)
    delta_weights = diffusers_component.state_dict()
    diffusers_pipeline = merge_delta_weights_into_unet(diffusers_pipeline, delta_weights)
    diffusers_pipeline.scheduler = PeRFlowScheduler.from_config(
        diffusers_pipeline.scheduler.config, prediction_type="epsilon", num_time_windows=4
    )
    return (diffusers_pipeline,)
