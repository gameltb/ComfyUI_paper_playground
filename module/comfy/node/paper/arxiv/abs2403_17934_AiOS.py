import os
from typing import Annotated

import comfy.utils
import mmcv
import numpy as np
import torch

from .....common import file_get_tool, path_tool
from .....core.runtime_resource_management import AutoManage
from .....paper.arxiv.abs2403_17934_AiOS.config import aios_smplx_inference
from .....paper.arxiv.abs2403_17934_AiOS.models.aios.aios_smplx import AiOSSMPLX, SetCriterion
from .....paper.arxiv.abs2403_17934_AiOS.models.registry import MODULE_BUILD_FUNCS
from .....paper.arxiv.abs2403_17934_AiOS.util.formatting import DefaultFormatBundle
from .....paper.arxiv.abs2403_17934_AiOS.util.preprocessing import augmentation_keep_size
from .....pipelines.playground_pipeline import PlaygroundPipeline
from .....utils.json import np_dumps
from ....registry import register_node
from ....types import ImageType, StringType, gen_widget


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get("img_fields", ["img"]):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={list(self.mean)}, "
        repr_str += f"std={list(self.std)}, "
        repr_str += f"to_rgb={self.to_rgb})"
        return repr_str


class AiOSPipeline(PlaygroundPipeline):
    def __init__(
        self, model: AiOSSMPLX, criterion: SetCriterion, postprocessors: dict, postprocessors_aios: dict
    ) -> None:
        super().__init__()

        self.register_modules(
            criterion=criterion,
            model=model,
        )
        self.criterion = criterion.eval()
        self.model = model.eval()
        self.postprocessors = postprocessors
        self.postprocessors_aios = postprocessors_aios

    @torch.no_grad()
    def __call__(self, data_batch, cfg):
        with AutoManage(self.model) as am:
            data_batch["img"] = data_batch["img"].unsqueeze(0).to(am.get_device())
            with torch.cuda.amp.autocast(enabled=True):
                outputs, targets, data_batch_nc = self.model(data_batch, cfg)

            orig_target_sizes = torch.stack([torch.Tensor(t["size"]) for t in targets], dim=0)
            result = self.postprocessors["bbox"](outputs, orig_target_sizes, targets, data_batch_nc)

        return result

    def out_postprocessors(self, outs, ori_imgs):
        output = []
        output_img = []

        for out, ori_img in zip(outs, ori_imgs):
            scores = out["scores"].clone().cpu().numpy()

            joint_proj = out["smplx_joint_proj"].clone().cpu().numpy()
            joint_vis = out["smplx_joint_proj"].clone().cpu().numpy()
            joint_coco = out["keypoints_coco"].clone().cpu().numpy()

            body_bbox = out["body_bbox"].clone().cpu().numpy()
            lhand_bbox = out["lhand_bbox"].clone().cpu().numpy()
            rhand_bbox = out["rhand_bbox"].clone().cpu().numpy()
            face_bbox = out["face_bbox"].clone().cpu().numpy()

            scale = out["bb2img_trans"][0].numpy()
            joint_proj[:, :, 0] = joint_proj[:, :, 0] * scale
            joint_proj[:, :, 1] = joint_proj[:, :, 1] * scale
            joint_vis[:, :, 0] = joint_vis[:, :, 0] * scale
            joint_vis[:, :, 1] = joint_vis[:, :, 1] * scale

            joint_coco[:, :, 0] = joint_coco[:, :, 0] * scale
            joint_coco[:, :, 1] = joint_coco[:, :, 1] * scale

            scale = np.array([scale, scale, scale, scale])

            body_bbox = body_bbox * scale
            lhand_bbox = lhand_bbox * scale
            rhand_bbox = rhand_bbox * scale
            face_bbox = face_bbox * scale

            for i, score in enumerate(scores):
                if score < 0.2:
                    break

                # save_name = img_paths[ann_idx].split("/")[-1][:-4]  # if not crop should be -4
                # if self.resolution == (2160, 3840):
                #     save_name = save_name.split("_ann_id")[0]
                # else:
                #     save_name = save_name.split("_1280x720")[0]

                save_dict = {
                    "params": {
                        "transl": out["cam_trans"][i].reshape(1, -1).cpu().numpy(),
                        "global_orient": out["smplx_root_pose"][i].reshape(1, -1).cpu().numpy(),
                        "body_pose": out["smplx_body_pose"][i].reshape(1, -1).cpu().numpy(),
                        "left_hand_pose": out["smplx_lhand_pose"][i].reshape(1, -1).cpu().numpy(),
                        "right_hand_pose": out["smplx_rhand_pose"][i].reshape(1, -1).cpu().numpy(),
                        "reye_pose": np.zeros((1, 3)),
                        "leye_pose": np.zeros((1, 3)),
                        "jaw_pose": out["smplx_jaw_pose"][i].reshape(1, -1).cpu().numpy(),
                        "expression": out["smplx_expr"][i].reshape(1, -1).cpu().numpy(),
                        "betas": out["smplx_shape"][i].reshape(1, -1).cpu().numpy(),
                    },
                    "joints": joint_proj[i].reshape(1, -1, 2)[0, :24],
                }

                output.append(save_dict)

            # show bbox
            show = False
            ori_img = mmcv.imshow_bboxes(ori_img, body_bbox[:i], show=show, colors="green")
            ori_img = mmcv.imshow_bboxes(ori_img, lhand_bbox[:i], show=show, colors="blue")
            ori_img = mmcv.imshow_bboxes(ori_img, rhand_bbox[:i], show=show, colors="yellow")
            ori_img = mmcv.imshow_bboxes(ori_img, face_bbox[:i], show=show, colors="red")
            output_img.append(ori_img)
        return output, output_img


AiOSPipelineType = Annotated[AiOSPipeline, gen_widget("AIOS_PIPELINE")]
AiOSFrameType = Annotated[dict, gen_widget("AIOS_FRAME")]


@register_node(category="arxiv/abs2403_17934_AiOS")
def load_aios() -> tuple[AiOSPipelineType]:
    model_path = file_get_tool.find_or_download_file(
        [
            file_get_tool.FileSource(
                loacal_folder=path_tool.get_data_path(__name__),
            ),
        ]
    )
    ckpt_path = os.path.join(model_path, "aios_checkpoint.pth")
    smplx_path = os.path.join(model_path, "smplx")

    cfg = aios_smplx_inference
    setattr(cfg, "eval", True)
    setattr(cfg, "inference", True)

    cfg.body_model_test = dict(
        type="smplx",
        keypoint_src="smplx",
        num_expression_coeffs=10,
        num_betas=10,
        keypoint_dst="smplx",
        model_path=smplx_path,
        use_pca=False,
        use_face_contour=True,
    )
    cfg.body_model_train = dict(
        type="smplx",
        keypoint_src="smplx",
        num_expression_coeffs=10,
        keypoint_dst="smplx_137",
        model_path=smplx_path,
        use_pca=False,
        use_face_contour=True,
    )
    cfg.body_model_post_process = dict(
        type="smplx",
        keypoint_src="smplx",
        num_expression_coeffs=10,
        num_betas=10,
        gender="neutral",
        keypoint_dst="smplx_137",
        model_path=smplx_path,
        use_pca=False,
        use_face_contour=True,
    )

    cfg.pretrained_model_path

    assert cfg.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    model, criterion, postprocessors, postprocessors_aios = build_func(cfg, cfg)

    checkpoint = comfy.utils.load_torch_file(ckpt_path)

    model.load_state_dict(checkpoint["model"])

    return (AiOSPipeline(model, criterion, postprocessors, postprocessors_aios),)


@register_node(category="arxiv/abs2403_17934_AiOS")
def run_aios(pipeline: AiOSPipelineType, image: ImageType) -> tuple[AiOSFrameType, ImageType]:
    img = image[0].numpy() * 255
    ori_img = img[:, :, ::-1]
    img = ori_img.copy()  # BGR,H,W,C,0-255
    img_whole_bbox = np.array([0, 0, img.shape[1], img.shape[0]])
    img, img2bb_trans, bb2img_trans, _, _ = augmentation_keep_size(img, img_whole_bbox, "test", aios_smplx_inference)

    img = img.astype(np.float32)

    inputs = {"img": img}
    targets = {"body_bbox_center": np.array(img_whole_bbox[None]), "body_bbox_size": np.array(img_whole_bbox[None])}
    meta_info = {
        # 'ori_shape':np.array(self.resolution),
        "img_shape": np.array(img.shape[:2]),
        "img2bb_trans": img2bb_trans,
        "bb2img_trans": bb2img_trans,
        # 'ann_idx': 0
    }
    result = {**inputs, **targets, **meta_info}

    format = DefaultFormatBundle()
    normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    result = normalize(result)
    result = format(result)
    for k, v in result.items():
        if hasattr(v, "datatype"):
            result[k] = v.data
    result["img_shape"] = [result["img_shape"]]
    result["body_bbox_center"] = [result["body_bbox_center"]]
    result["body_bbox_size"] = [result["body_bbox_size"]]
    result["ann_idx"] = [torch.Tensor([0])]
    result = pipeline(result, aios_smplx_inference)
    result, bbox_img = pipeline.out_postprocessors(result, [image[0].numpy()[:, :, ::-1]])
    return (result, torch.Tensor(bbox_img[0][:, :, ::-1].copy()).unsqueeze(0))


@register_node(category="arxiv/abs2403_17934_AiOS")
def aios_to_string(
    aios_frame: AiOSFrameType,
) -> tuple[StringType]:
    return (np_dumps(aios_frame),)
