import os
from dataclasses import dataclass

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN

from ..paper.arxiv.abs2304_05690.hybrik.models import HRNetSMPLXCamKidReg
from ..paper.arxiv.abs2304_05690.hybrik.utils.presets import SimpleTransform3DSMPLX
from ..paper.arxiv.abs2304_05690.hybrik.utils.vis import get_one_box
from .playground_pipeline import PlaygroundPipeline


det_transform = T.Compose([T.ToTensor()])

halpe_wrist_ids = [94, 115]
halpe_left_hand_ids = np.concatenate(
    [
        [5, 6, 7],
        [9, 10, 11],
        [17, 18, 19],
        [13, 14, 15],
        [1, 2, 3],
    ]
)

halpe_right_hand_ids = np.concatenate(
    [
        [5, 6, 7],
        [9, 10, 11],
        [17, 18, 19],
        [13, 14, 15],
        [1, 2, 3],
    ]
)

halpe_lhand_leaves = [8, 12, 20, 16, 4]
halpe_rhand_leaves = [8, 12, 20, 16, 4]


halpe_hand_ids = [i + 94 for i in halpe_left_hand_ids] + [i + 115 for i in halpe_right_hand_ids]
halpe_hand_leaves_ids = [i + 94 for i in halpe_lhand_leaves] + [i + 115 for i in halpe_rhand_leaves]


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def integral_hm(hms):
    # hms: [B, K, H, W]
    B, K, H, W = hms.shape
    hms = hms.sigmoid()
    hms = hms.reshape(B, K, -1)
    hms = hms / hms.sum(dim=2, keepdim=True)
    hms = hms.reshape(B, K, H, W)

    hm_x = hms.sum((2,))
    hm_y = hms.sum((3,))

    w_x = torch.arange(hms.shape[3]).to(hms.device).float()
    w_y = torch.arange(hms.shape[2]).to(hms.device).float()

    hm_x = hm_x * w_x
    hm_y = hm_y * w_y

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hms.shape[3]) - 0.5
    coord_y = coord_y / float(hms.shape[2]) - 0.5

    coord_uv = torch.cat((coord_x, coord_y), dim=2)
    return coord_uv


class HybrikXPipeline(PlaygroundPipeline):
    def __init__(
        self, transformation: SimpleTransform3DSMPLX, det_model: FasterRCNN, hybrik_model: HRNetSMPLXCamKidReg
    ) -> None:
        super().__init__()

        self.register_modules(
            transformation=transformation,
            det_model=det_model,
            hybrik_model=hybrik_model,
        )

    @property
    def tight_bbox(self):
        return self._tight_bbox

    @torch.no_grad()
    def __call__(self, input_image):
        # Run Detection
        det_input = det_transform(input_image).cuda()
        det_output = self.det_model([det_input])[0]

        self._tight_bbox = get_one_box(det_output)  # xyxy

        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = self.transformation.test_transform(input_image.copy(), self._tight_bbox)
        pose_input = pose_input[None, :, :, :].cuda()

        """
        pose_input_192 = pose_input[:, :, :, 32:-32].clone()

        # pose_input_192, bbox192 = al_transformation.test_transform(
        #     input_image.copy(), tight_bbox)
        # pose_input_192 = pose_input_192.to(opt.gpu)[None, :, :, :]
        pose_input_192[:, 0] = pose_input_192[:, 0] * 0.225
        pose_input_192[:, 1] = pose_input_192[:, 1] * 0.224
        pose_input_192[:, 2] = pose_input_192[:, 2] * 0.229
        al_output = alphapose_model(pose_input_192)
        al_uv_jts = integral_hm(al_output).squeeze(0)
        # hand_uv_jts = al_uv_jts[halpe_hand_ids, :]
        al_uv_jts[:, 0] = al_uv_jts[:, 0] * 192 / 256
        wrist_uv_jts = al_uv_jts[halpe_wrist_ids, :]
        hand_uv_jts = al_uv_jts[halpe_hand_ids, :]
        hand_leaf_uv_jts = al_uv_jts[halpe_hand_leaves_ids, :]
        """
        # vis 2d
        bbox_xywh = xyxy2xywh(bbox)
        """
        al_fb_pts = al_uv_jts.clone() * bbox_xywh[2]
        al_fb_pts[:, 0] = al_fb_pts[:, 0] + bbox_xywh[0]
        al_fb_pts[:, 1] = al_fb_pts[:, 1] + bbox_xywh[1]
        """

        pose_output = self.hybrik_model(
            pose_input,
            flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            # al_hands=hand_uv_jts.to(pose_input.device).unsqueeze(0).float(),
            # al_hands_leaf=hand_leaf_uv_jts.to(pose_input.device).unsqueeze(0).float(),
        )

        uv_jts = pose_output.pred_uvd_jts.reshape(-1, 3)[:, :2]
        # uv_jts[25:55, :2] = hand_uv_jts
        # uv_jts[-10:, :2] = hand_leaf_uv_jts
        transl = pose_output.transl.detach()

        # Visualization
        focal = 1000.0
        bbox_xywh = xyxy2xywh(bbox)
        transl_camsys = transl.clone()
        transl_camsys = transl_camsys * 256 / bbox_xywh[2]

        focal = focal / 256 * bbox_xywh[2]

        # vertices = pose_output.pred_vertices.detach()

        # verts_batch = vertices
        # transl_batch = translvideo_basename

        assert pose_input.shape[0] == 1, "Only support single batch inference for now"
        pred_uvd_jts = pose_output.pred_uvd_jts.reshape(-1, 3).cpu().data.numpy()
        pred_scores = pose_output.maxvals.cpu().data[:, :29].reshape(29).numpy()
        pred_camera = pose_output.pred_camera.squeeze(dim=0).cpu().data.numpy()
        pred_betas = pose_output.pred_shape_full.squeeze(dim=0).cpu().data.numpy()
        pred_theta = pose_output.pred_theta_mat.squeeze(dim=0).cpu().data.numpy()
        pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
        pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
        img_size = np.array((input_image.shape[0], input_image.shape[1]))

        res_db = {}
        res_db["pred_uvd"] = pred_uvd_jts
        res_db["pred_scores"] = pred_scores
        res_db["pred_camera"] = pred_camera
        res_db["f"] = focal
        res_db["pred_betas"] = pred_betas
        res_db["pred_thetas"] = pred_theta
        res_db["pred_phi"] = pred_phi
        res_db["pred_cam_root"] = pred_cam_root
        res_db["transl"] = transl[0].cpu().data.numpy()
        res_db["transl_camsys"] = transl_camsys[0].cpu().data.numpy()
        res_db["bbox"] = np.array(bbox)
        res_db["height"] = img_size[0]
        res_db["width"] = img_size[1]
        return res_db
