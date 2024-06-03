# Copyright (c) OpenMMLab. All rights reserved.

import torch

# from detrsmpl.core.post_processing.bbox.transforms
# import bbox_cxcywh_to_xyxy
from ..match_costs import build_match_cost
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from .builder import BBOX_ASSIGNERS

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """
    def __init__(
        self,
        #  cls_cost=dict(type='ClassificationCost', weight=1.),
        kp3d_cost=dict(type='Keypoints3DCost', covention='smpl_54',
                       weight=1.0),
        kp2d_cost=dict(type='Keypoints2DCost', covention='smpl_54',
                       weight=1.0),
    ):
        # self.cls_cost = build_match_cost(cls_cost)
        self.kp2d_cost = build_match_cost(kp2d_cost)
        self.kp3d_cost = build_match_cost(kp3d_cost)

    def assign(
        self,
        pred_smpl_pose,
        pred_smpl_shape,
        pred_kp3d,
        pred_vert,
        pred_cam,
        gt_smpl_pose,
        gt_smpl_shape,
        gt_kp2d,
        gt_kp3d,
        has_keypoints2d,
        has_keypoints3d,
        has_smpl,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
        # pred_smpl_orient,
        # pred_keypoints2d,
        # gt_bboxes,
        # gt_labels,
    ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_smpl_pose.size(0), pred_smpl_pose.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = pred_smpl_pose.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)
        assigned_labels = pred_smpl_pose.new_full((num_bboxes, ),
                                                  -1,
                                                  dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts,
                                assigned_gt_inds,
                                None,
                                labels=assigned_labels)
        # img_h, img_w, _ = img_meta['img_shape']
        # factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
        #                                img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        # cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        # normalize_gt_bboxes = gt_bboxes / factor

        # kp3d_cost
        kp3d_cost = self.kp3d_cost(pred_kp3d, gt_kp3d)

        # kp2d_cost
        kp2d_cost = self.kp2d_cost(pred_kp3d, pred_cam, gt_kp2d)
        # smpl_pose_cost

        # smpl_betas_cost

        # verts_cost

        # TODO: bbox_cost

        # TODO: occlusion == bbox insecaa

        # regression iou cost, defaultly giou is used in official DETR.
        # bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        # iou_cost = self.iou_cost(pred_smpl_pose, gt_smpl_pose)
        # weighted sum of above three costs
        cost = kp2d_cost  # + kp3d_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            pred_smpl_pose.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            pred_smpl_pose.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        # assigned_labels[matched_row_inds] = None # gt_smpl_pose[matched_col_inds]
        assigned_labels = None
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            None,
                            labels=assigned_labels)

        # num_gt: instance_num
        # assigned_gt_inds: self.gt_inds
        #
