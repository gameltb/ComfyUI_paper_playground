from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from PIL import Image


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


class ToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class ImageToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


class Transpose(object):
    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


class ToPIL(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = Image.fromarray(results['img'])
        return results


class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = np.array(results['img'], dtype=np.float32)
        return results


class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img" and "gt_label".

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('filename', 'ori_shape', 'img_shape', 'flip',
            'flip_direction', 'img_norm_cfg')``

    Returns:
        dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas`` if available
    """
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'flip', 'flip_direction',
                            'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


class ToDataContainer:
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True), dict(key='gt_bboxes'),
            dict(key='gt_labels'))``.
    """
    def __init__(self,
                 fields=(dict(key='img', stack=True), dict(key='gt_bboxes'),
                         dict(key='gt_labels'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to \
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


class DefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
                       (3)to DataContainer (stack=True)

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """
    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """
        data_keys = [
            'joint_img',  # keypoints2d
            'smplx_joint_img',  #smplx_joint_img, # projected smplx if valid cam_param, else same as keypoints2d
            'joint_cam',  # joint_cam actually not used in any loss, # raw kps3d probably without ra
            'smplx_joint_cam',  # kps3d with body, face, hand ra
            'smplx_pose',
            'smplx_shape',
            'smplx_expr',
            'lhand_bbox_center',
            'lhand_bbox_size',
            'rhand_bbox_center',
            'rhand_bbox_size',
            'face_bbox_center',
            'face_bbox_size',
            'body_bbox_center',
            'body_bbox_size',
            'joint_valid',
            'joint_trunc',
            'smplx_joint_valid',
            'smplx_joint_trunc',
            'smplx_pose_valid',
            'smplx_shape_valid',
            'smplx_expr_valid',
            'is_3D',
            'lhand_bbox_valid',
            'rhand_bbox_valid',
            'face_bbox_valid',
            'body_bbox_valid',
            'body_bbox',
            'lhand_bbox',
            'rhand_bbox',
            'face_bbox',
            'gender',
            'bb2img_trans',
            'img2bb_trans',
            'ann_idx'
        ]
        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys

            # import ipdb;ipdb.set_trace()
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # import ipdb;ipdb.set_trace()
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img),
                                padding_value=self.pad_val['img'],
                                stack=True)
        for key in data_keys:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        # if 'gt_masks' in results:
        #     results['gt_masks'] = DC(
        #         results['gt_masks'],
        #         padding_value=self.pad_val['masks'],
        #         cpu_only=True)
        # if 'gt_semantic_seg' in results:
        #     results['gt_semantic_seg'] = DC(
        #         to_tensor(results['gt_semantic_seg'][None, ...]),
        #         padding_value=self.pad_val['seg'],
        #         stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(mean=np.zeros(num_channels, dtype=np.float32),
                 std=np.ones(num_channels, dtype=np.float32),
                 to_rgb=False))
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'


class WrapFieldsToLists(object):
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapIntoLists')
        >>> ]
    """
    def __call__(self, results):
        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'
