import os
import os.path as osp
from glob import glob
import numpy as np
from config.config import cfg
import copy
import json
import pickle
import cv2
import torch
from pycocotools.coco import COCO
from util.human_models import smpl_x
from util.preprocessing import load_img, sanitize_bbox, process_bbox,augmentation_keep_size, load_ply, load_obj
from util.transforms import rigid_align, rigid_align_batch
import tqdm
import random
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
from humandata import HumanDataset
from detrsmpl.utils.demo_utils import xywh2xyxy, xyxy2xywh, box2cs
from detrsmpl.core.conventions.keypoints_mapping import convert_kps
import mmcv
import cv2
import numpy as np
from detrsmpl.core.visualization.visualize_keypoints2d import visualize_kp2d
from detrsmpl.core.visualization.visualize_smpl import visualize_smpl_hmr,render_smpl
from detrsmpl.models.body_models.builder import build_body_model
from detrsmpl.utils.geometry import estimate_cam_weakperspective_batch, pred_cam_to_transl,estimate_translation
from pytorch3d.io import save_obj
from detrsmpl.core.visualization.visualize_keypoints3d import visualize_kp3d
from detrsmpl.data.data_structures.multi_human_data import MultiHumanData

class INFERENCE_BEDLAM(torch.utils.data.Dataset):
    def __init__(self, img_dir=None,out_path=None):
        
        self.img_dir = os.path.join(img_dir,'**/*.png')
        self.out_path = out_path
        self.img_paths = sorted(glob(self.img_dir,recursive=True)) 
        assert not os.path.exists(os.path.join(self.out_path,'predictions')), "Predictions path already exists: {}".format(self.out_path)

        self.score_threshold = 0.85
        self.resolution = [720,1280] # AGORA test     
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
       
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_img(self.img_paths[idx],'BGR')
        img_whole_bbox = np.array([0, 0, img.shape[1],img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        cropped_img_shape=img.shape[:2]
        img = (img.astype(np.float32)) 
        

        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])}
        meta_info = {
            'ori_shape':np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx}
        result = {**inputs, **targets, **meta_info}
        
        result = self.normalize(result)
        result = self.format(result)
            
        return result
        
    def inference(self, outs):
        img_paths = self.img_paths
        sample_num = len(outs)
        output = {}
        
        for out in outs:
            ann_idx = out['image_idx']
            img_cropped = mmcv.imdenormalize(
                img=(out['img'].cpu().numpy()).transpose(1, 2, 0), 
                mean=np.array([123.675, 116.28, 103.53]), 
                std=np.array([58.395, 57.12, 57.375]),
                to_bgr=True).astype(np.uint8)
            # bb2img_trans = out['bb2img_trans']
            # img2bb_trans = out['img2bb_trans']
            scores = out['scores'].clone().cpu().numpy()
            img_shape = out['img_shape'].cpu().numpy()[::-1] # w, h
            
            img = cv2.imread(img_paths[ann_idx]) # h, w
            # os.makedirs(osp.join(self.out_path, 'vis'), exist_ok=True)
            # os.makedirs(osp.join(self.out_path, 'failed'), exist_ok=True)

                
            joint_proj = out['smplx_joint_proj'].clone().cpu().numpy()
            joint_vis = out['smplx_joint_proj'].clone().cpu().numpy()
            joint_coco = out['keypoints_coco'].clone().cpu().numpy()
            smpl_kp3d, _ = convert_kps(
                out['smpl_kp3d'].clone().cpu().numpy(),
                src='smplx',dst='agora', approximate=True)
            
            smpl_verts = out['smpl_verts'].clone().cpu().numpy()
            
            body_bbox = out['body_bbox'].clone().cpu().numpy()
            lhand_bbox = out['lhand_bbox'].clone().cpu().numpy()
            rhand_bbox = out['rhand_bbox'].clone().cpu().numpy()
            face_bbox = out['face_bbox'].clone().cpu().numpy()

            scale = np.array([
                    img.shape[1]/img_shape[0],
                    img.shape[1]/img_shape[0], 
                    img.shape[1]/img_shape[0], 
                    img.shape[1]/img_shape[0], 
                    ])

            body_bbox = body_bbox * scale
            lhand_bbox = lhand_bbox * scale
            rhand_bbox = rhand_bbox * scale
            face_bbox = face_bbox * scale

            joint_proj = joint_proj * scale[None, None, :2]
            joint_vis = joint_vis * joint_proj * scale[None, None, :2]

            for i, score in enumerate(scores):
                # if i ==24 or i==25 or i==23 or i==26:
                #     continue
                max_person_num=3
                # if score < self.score_threshold or i >max_person_num-1:
                if score < self.score_threshold:
                    break
                
                # for AGORA
                # save_name = img_paths[ann_idx].split('/')[-1][:-4] # if not crop should be -4
                # for bedlam 
                save_name = img_paths[ann_idx].split('/')[-4] + '_frameID_' + img_paths[ann_idx].split('/')[-1]
                save_name = save_name.replace('.png','')
                save_dict = {

                    'verts': smpl_verts[i].reshape(-1, 3),
                    'joints': joint_proj[i].reshape(-1, 2)[:24],
                    'allSmplJoints3d':smpl_kp3d[i].reshape(-1, 3), 
                    
                    }
                
                # save
                exist_result_path = glob(osp.join(self.out_path, 'predictions', save_name + '*'))
                if len(exist_result_path) == 0:
                    person_idx = 0
                else:
                    last_person_idx = max([
                        int(name.split('personId_')[1].split('.pkl')[0])
                        for name in exist_result_path
                    ])
                    person_idx = last_person_idx + 1

                save_name += '_personId_' + str(person_idx) + '.pkl'
                os.makedirs(osp.join(self.out_path, 'predictions'), exist_ok=True)
                with open(osp.join(self.out_path, 'predictions', save_name),'wb') as f:
                    pickle.dump(save_dict, f)
            
        return output

