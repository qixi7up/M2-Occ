# Modified by The M²-Occ Authors (M2Occ), 2026. Derived from SurroundOcc
# (Wei et al., ICCV 2023, Apache License 2.0). See NOTICE and LICENSE.

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# import open3d as o3d
from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_reconstruction, evaluation_semantic
from sklearn.metrics import confusion_matrix as CM
import time, yaml, os
import torch.nn as nn
import pdb


@DETECTORS.register_module()
class SurroundOcc(MVXTwoStageDetector):
    """SurroundOcc 3D scene reconstruction model, augmented with M2Occ's
    view-mask (RVM) and Multi-view Masked Reconstruction (MMR) modules.

    Highlights:
        1. Random view masking (RVM) to improve robustness.
        2. Multi-view Masked Reconstruction (MMR) module to handle missing views.
        3. Supports both semantic and geometric occupancy prediction.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_mmr=True,
                 use_semantic=True,
                 is_vis=False,
                 version='v1',
                 ):

        super(SurroundOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.use_semantic = use_semantic
        self.is_vis = is_vis

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract multi-scale image features.

        Args:
            img (Tensor): input image tensor [B, N, C, H, W].
            img_metas (list): list of image meta infos.
            len_queue (int, optional): temporal queue length.

        Returns:
            list[Tensor]: multi-scale feature maps.
        """
        B = img.size(0)
        if img is not None:
            # handle 5-D input (B, N, C, H, W)
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)  # apply grid mask augmentation

            img_feats = self.img_backbone(img)  # extract features with backbone
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)  # FPN feature fusion
        # reshape feature dimensions
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images (this implementation uses images only)."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas):
        """Training forward pass of the occupancy prediction head.

        Args:
            pts_feats (list): multi-scale feature maps.
            gt_occ (Tensor): 3D occupancy ground truth.
            img_metas (list): image meta infos.

        Returns:
            dict: loss dictionary.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas)
        loss_inputs = [gt_occ, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        """Dummy forward (used to compute FLOPs)."""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      gt_occ=None,
                      img=None
                      ):
        """Training-mode forward pass.

        Args:
            img_metas (list): image meta infos.
            gt_occ (Tensor): 3D occupancy ground truth.
            img (Tensor): input image.

        Returns:
            dict: dictionary containing the losses.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_occ,
                                             img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        """Test-mode forward pass.

        Args:
            img_metas (list): image meta infos.
            img (Tensor): input image.
            gt_occ (Tensor): 3D occupancy ground truth (for evaluation).

        Returns:
            dict: evaluation results or visualization output.
        """
        output = self.simple_test(
            img_metas, img, **kwargs)

        pred_occ = output['occ_preds']  # get predictions
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]  # take the highest-level prediction
        # visualization mode
        if self.is_vis:
            self.generate_output(pred_occ, img_metas)
            return pred_occ.shape[0]
        # semantic segmentation evaluation
        if self.use_semantic:
            class_num = pred_occ.shape[1]
            _, pred_occ = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
            eval_results = evaluation_semantic(pred_occ, gt_occ, img_metas[0], class_num)
        # geometric reconstruction evaluation
        else:
            pred_occ = torch.sigmoid(pred_occ[:, 0])
            eval_results = evaluation_reconstruction(pred_occ, gt_occ, img_metas[0])
        return {'evaluation': eval_results}

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Occupancy prediction test (without augmentation)."""
        outs = self.pts_bbox_head(x, img_metas)

        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        output = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

        return output

    def generate_output(self, pred_occ, img_metas):
        """Generate visualization output (point-cloud format).

        Args:
            pred_occ (Tensor): predicted 3D occupancy.
            img_metas (list): image meta infos.
        """
        import open3d as o3d
        # 17-class color map (RGBA)
        color_map = np.array(
                [
                    [0, 0, 0, 255],       # background
                    [255, 120, 50, 255],  # barrier              orangey
                    [255, 192, 203, 255],  # bicycle              pink
                    [255, 255, 0, 255],  # bus                  yellow
                    [0, 150, 245, 255],  # car                  blue
                    [0, 255, 255, 255],  # construction_vehicle cyan
                    [200, 180, 0, 255],  # motorcycle           dark orange
                    [255, 0, 0, 255],  # pedestrian           red
                    [255, 240, 150, 255],  # traffic_cone         light yellow
                    [135, 60, 0, 255],  # trailer              brown
                    [160, 32, 240, 255],  # truck                purple
                    [255, 0, 255, 255],  # driveable_surface    dark pink
                    # [175,   0,  75, 255],       # other_flat           dark red
                    [139, 137, 137, 255],
                    [75, 0, 75, 255],  # sidewalk             dard purple
                    [150, 240, 80, 255],  # terrain              light green
                    [230, 230, 250, 255],  # manmade              white
                    [0, 175, 0, 255],  # vegetation           green
                ]
            )
        # get predictions (semantic or geometric)
        if self.use_semantic:
            _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        else:
            voxel = torch.sigmoid(pred_occ[:, 0])
        # process each sample
        for i in range(voxel.shape[0]):
            # build 3D grid coordinates
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2])
            X, Y, Z = torch.meshgrid(x, y, z)
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)
            # extract occupied voxel vertices
            vertices = vv[voxel[i] > 0.5]
            # convert voxel coordinates to world coordinates
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) /  img_metas[i]['occ_size'][0]  + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) /  img_metas[i]['occ_size'][1]  + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) /  img_metas[i]['occ_size'][2]  + img_metas[i]['pc_range'][2]

            vertices = vertices.cpu().numpy()
            # create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            # for semantic predictions, add colors
            if self.use_semantic:
                semantics = voxel[i][voxel[i] > 0].cpu().numpy()
                color = color_map[semantics] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
                vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)
            # save results
            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)

            o3d.io.write_point_cloud(os.path.join(save_dir, 'pred.ply'), pcd)
            np.save(os.path.join(save_dir, 'pred.npy'), vertices)
            # save input images
            for cam_id, cam_path in enumerate(img_metas[i]['filename']):
                os.system('cp {} {}/{}.jpg'.format(cam_path, save_dir, cam_id))
