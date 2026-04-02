import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from rcdfnet.models.custum_detectors import MVXFasterRCNN
from .OFTNet import OftNet
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
from .BEVCross_modal_attention import Cross_Modal_Fusion,Not_Cross_Modal_Fusion

class SE_Block(nn.Module):  ##基于SE融合
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.att(x)


@DETECTORS.register_module()
class RCFusion_FasterRCNN(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self,img_bev_harf=False, rc_fusion="concat", camera_stream=True,
                 grid_size=(69.12,79.36), grid_offset=(0,-39.68, 0), grid_res=0.32, grid_z_min=-3,
                 grid_z_max=2.76, se=False, imc=256,radc=384, **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(RCFusion_FasterRCNN, self).__init__(**kwargs)
        self.rc_fusion = rc_fusion
        self.lift = camera_stream
        self.se = se
        ####-----------LSS分支初始化--------
        if camera_stream:
            self.oft_BEVencoder = OftNet(img_bev_harf=img_bev_harf,grid_size=grid_size, grid_offset=grid_offset, grid_res=grid_res, grid_z_min=grid_z_min, grid_z_max=grid_z_max)
        if rc_fusion == "concat":  ##融合层
            if se:
                self.seblock = SE_Block(radc)
            self.reduc_conv = ConvModule(
                radc + imc,
                radc,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        elif rc_fusion == "cross_attention":
            # self.reduc_radar_conv = ConvModule(
            #     radc,
            #     imc,
            #     1,
            #     padding=0,
            #     conv_cfg=None,
            #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            #     act_cfg=dict(type='ReLU'),
            #     inplace=False)
            #------------------
            self.cross_attention = Cross_Modal_Fusion(kernel_size=3)

            #------------------XIAORONG----------------------
            #self.cross_attention = Not_Cross_Modal_Fusion(kernel_size=3)

        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights()
        self.freeze()

    ####-----------------------冻结图像部分分支---------------------------
    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            # if self.lift:
            #     for param in self.lift_splat_shot_vis.parameters():
            #         param.requires_grad = False

    def extract_pts_feat(self, pts, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0].item() + 1    ###这里加了.item()
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)  ##图像特征
        pts_feats = self.extract_pts_feat(points, img_metas)  # 点云特征ortho  ###(B, 384, 124, 108)
        # -------------提升的过程写在此处--------------------
        if self.lift:
            ###判断batch一致
            assert img_feats[0].shape[0] == pts_feats[0].shape[0] == len(img_metas)
            batch_size, _, _, _ = img_feats[0].shape
            calib = []
            for sample_idx in range(batch_size):
                mat = img_metas[sample_idx]['lidar2img']
                mat = torch.Tensor(mat[:3,:]).to(img_feats[0].device)
                calib.append(mat)
            calib = torch.stack(calib)

            img_bev_feat = self.oft_BEVencoder(img_feats, calib)  ###(B, 256, 124, 108)
            # print(img_bev_feat.shape, pts_feats[-1].shape)
            if pts_feats is None:
                pts_feats = [img_bev_feat]  ####cam stream only
            else:
                # --------融合成为pts_feats----------------
                if self.rc_fusion == "concat":  # 大小shape不一致，用插值
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear',
                                                     align_corners=True)
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
                elif self.rc_fusion == "cross_attention":
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear',
                                                     align_corners=True)
                    #pts_feats = [self.reduc_radar_conv(pts_feats[0])]
                    pts_feats = [self.cross_attention(img_bev_feat, pts_feats[0])]
        return dict(
            img_feats=img_feats,
            pts_feats=pts_feats,
        )
        # return (img_feats, pts_feats, depth_dist)

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(  ## -------这里没有用到-------
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

if __name__=="__main__":
    img_bev = torch.randn((2, 256, 124, 108))
    radar_bev = torch.randn((2, 256, 124, 108))
    cross_attention = Transformer_cross_attention(dim=256, num_heads=8, qkv_bias=True)
    out = cross_attention(img_bev,radar_bev)
    print(out.shape)