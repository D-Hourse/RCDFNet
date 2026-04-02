import os
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np
import cv2

from mmdet.models import DETECTORS
from rcdfnet.models.custum_detectors import MVXFasterRCNN
from .OFTNet import OftNet
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
from .BEVCross_modal_attention import Cross_Modal_Fusion
import matplotlib.pyplot as plt

import mmcv
from rcdfnet.core.bbox.transforms import bbox3d2result
from tools.misc.visualization import *
from tools.misc.visualization import draw_bev_pts_bboxes
from rcdfnet.core.bbox.transforms import bbox3d2result

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
class Best_RCFusion_FasterRCNN_yh_viz_vod(MVXFasterRCNN):
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
        super(Best_RCFusion_FasterRCNN_yh_viz_vod, self).__init__(**kwargs)
        self.rc_fusion = rc_fusion
        self.lift = camera_stream
        self.se = se

        ####------------viz------------
        self.figures_path_det3d_train = None
        self.figures_path_det3d_test = None
        self.vis_time_box3d = 0
        self.SAVE_INTERVALS = 1
        self.mean = np.array([103.53, 116.28, 123.675])
        self.std = np.array([1.0, 1.0, 1.0])
        self.xlim, self.ylim = [0, 51.2], [-25.6, 25.6]
        ####-----------draw heatmap------------
        self.figures_path_bevnd_test = None
        ####-----------LSS分支初始化--------
        if camera_stream:
            self.oft_BEVencoder = OftNet(img_bev_harf=img_bev_harf,grid_size=grid_size, grid_offset=grid_offset, grid_res=grid_res, grid_z_min=grid_z_min, grid_z_max=grid_z_max)
            # self.lss_BEVencoder =
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
            self.cross_attention = Cross_Modal_Fusion(kernel_size=3)
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

    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats_fusion'):
        # if bev_feats_name == 'bev_feats_fusion_refined': self.vis_time_bevnd += 1
        # if not self.vis_time_bevnd % self.SAVE_INTERVALS == 0: return

        figures_path_bevnd = self.figures_path_bevnd_test

        b, _, h, w = bev_feats.shape
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values  # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2])  # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i + 1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min()) / (bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            if bev_feats_name == 'bev_feats_img': bev_feats_tmp = bev_feats_tmp * 25
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3]
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd,
                                                            '_' + img_name + '_'  + '.png'))

    def draw_gt_pred_figures_3d(self, points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, outs_pts, rescale=False,
                                threshold=0.1, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return
        # filter out the ignored labels
        if self.training:
            figures_path_det3d = self.figures_path_det3d_train
        else:
            figures_path_det3d = self.figures_path_det3d_test
        gt_bboxes_3ds = [gt_bboxes_3ds[i][gt_labels_3ds[i] != -1] for i in range(len(img_metas))]
        outs_pts = outs_pts
        if outs_pts is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(*outs_pts, img_metas, rescale=False)
            bbox_list = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        else:
            bbox_list = None

        # starting visualization
        for i in range(imgs.shape[1]):  # batch size
            # preparation
            imgs = imgs.squeeze(dim=0)
            input_img = np.array(imgs[i].cpu()).transpose(1, 2, 0)
            input_img = input_img * self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d > threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            proj_mat = img_metas[i]["lidar2img"]  # update lidar2img
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            # project 3D bboxes to image and get show figures
            if pred_bboxes_3d is not None:
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None

            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
            result_path = figures_path_det3d
            mmcv.mkdir_or_exist(result_path)
            # show_multi_modality_result(img=input_img, gt_bboxes=gt_bboxes_3d, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            # draw in bev view
            save_path = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev.png')
            save_path_paper = os.path.join(figures_path_det3d,
                                           str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper.png')
            point = points[i].cpu().detach().numpy()[:, :3]
            pd_bbox_corners = pred_bboxes_3d.corners[:, [0, 2, 4, 6], :2].numpy()[:, (0, 1, 3, 2),
                              :] if pred_bboxes_3d is not None else None
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0, 2, 4, 6], :2].numpy()[:, (0, 1, 3, 2),
                              :] if gt_bboxes_3d is not None else None
            draw_bev_pts_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path, xlim=self.xlim,
                                ylim=self.ylim)
            # for paper figures
            tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, cv2.resize(input_img, (1936,1216)), proj_mat, img_metas,
                                                           color=(61, 102, 255), thickness=2, scale_factor=3)
            tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, cv2.resize(input_img, (1936,1216)), proj_mat, img_metas,
                                                           color=(241, 101, 72), thickness=2, scale_factor=3)
            tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true, proj_mat, img_metas,
                                                           color=(241, 101, 72), thickness=2, scale_factor=3)
            mmcv.imwrite(tmp_img_true, os.path.join(result_path, f'{filename}_gt.png'))
            mmcv.imwrite(tmp_img_pred, os.path.join(result_path, f'{filename}_pred.png'))
            mmcv.imwrite(tmp_img_alls, os.path.join(result_path, f'{filename}.png'))
            draw_paper_bboxes(point, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper, xlim=self.xlim,
                              ylim=self.ylim)

    def simple_test(self, points, img_metas, img=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        self.draw_bev_feature_map(pts_feats[0], img_metas)

        # # draw bev features map
        img_name = img_metas[0]['pts_filename'].split('/')[-1].split('.')[0]
        bev_feature_map = cv2.imread(os.path.join(self.figures_path_bevnd_test, '_' + img_name + '_' + '.png'))
        bev_feature_map = cv2.resize(bev_feature_map, (640, 640))
        bev_feature_map = cv2.rotate(bev_feature_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #
        # global num
        # cv2.imwrite('/home/chengpeifeng/Documents/VoD_mmdet3d/plot/bev_feature_results/bev' + str(num) + '.jpg', bev_feature_map)
        #
        # canvas_temp = gt_bev_mask[0][0].cpu().numpy().copy()
        # # 查找Counter
        # ret, thresh = cv2.threshold(np.uint8(canvas_temp), 0, 255, 0)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # # 绘制外接矩形
        # for contour in contours:
        #     contour = np.array(contour)
        #     rect = cv2.minAreaRect(contour)
        #     box_origin = cv2.boxPoints(rect)
        #     box_origin = np.int32(box_origin * (640 / 160))
        #     box_origin = box_origin.reshape((-1, 1, 2))
        #     cv2.polylines(bev_feature_map, [box_origin], isClosed=True, color=(0, 255, 0), thickness=2)

        # store结果图像
        cv2.imwrite(os.path.join(self.figures_path_bevnd_test, '_' + img_name + '_gt' + '.png'), bev_feature_map)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts, output_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
            # viz
            self.draw_gt_pred_figures_3d(points, img.unsqueeze(0), gt_bboxes_3d[0], gt_labels_3d[0], img_metas, output_pts)
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
                      depth=None,
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