import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from rcdfnet.models.custum_detectors import MVXFasterRCNN, RC_backbone
from .OFTNet import OftNet
from .LSSNet import LSSNet
from ..module.EMCAD import EMCAD
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
from .BEVCross_modal_attention import Cross_Modal_Fusion, Cross_Modal_Fusion_test
from .BEV_deformable_fuser import BEVFuser
from ..module.MCMA import MCAM
from plot.heatmap import draw_feature_fuser_map, draw_feature_pts_map, draw_feature_img_LSS_map, draw_feature_img_OFT_map, save_image_tensor2cv2, draw_feature_img_fuser_map

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
class Best_RCFusion_vodseg_OFT1(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self,image_size=None, img_bev_harf=False, rc_fusion="concat", camera_stream=True, segmentation=False,
                 grid_size=(69.12,79.36), grid_offset=(0,-39.68, 0), grid_res=0.32, grid_z_min=-3, num_class=3,
                 grid_z_max=2.76, se=False, imc=256,radc=384,x_bound=[0, 69.12, 0.32],y_bound=[-39.68, 39.68, 0.32],
                 z_bound=[-3,2.76,5.76],d_bound=[0, 69,12, 0.32], depth_channels=80, final_dim = (832, 1280),
                 downsample_factor=32, depth_net_in_channels=256, depth_net_mid_channels=256, return_depth=False,
                 loss_depth_weight=None, loss_seg_weight=None,**kwargs):
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
        img_seg_decoder_config = kwargs.get('img_seg_decoder', None)
        kwargs.pop('img_seg_decoder')
        img_seg_head_config = kwargs.get('img_seg_head', None)
        kwargs.pop('img_seg_head')
        freeze_radar = kwargs.get('freeze_radar', None)
        kwargs.pop('freeze_radar')
        super(Best_RCFusion_vodseg_OFT1, self).__init__(**kwargs)
        self.rc_fusion = rc_fusion
        self.lift = camera_stream
        self.segmentation = segmentation
        self.se = se
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.z_bound = z_bound
        self.d_bound = d_bound
        self.depth_channels = depth_channels
        self.image_size = image_size
        self.downsample = downsample_factor
        self.return_depth = return_depth
        self.loss_depth_weight = loss_depth_weight
        self.loss_seg_weight = loss_seg_weight
        self.num_classes = num_class

        ####-----------语义分割分支---------
        if self.segmentation:
            self.seg_decoder = EMCAD(**img_seg_decoder_config)
            self.seg_head = nn.Conv2d(img_seg_head_config['in_channels'], img_seg_head_config['out_classes'], img_seg_head_config['kernel'])

        ####-----------LSS分支初始化--------
        if camera_stream:
            self.oft_BEVencoder = OftNet(img_bev_harf=img_bev_harf, grid_size=grid_size, grid_offset=grid_offset,
                                         grid_res=grid_res, grid_z_min=grid_z_min, grid_z_max=grid_z_max)
            self.lss_BEVencoder = LSSNet(x_bound=x_bound, y_bound=y_bound, z_bound=z_bound,
                                         d_bound=d_bound, final_dim=final_dim, downsample_factor=downsample_factor,
                                         depth_net_conf=dict(in_channels=depth_net_in_channels,
                                                             mid_channels=depth_net_mid_channels),
                                         depth_channels=self.depth_channels, return_depth=return_depth)
            self.bev_attention = BEVFuser(bev1_dims=256, bev2_dims=256, embed_dims=256, num_layers=4, num_heads=4,
                                          bev_shape=(160, 160))
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
            self.reduc_radar_conv = ConvModule(
                radc,
                imc,
                1,
                padding=0,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
            self.cross_attention = Cross_Modal_Fusion(kernel_size=3)
            # self.cross_attention = BEVFuser(bev1_dims=256, bev2_dims=384, embed_dims=256, num_layers=2, num_heads=4, bev_shape=(160, 160))
            # self.cross_attention = MCAM(in_channels=256, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True)
        self.freeze_img = kwargs.get('freeze_img', False)
        self.freeze_radar = freeze_radar
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
        if self.freeze_radar:
            if self.with_pts_backbone:
                for param in self.pts_backbone.parameters():
                    param.requires_grad = False
            if self.with_pts_neck:
                for param in self.pts_neck.parameters():
                    param.requires_grad = False
            if self.with_pts_bbox:
                for param in self.pts_bbox_head.parameters():
                    param.requires_grad = False

    def extract_pts_feat(self, pts, img_metas):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors, )
        batch_size = coors[-1, 0].item() + 1  ###这里加了.item()
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        # output_path = '/plot/Best_RCFusion_fasterRCNN/output_image.png'
        img_feats = self.extract_img_feat(img, img_metas)  ##图像特征
        # print(img_feats[0].size(),img_feats[1].size(),img_feats[2].size(),img_feats[3].size(),img_feats[4].size())
        pts_feats = self.extract_pts_feat(points, img_metas)  # 点云特征ortho  ###(B, 384, 124, 108)
        # pts_feats = None
        # -------------segmentation-----------------------
        if self.segmentation:
            seg_decoder_output = self.seg_decoder(img_feats)
            img_feats = seg_decoder_output
            seg_output = self.seg_head(seg_decoder_output[0])
            seg_output = F.interpolate(seg_output, size=self.image_size[1], mode='bilinear', align_corners=True)
            seg_output = seg_output.softmax(1)
        # -------------提升的过程写在此处--------------------
        if self.lift:
            ###判断batch一致
            batch_size, _, _, _ = img_feats[0].shape
            calib = []
            for sample_idx in range(batch_size):
                mat = img_metas[sample_idx]['lidar2img']
                mat = torch.Tensor(mat[:3, :]).to(img_feats[0].device)
                calib.append(mat)
            calib = torch.stack(calib)

            if self.return_depth:
                img_bev_feat_L_scale = self.oft_BEVencoder(img_feats, calib)
                img_bev_feat_S_scale, depth = self.lss_BEVencoder(img_feats, calib)
                depth = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
            else:
                img_bev_feat_L_scale = self.oft_BEVencoder(img_feats, calib)
                img_bev_feat_S_scale = self.lss_BEVencoder(img_feats, calib)  ###(B, 256, 124, 108)

            # draw_feature_img_LSS_map(img_bev_feat_S_scale)
            # draw_feature_img_OFT_map(img_bev_feat_L_scale)
            img_bev_feat = self.bev_attention(img_bev_feat_L_scale, img_bev_feat_S_scale)
            # draw_feature_img_fuser_map(img_bev_feat)
            # img_bev_feat = img_bev_feat_S_scale
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
                    # draw_feature_pts_map(pts_feats[0])
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear',
                                                     align_corners=True)
                    # pts_feats = [self.reduc_radar_conv(pts_feats[0])]
                    pts_feats = [self.cross_attention(img_bev_feat, pts_feats[0])]
                    # draw_feature_fuser_map(pts_feats[0])
                    # print("draw done")
        return_dict = dict(img_feats=img_feats, pts_feats=pts_feats)
        if self.return_depth:
            return_dict.update(depth_preds=depth)
        if self.segmentation:
            return_dict.update(seg_preds=seg_output)
        return return_dict

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)

        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        ####################################
        # pts_feats = self.bev_conv(pts_feats)
        ####################################

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

    def seg_gt_transform(self, segmentation):
        seg_gt_list = []
        for seg in segmentation:
            seg_gt_list.append(seg.unsqueeze(dim=0))
        segmentation = torch.cat(seg_gt_list, dim=0)
        one_hot = torch.zeros(segmentation.size(0), self.num_classes+1, segmentation.size(1), segmentation.size(2), dtype=torch.float32).to(segmentation.device)
        one_hot.scatter_(1, segmentation.unsqueeze(1), 1)

        return one_hot



    def depth_gt_transform(self, depth):
        depth_map_return = []
        for depth_batch in depth:
            valid_depth = depth_batch[:, 2] < self.d_bound[1]
            cam_depth = depth_batch[valid_depth, :]

            if self.image_size:
                H_ori, W_ori = self.image_size[0]
                H_resize, W_resize = self.image_size[1]
                resize_H = H_resize / H_ori
                resize_W = W_resize / W_ori

                cam_depth[:, 0] = cam_depth[:, 0] * resize_W
                cam_depth[:, 1] = cam_depth[:, 1] * resize_H

                depth_coords = cam_depth[:, :2].to(torch.long)
                depth_map = torch.zeros((H_resize, W_resize)).to(depth_coords.device)
                valid_mask = ((depth_coords[:, 1] < H_resize)
                              & (depth_coords[:, 0] < W_resize)
                              & (depth_coords[:, 1] >= 0)
                              & (depth_coords[:, 0] >= 0))
                depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]
                depth_map_return.append(depth_map.unsqueeze(0))
            else:
                return None
        gt_depths = torch.cat(depth_map_return, dim=0)
        B, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (gt_depths -
                     (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths > 0.),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
            -1, self.depth_channels + 1)[:, 1:]
        return gt_depths.float()

    def bev_conv(self, x):
        bev_out = x
        x = x[0]
        x = self.bev_backbone.conv1(x)
        x = self.bev_backbone.norm1(x)
        x = self.bev_backbone.relu(x)

        for i, layer_name in enumerate(self.bev_backbone.res_layers):
            res_layer = getattr(self.bev_backbone, layer_name)
            x = res_layer(x)  # layer1 layer2 layer3
            if i in self.bev_backbone.out_indices:
                bev_out.append(x)
        fpn_output = self.bev_neck(bev_out)  # fpn_output-tensor(1,256,128,128)
        return fpn_output

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      depth=None,
                      seg_label=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        if self.segmentation:
            seg_gt = self.seg_gt_transform(seg_label)
        if self.return_depth:
            depth_gt = self.depth_gt_transform(depth)
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        # bev_out = self.bev_conv(pts_feats)

        if self.return_depth:
            depth_preds = feature_dict['depth_preds']
        if self.segmentation:
            seg_preds = feature_dict['seg_preds']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
            if self.return_depth:
                losses_depth = self.depth_loss(depth_preds, depth_gt)
                losses.update(losses_depth)
            if self.segmentation:
                losses_seg = self.seg_loss(seg_preds, seg_gt)
                losses.update(losses_seg)

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

    def depth_loss(self, depth_preds, depth_gt):
        fg_mask = torch.max(depth_preds, dim=1).values > 0.0
        # depth_tmp = depth_preds.cpu().detach().numpy()
        # depth_gt_tmp = depth_gt.cpu().detach().numpy()
        loss_depth = (F.binary_cross_entropy(depth_preds[fg_mask], depth_gt[fg_mask], reduction='none', ).sum() / max(
            1.0, fg_mask.sum())) * self.loss_depth_weight
        return dict(loss_depth=[loss_depth])

    def seg_loss(self, seg_preds, seg_gt):
        B, H, W, num_classes = seg_preds.shape
        loss_seg = (F.binary_cross_entropy(seg_preds.reshape(-1, self.num_classes+1), seg_gt.reshape(-1, self.num_classes+1), reduction='none', ).sum() / max(
            1.0, B*H*W)) * self.loss_seg_weight
        return dict(loss_seg=[loss_seg])


if __name__ == "__main__":
    img_bev = torch.randn((2, 256, 124, 108))
    radar_bev = torch.randn((2, 256, 124, 108))
    cross_attention = Transformer_cross_attention(dim=256, num_heads=8, qkv_bias=True)
    out = cross_attention(img_bev, radar_bev)
    print(out.shape)