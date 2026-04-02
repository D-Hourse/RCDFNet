import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
from torch.cuda.amp.autocast_mode import autocast
import numpy as np
from PIL import Image

from mmdet.models import DETECTORS
from rcdfnet.models.custum_detectors import MVXFasterRCNN
from .DualNet import DualNet

from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
from .. import builder

import torch.nn as nn
from .BEVCross_modal_attention import Cross_Modal_Fusion, Cross_Modal_Fusion_test
from .other_modules import PointFusion
from .BEV_deformable_fuser import BEVFuser
from ..module.MCMA import MCAM
from .DualNet import BEVGeomAttention, ProbNet, LearnedPositionalEncoding, DualFeatFusion
from plot.heatmap import draw_feature_fuser_map, draw_feature_pts_map, draw_feature_img_LSS_map, draw_feature_img_OFT_map, save_image_tensor2cv2, draw_feature_img_fuser_map


####----------------------------debug image------------------------------------
def denormalize(img_tensor):
    # 定义用于标准化的均值和标准差
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    # 将 PyTorch tensor 转换为 NumPy 数组
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # 反标准化
    img_np = img_np * std + mean

    # 限制像素值在 [0, 255] 范围内
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np


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
class Best_RC_Second_pointfusion80_HTbev_bevconv(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self,image_size=None, img_bev_harf=False, rc_fusion="concat", camera_stream=True,
                 grid_size=(69.12,79.36), grid_offset=(0,-39.68, 0), grid_res=0.32, grid_z_min=-3, num_class=3, voxel_size=None,
                 grid_z_max=2.76, se=False, imc=256,radc=384, return_depth=True, return_mask=True,loss_depth_weight=None,
                 **kwargs):
        point_fusion_config = kwargs.get('point_fusion', None)
        kwargs.pop('point_fusion')
        pts_output_config = kwargs.get('pts_output_channels', None)
        kwargs.pop('pts_output_channels')
        DualTransform_config = kwargs.get('DualTransform', None)
        kwargs.pop('DualTransform')
        freeze_radar = kwargs.get('freeze_radar', None)
        kwargs.pop('freeze_radar')
        img_bev_encoder_backbone = kwargs.get('img_bev_encoder_backbone', None)
        kwargs.pop('img_bev_encoder_backbone')
        img_bev_encoder_neck = kwargs.get('img_bev_encoder_neck', None)
        kwargs.pop('img_bev_encoder_neck')
        super(Best_RC_Second_pointfusion80_HTbev_bevconv, self).__init__(**kwargs)
        self.rc_fusion = rc_fusion
        self.lift = camera_stream
        self.se = se
        self.image_size = image_size
        self.loss_depth_weight = loss_depth_weight
        self.num_classes = num_class
        self.return_depth = return_depth
        self.return_mask = return_mask
        self.pillar_size = voxel_size
        self.depth_channels = DualTransform_config['depth_channels']
        self.downsample = DualTransform_config['downsample_factor']
        self.d_bound = DualTransform_config['d_bound']
        self.bev_h = int(grid_size[0]//grid_res)
        self.bev_w = int(grid_size[1]//grid_res)
        self.output_channels = pts_output_config['out_channels']

        ####-----------pts output---------
        if pts_output_config is not None:
            self.pts_pred_context = nn.Sequential(
                nn.Conv2d(int(pts_output_config['in_channels']),
                          int(pts_output_config['in_channels']//2),
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect'),
                nn.BatchNorm2d(int(pts_output_config['in_channels']//2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(pts_output_config['in_channels'] // 2),
                          int(pts_output_config['out_channels']),
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          ),
            )

        ####-----------BEV conv-----------
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = None
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        else:
            self.img_bev_encoder_neck = None
        ####-----------LSS分支初始化--------
        if camera_stream:
            self.BEVencoder = DualNet(**DualTransform_config)

        ####-----------fuser---------------
        if point_fusion_config is not None:
            self.point_Fusion = PointFusion(**point_fusion_config)
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
            self.cross_attention = Cross_Modal_Fusion(kernel_size=3, radc=radc, imc=imc)

        ####----------BEV mask---------------
        if self.return_mask:
            self.geom_att = BEVGeomAttention()
            self.prob = ProbNet(in_channels=self.output_channels, with_centerness=True,
                                bev_size=(self.bev_h, self.bev_w))
            self.positional_encoding = LearnedPositionalEncoding(self.output_channels // 2, self.bev_h, self.bev_w)

        self.freeze_img = kwargs.get('freeze_img', False)
        self.freeze_radar = freeze_radar
        self.init_weights()
        self.freeze()

    ####-----------------------数据预处理-----------------------------
    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        # assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs
        mlp_input = self.get_mlp_input(sensor2egos, ego2globals, intrins,
                                                            post_rots, post_trans, bda)

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double().to('cpu')).to(keyego2global.device)
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda, mlp_input]

    ####-----------------------冻结部分分支---------------------------

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            # if self.BEVencoder is not None:
            #     for param in self.BEVencoder.depth_net.parameters():
            #         param.requires_grad = False
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
            # if self.with_pts_bbox:
            #     for param in self.pts_bbox_head.parameters():
            #         param.requires_grad = False

    ####--------------------mlp input----------------------------------
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

    ####--------------------BEV conv----------------------------------
    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    ####-------------------PointFusion---------------------------------
    def isfusion(self, pts, pillars, pillars_num_points, pillar_coors, pts_feats, img_feats, img_metas, batch_size):
        kwargs = {}
        # create BEV space
        pts_metas = {}
        pts_metas['pillars'] = pillars
        pts_metas['pillars_num_points'] = pillars_num_points
        pts_metas['pillar_coors'] = pillar_coors
        pts_metas['pts'] = pts
        pts_metas['pillar_size'] = self.pillar_size

        post_rots = img_metas['post_rots']
        post_trans = img_metas['post_trans'].unsqueeze(-1)
        img_aug_matrix = torch.cat((post_rots, post_trans), dim=-1).squeeze(dim=1)
        lidar2img = img_metas['lidar2img'].squeeze(dim=1)

        kwargs.update(dict(pts_metas=pts_metas))
        kwargs.update(dict(img_metas=img_metas))
        kwargs.update(dict(img_aug_matrix=img_aug_matrix))
        kwargs.update(dict(lidar2img=lidar2img))
        kwargs.update(dict(img_shape=self.image_size[1]))

        img_feats_use = (img_feats)
        x = self.point_Fusion(img_feats_use, pts_feats, batch_size, **kwargs)
        return x

    ####-----------------------------pts feat------------------------
    def extract_pts_feat(self, pts, img_feats, img_metas):
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
        if self.pts_pred_context:
            x = self.pts_pred_context(x[0])
        return [x], voxels, num_points, coors

    ####----------------------------fuser feat--------------------------------
    def extract_feat(self, points, img_inputs, img_metas):
        """Extract features from images and points."""
        # output_path = '/plot/Best_RCFusion_fasterRCNN/output_image.png'
        imgs, sensor2keyegos, ego2globals, intrins, \
            post_rots, post_trans, bda, _ = self.prepare_inputs(img_inputs)
        ####------------------------img metas generate-----------------------
        img_metas_temp = dict()
        batch_size, _, _, _, _ = imgs.shape
        calib = []
        for sample_idx in range(batch_size):
            mat = img_metas[sample_idx]['lidar2img']
            mat = torch.Tensor(mat[:3, :]).to(imgs[0].device)
            calib.append(mat)
        if batch_size > 1:
            calib = torch.stack(calib)
        else:
            calib = torch.stack(calib).unsqueeze(dim=0)
        img_metas_temp.update(lidar2img=calib)
        img_metas_temp.update(sensor2keyegos=sensor2keyegos)
        img_metas_temp.update(intrins=intrins)
        img_metas_temp.update(post_rots=post_rots)
        img_metas_temp.update(post_trans=post_trans)
        img_metas_temp.update(bda=bda)

        img_feats = self.extract_img_feat(imgs, img_metas)  ##图像特征
        pts_feats, voxels, num_points, coors = self.extract_pts_feat(points, img_feats, img_metas_temp)


        # -------------提升的过程写在此处--------------------
        if self.lift:
            ###判断batch一致
            batch_size, _, _, _ = img_feats[0].shape
            img_bev_feat, depth, img_context = self.BEVencoder(img_feats, img_metas_temp)
            depth = depth.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)

            # -------------------------融合部分----------------
            if pts_feats is None:
                pts_feats = [img_bev_feat]  ####cam stream only
            else:
                # pointfusion
                if self.point_Fusion:
                    pts_feats = [self.isfusion(points, voxels, num_points, coors, pts_feats, img_context, img_metas_temp, batch_size)]

                # cross attention
                if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                    img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear',
                                                 align_corners=True)
                fuser_feats = self.cross_attention(img_bev_feat, pts_feats[0])

            if self.return_mask:
                mask = torch.zeros((batch_size, self.bev_h, self.bev_w), device=fuser_feats.device).to(fuser_feats.dtype)
                bev_pos = self.positional_encoding(mask).to(fuser_feats.dtype)
                bev_mask_logit = self.prob(bev_pos + fuser_feats)
                fuser_feats = self.geom_att(fuser_feats, bev_mask_logit) * fuser_feats

            fuser_feats = [self.bev_encoder(fuser_feats)]

        return_dict = dict(img_feats=img_feats, pts_feats=fuser_feats, depth_preds=depth)
        if self.return_mask:
            return_dict.update(bev_mask=bev_mask_logit)

        return return_dict

    def simple_test(self, points, img_metas, img_inputs=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas)

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

    def seg_gt_transform(self, segmentation, label_H, label_W):
        seg_gt_list = []
        for seg in segmentation:
            seg[seg > 0] = 1
            H, W = seg.shape
            seg = seg.view(label_H, H//label_H, label_W, W//label_W).contiguous()
            seg = seg.view(-1, H//label_H * W//label_W)
            seg = torch.max(seg, dim=-1).values
            seg = seg.view(label_H, label_W).contiguous()
            seg_gt_list.append(seg.unsqueeze(dim=0))
        segmentation = torch.cat(seg_gt_list, dim=0)
        one_hot = torch.zeros(segmentation.size(0), self.num_classes+1, segmentation.size(1), segmentation.size(2), dtype=torch.float32).to(segmentation.device)
        one_hot.scatter_(1, segmentation.unsqueeze(1), 1)

        return one_hot

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (gt_depths - (self.d_bound[0] - self.d_bound[2])) / self.d_bound[2]

        gt_depths = torch.where((gt_depths < self.depth_channels) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels).view(-1, self.depth_channels)
        return gt_depths.float()



    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      gt_depth=None,
                      seg_label=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_bev_mask=None):

        feature_dict = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        if self.return_depth:
            depth_preds = feature_dict['depth_preds']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
            if self.return_depth:
                losses_depth = self.get_depth_loss(gt_depth, depth_preds)
                losses.update(loss_depth=[losses_depth])
            if self.return_mask:
                losses_mask_ce, losses_dice = self.prob.get_bev_mask_loss(gt_bev_mask, feature_dict['bev_mask'])
                losses.update(losses_mask_ce=[losses_mask_ce])
                losses.update(losses_dice=[losses_dice])

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

    def forward_test(self, points, img_metas, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        img = img_inputs
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    def depth_loss(self, depth_preds, depth_gt):
        fg_mask = torch.max(depth_preds, dim=1).values > 0.0
        # depth_tmp = depth_preds.cpu().detach().numpy()
        # depth_gt_tmp = depth_gt.cpu().detach().numpy()
        loss_depth = (F.binary_cross_entropy(depth_preds[fg_mask], depth_gt[fg_mask], reduction='none', ).sum() / max(
            1.0, fg_mask.sum())) * self.loss_depth_weight
        return dict(loss_depth=[loss_depth])

    def seg_loss(self, seg_preds, seg_label):
        loss_seg = 0
        B, num_classes, H, W = seg_preds.shape
        seg_gt = self.seg_gt_transform(seg_label, H, W)
        seg_gt = seg_gt.permute(0, 2, 3, 1).contiguous().reshape(-1, self.num_classes+1)
        seg_pred = seg_preds.permute(0, 2, 3, 1).contiguous()
        loss_seg_temp = (F.binary_cross_entropy(seg_pred.reshape(-1, num_classes), seg_gt, reduction='none', ).sum() / max(
            1.0, B*H*W)) * self.loss_seg_weight
        loss_seg = loss_seg + loss_seg_temp
        return dict(loss_seg=[loss_seg])


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保输入是概率
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # 计算交叉熵
        log_probs = torch.log(inputs + 1e-8)
        ce_loss = -targets * log_probs

        # 计算焦点损失
        probs = torch.gather(inputs, 1, targets.argmax(dim=1, keepdim=True))
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.sum() / inputs.size(0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    img_bev = torch.randn((2, 256, 124, 108))
    radar_bev = torch.randn((2, 256, 124, 108))
    cross_attention = Transformer_cross_attention(dim=256, num_heads=8, qkv_bias=True)
    out = cross_attention(img_bev, radar_bev)
    print(out.shape)