# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from .LSSNet import LSSNet
from .other_modules import GSAU
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from rcdfnet.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.utils import LearnedPositionalEncoding
from mmdet.models.backbones.resnet import BasicBlock
from rcdfnet.ops.voxel_pooling_v2 import voxel_pooling
from rcdfnet.ops.bev_pool_v2.bev_pool import bev_pool_v2

class MS_CAM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"
    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        return self.sigmoid(xlg)

class BEVGeomAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVGeomAttention, self).__init__()
        # self.GSAU = GSAU(mid_channels)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1+bev_prob)

class BEVGeomAttention_v2(nn.Module):

    def __init__(self, input_channels, kernel_size=7):
        super(BEVGeomAttention_v2, self).__init__()
        self.GSAU = GSAU(input_channels)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bev_prob):
        x = self.GSAU(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1+bev_prob)

class BEVSpatialAtt(nn.Module):

    def __init__(self, kernel_size=7):
        super(BEVSpatialAtt, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)

class ChannelAttention(nn.Module):

    def __init__(self, input_channel, output_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(input_channel, input_channel // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(input_channel // ratio, output_channel, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResCBAMBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes,planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ProbNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(ProbNet, self).__init__()
        self.loss_weight=loss_weight
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.prob_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),
        )
        self.mask_net = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0], bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        height_feat = self.base_conv(input)
        height_feat = self.prob_conv(height_feat)
        bev_prob = self.mask_net(height_feat)
        return bev_prob

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h, 1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return self.loss_weight*mask_ce_loss, mask_dice_loss
        # return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)



class BEVMaskNet(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(BEVMaskNet, self).__init__()
        self.loss_weight=loss_weight
        self.bev_sa = BEVSpatialAtt()
        mid_channels=in_channels//2
        self.base_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.CBAM_conv = nn.Sequential(
            ResCBAMBlock(mid_channels, mid_channels),
        )

        self.mask_net1 = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0], bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        input = self.base_conv(input)
        input = self.CBAM_conv(input)
        pred_mask = self.mask_net1(input)
        return pred_mask

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h, 1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return self.loss_weight*mask_ce_loss, mask_dice_loss
        # return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)

class BEVSupModule(BaseModule):

    def __init__(
        self,
        in_channels=512,
        scale_factor=1,
        with_centerness=False,
        loss_weight=6.0,
        bev_size=None,
    ):
        super(BEVSupModule, self).__init__()
        self.loss_weight=loss_weight
        self.bev_sa = BEVSpatialAtt()
        mid_channels=in_channels

        self.mask_net1 = nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, stride=1)

        self.with_centerness=with_centerness
        if with_centerness:
            self.centerness = bev_centerness_weight(bev_size[0], bev_size[1]).cuda()
        self.dice_loss = DiceLoss(use_sigmoid=True, loss_weight=self.loss_weight)
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))

    def forward(self, input):
        pred_mask = self.mask_net1(input)
        return pred_mask

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, bev_h, bev_w = gt_bev_mask.shape
        b = gt_bev_mask.reshape(bs , bev_w * bev_h).permute(1, 0).to(torch.float)
        a = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        if self.with_centerness:
            self.ce_loss.reduction='none'
            tmp_loss = self.ce_loss(a, b)
            mask_ce_loss=(tmp_loss*self.centerness.reshape(bev_w * bev_h, 1)).mean()
        else:
            mask_ce_loss = self.ce_loss(a, b)
        mask_dice_loss = self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
        return self.loss_weight*mask_ce_loss, mask_dice_loss
        # return dict(mask_ce_loss=self.loss_weight*mask_ce_loss, mask_dice_loss=mask_dice_loss)


class DualFeatFusion(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(DualFeatFusion, self).__init__()
        self.ca = MS_CAM(input_channel, output_channel)

    def forward(self, x1, x2):
        channel_factor = self.ca(torch.cat((x1, x2), 1))
        out = channel_factor * x1 + (1 - channel_factor) * x2

        return out


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)
        self.flatten = nn.Flatten()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(-1, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        return x


class BEVfusion_mask_att(nn.Module):
    def __init__(self,
                 in_channels=256,
                 with_centerness=True,
                 bev_size=(160, 160)
                 ):
        super(BEVfusion_mask_att, self).__init__()
        self.geom_att = BEVGeomAttention_v2(input_channels=in_channels)
        self.prob = ProbNet(in_channels=in_channels, with_centerness=with_centerness, bev_size=bev_size)
        self.positional_encoding = LearnedPositionalEncoding(in_channels // 2, bev_size[0], bev_size[1])

    def forward(self, mask, feats):
        bev_pos = self.positional_encoding(mask).to(feats.dtype)
        bev_mask_logit = self.prob(bev_pos + feats)
        fuser_feats = self.geom_att(feats, bev_mask_logit) * feats
        return fuser_feats, bev_mask_logit

class BEVfusion_mask_att_v2(nn.Module):
    def __init__(self,
                 in_channels=256,
                 with_centerness=True,
                 bev_size=(160, 160),
                 name=None
                 ):
        super(BEVfusion_mask_att_v2, self).__init__()
        self.geom_att = BEVGeomAttention_v2(input_channels=in_channels)
        self.bev_mask_att = BEVMaskNet(in_channels=in_channels, with_centerness=with_centerness, bev_size=bev_size)
        self.positional_encoding = LearnedPositionalEncoding(in_channels // 2, bev_size[0], bev_size[1])

    def forward(self, mask, feats):
        bev_pos = self.positional_encoding(mask).to(feats.dtype)
        bev_mask_logit = self.bev_mask_att(bev_pos + feats)
        fuser_feats = self.geom_att(feats, bev_mask_logit) * feats
        return fuser_feats, bev_mask_logit

class BEVfusion_mask_att_v2_without_geomatt(nn.Module):
    def __init__(self,
                 in_channels=256,
                 with_centerness=True,
                 bev_size=(160, 160),
                 name=None
                 ):
        super(BEVfusion_mask_att_v2_without_geomatt, self).__init__()
        # self.geom_att = BEVGeomAttention_v2(input_channels=in_channels)
        self.bev_mask_att = BEVMaskNet(in_channels=in_channels, with_centerness=with_centerness, bev_size=bev_size)
        self.positional_encoding = LearnedPositionalEncoding(in_channels // 2, bev_size[0], bev_size[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask, feats):
        bev_pos = self.positional_encoding(mask).to(feats.dtype)
        bev_mask_logit = self.bev_mask_att(bev_pos + feats)
        fuser_feats = self.sigmoid(bev_mask_logit) * feats
        return fuser_feats, bev_mask_logit


class BEVfusion_mask_att_v2_without_AttConv(nn.Module):
    def __init__(self,
                 in_channels=256,
                 with_centerness=True,
                 bev_size=(160, 160),
                 name=None
                 ):
        super(BEVfusion_mask_att_v2_without_AttConv, self).__init__()
        self.geom_att = BEVGeomAttention_v2(input_channels=in_channels)
        self.bev_mask_att = BEVSupModule(in_channels=in_channels, with_centerness=with_centerness, bev_size=bev_size)
        self.positional_encoding = LearnedPositionalEncoding(in_channels // 2, bev_size[0], bev_size[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask, feats):
        bev_pos = self.positional_encoding(mask).to(feats.dtype)
        bev_mask_logit = self.bev_mask_att(bev_pos + feats)
        fuser_feats = self.geom_att(feats, bev_mask_logit) * feats
        return fuser_feats, bev_mask_logit

class BEVfusion_mask_att_v2_without_ALL(nn.Module):
    def __init__(self,
                 in_channels=256,
                 with_centerness=True,
                 bev_size=(160, 160),
                 name=None
                 ):
        super(BEVfusion_mask_att_v2_without_ALL, self).__init__()
        # self.geom_att = BEVGeomAttention_v2(input_channels=in_channels)
        self.bev_mask_att = BEVSupModule(in_channels=in_channels, with_centerness=with_centerness, bev_size=bev_size)
        self.positional_encoding = LearnedPositionalEncoding(in_channels // 2, bev_size[0], bev_size[1])
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask, feats):
        bev_pos = self.positional_encoding(mask).to(feats.dtype)
        bev_mask_logit = self.bev_mask_att(bev_pos + feats)
        fuser_feats = self.sigmoid(bev_mask_logit) * feats
        return fuser_feats, bev_mask_logit


class ViewTransformNet(LSSNet):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim, collapse_z,
                 downsample_factor, depth_net_conf, depth_channels, return_depth,
                 output_channels, return_mask, num_height, pc_range, gird_config,
                 dim=3, mean=False, return_context=False):

        super(ViewTransformNet, self).__init__(x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, depth_net_conf, depth_channels, return_depth, output_channels,dim)
        self.return_mask = return_mask
        self.return_context = return_context
        self.num_height = num_height
        self.input_shape = final_dim
        self.pc_range = pc_range
        self.collapse_z = collapse_z
        self.grid_config = gird_config
        self.bev_h = int((self.grid_config['y'][1] - self.grid_config['y'][0]) / self.grid_config['y'][2])
        self.bev_w = int((self.grid_config['x'][1] - self.grid_config['x'][0]) / self.grid_config['x'][2])

        if True:
            self.fuser = DualFeatFusion(2 * self.output_channels, self.output_channels)
            self.geom_att = BEVGeomAttention()
            self.prob = ProbNet(in_channels=self.output_channels, with_centerness=True, bev_size=(self.bev_h, self.bev_w))
            self.positional_encoding = LearnedPositionalEncoding(self.output_channels // 2, self.bev_h, self.bev_w)

        self.register_buffer('voxel_hp', self.get_reference_points_3d(self.bev_h, self.bev_w, num_points_in_pillar=self.num_height, is_mean=mean))

    def get_reference_points_3d(self, H, W, Z=8, num_points_in_pillar=13, device='cuda', dtype=torch.float, is_mean=False):
        """Get the reference points used in HT.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in HT, has \
                shape (bs, D, HW, 3).
        """
        if is_mean is False:
            zs_l = torch.linspace(3, Z-1, 5, dtype=dtype,device=device)
            zs_g = torch.linspace(0.5, Z - 0.5, num_points_in_pillar-5, dtype=dtype,device=device)
            zs = torch.cat((zs_l,zs_g)).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        else:
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        return ref_3d

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def get_sampling_point(self, reference_points, pc_range, depth_range, lidar2img, img_aug, image_shapes):
        # B, bev_z, bev_h* bev_w, 3
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # B, D, HW, 3
        B, Z, num_query = reference_points.size()[:3]
        reference_points = reference_points.view(B, -1, 3)
        num_cam = lidar2img.size(1)
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.view(B, 1, Z * num_query, 4).repeat(1, num_cam, 1, 1)

        lidar2img = lidar2img[..., :3, :].unsqueeze(1).view(B, num_cam, 1, 3, 4)
        img_aug = img_aug.unsqueeze(1).view(B, num_cam, 1, 3, 4)

        reference_points = lidar2img.matmul(reference_points.unsqueeze(-1)).squeeze(-1)

        eps = 1e-5
        referenece_depth = reference_points[..., 2:3].clone()
        bev_mask = (reference_points[..., 2:3] > eps)

        reference_points_cam = torch.cat((reference_points[..., 0:2] / torch.maximum(
            reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3]) * eps), reference_points[..., 2:3],
                                          torch.ones_like(reference_points[..., 2:3])), -1)

        reference_points_cam = torch.matmul(img_aug,
                                            reference_points_cam.unsqueeze(-1)).squeeze(-1)

        reference_points_cam = reference_points_cam[..., 0:2]
        reference_points_cam[..., 0] /= image_shapes[1]
        reference_points_cam[..., 1] /= image_shapes[0]

        reference_points_cam = reference_points_cam.view(B, num_cam, Z, num_query, 2)
        referenece_depth = referenece_depth.view(B, num_cam, Z, num_query, 1)
        bev_mask = bev_mask.view(B, num_cam, Z, num_query, 1)

        bev_mask = (bev_mask & (reference_points_cam[..., 0:1] > 0.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0))
        # D, B, N, num_query, 1
        if depth_range is not None:
            referenece_depth = (referenece_depth - depth_range[0]) / (depth_range[1] - depth_range[0])
            bev_mask = (bev_mask & (referenece_depth > 0.0)
                        & (referenece_depth < 1.0))
        # [B, N, Z*Nq, 1] bev_mask
        # [B, N, Z*Nq, 2] reference_points_cam
        # [B, N, Z*Nq, 1] referenece_depth
        bev_mask = torch.nan_to_num(bev_mask)
        return torch.cat((reference_points_cam, referenece_depth), -1), bev_mask


    def forward(self, features, calib):
        features_depth = features[self.dim]
        img_feats = features_depth
        depth_feature = self._forward_depth_net(img_feats, calib)
        B, C, H, W = depth_feature.shape
        N = 1                               # camera nums
        depth = depth_feature[:, :self.depth_channels].softmax(1)
        context = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)]

        #------------------------------transform bev-----------------------
        # 对depth和context增加维度适应ViewTransorm
        depth = depth.unsqueeze(dim=1)
        context = context.unsqueeze(dim=1)
        # lidar2img imgaug
        lidar2img = calib['lidar2img']
        img_aug = torch.cat((calib['post_rots'], calib['post_trans'].unsqueeze(-1)), -1)
        # 得到bev点投影到图像的坐标
        voxel_xyz = self.voxel_hp.clone()
        voxel_xyz = voxel_xyz[None].repeat(B, 1, 1, 1)
        coor, mask = self.get_sampling_point(voxel_xyz, self.pc_range, self.grid_config['depth'], lidar2img,
                                             img_aug, self.input_shape)
        # 得到反向投影的bev特征
        channel_feat = self.fast_sampling(
            coor, mask, depth.view(B, N, self.depth_channels, H, W),
            context.view(B, N, self.output_channels, H, W))

        depth = depth.squeeze(dim=1)
        context = context.squeeze(dim=1)

        if self.return_mask:
            mask = torch.zeros((B, self.bev_h, self.bev_w), device=channel_feat.device).to(channel_feat.dtype)
            bev_pos = self.positional_encoding(mask).to(channel_feat.dtype)
            bev_mask_logit = self.prob(bev_pos + channel_feat)
            channel_feat = self.geom_att(channel_feat, bev_mask_logit) * channel_feat
            return channel_feat, depth, bev_mask_logit

        if self.return_context:
            return channel_feat, depth, context
        else:
            return channel_feat, depth

    def init_acceleration_ht(self, coor, mask):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)

        self.ranks_bev_ht = ranks_bev.int().contiguous()
        self.ranks_feat_ht = ranks_feat.int().contiguous()
        self.ranks_depth_ht = ranks_depth.int().contiguous()
        self.interval_starts_ht = interval_starts.int().contiguous()
        self.interval_lengths_ht = interval_lengths.int().contiguous()

    def fast_sampling(self, coor, mask, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.fast_sample_prepare(coor, mask)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                1, int(self.bev_h), int(self.bev_w)
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def fast_sample_prepare(self, coor, mask):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the image space in
                shape (B, N, ZNq 3).
            mask (torch.tensor): mask of points in the imaage space in
                shape (B, N, ZNq, 1).
        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, Z, Nq, _ = coor.shape
        num_points = B * N * Z * Nq
        # record the index of selected points for acceleration purpose
        ranks_bev = torch.range(
            0, num_points // (N*Z) - 1, dtype=torch.int, device=coor.device)
        ranks_bev = ranks_bev.reshape(B, 1, 1, Nq)
        ranks_bev = ranks_bev.expand(B, N, Z, Nq).flatten()
        # convert coordinate into the image feat space
        self.W = self.input_shape[1] // self.downsample_factor
        self.H = self.input_shape[0] // self.downsample_factor
        self.D = self.depth_channels
        coor[..., 0] *= self.W
        coor[..., 1] *= self.H
        coor[..., 2] *= self.D
        # [B, N, Z, Nq, 3]
        coor = coor.round().long().view(num_points, 3)
        coor[..., 0].clamp_(min=0, max=self.W-1)
        coor[..., 1].clamp_(min=0, max=self.H-1)
        coor[..., 2].clamp_(min=0, max=self.D-1)
        batch_idx = torch.range(0, B*N-1).reshape(B*N, 1). \
            expand(B*N, num_points // (B*N)).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = mask.reshape(-1)
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_bev = \
            coor[kept], ranks_bev[kept]

        ranks_depth = coor[:, 3] * (self.D * self.W * self.H)
        ranks_depth += coor[:, 2] * (self.W * self.H)
        ranks_depth += coor[:, 1] * self.W + coor[:, 0]
        depth_size = B * N * self.D * self.W * self.H
        ranks_depth.clamp_(min=0, max=depth_size-1)

        ranks_feat = coor[:, 3] * (self.W * self.H)
        ranks_feat += coor[:, 1] * self.W + coor[:, 0]
        feat_size = B * N * self.W * self.H

        ranks_feat.clamp_(min=0, max=feat_size-1)

        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def get_ht_bev_feat(self, input, depth, tran_feat, bev_mask=None):
        B, N, C, H, W = input[0].shape
        self.H = H
        self.W = W

        # Prob-Sampling
        if self.accelerate:
            feat = tran_feat.view(B, N, self.output_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.depth_channels, H, W)
            bev_feat_shape = (depth.shape[0], 1,
                          int(self.bev_h), int(self.bev_w),
                          feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth_ht,
                                   self.ranks_feat_ht, self.ranks_bev_ht,
                                   bev_feat_shape, self.interval_starts_ht,
                                   self.interval_lengths_ht)
            if bev_mask is not None:
                bev_feat = bev_feat * bev_mask
            bev_feat = bev_feat.squeeze(2)
        else:
            lidar2img, img_aug = self.get_projection(*input[1:7])
            voxel = self.get_reference_points_3d(self.bev_h, self.bev_w, bs=B, num_points_in_pillar=self.num_height)
            coor, mask = self.get_sampling_point(voxel, self.pc_range, self.grid_config['depth'], lidar2img, img_aug, self.input_size)

            if bev_mask is not None:
                mask = bev_mask * mask.view(B, N, self.num_height, self.bev_h, self.bev_w)
            bev_feat = self.fast_sampling(
                coor, mask, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.output_channels, H, W))
        return bev_feat



def bev_centerness_weight(nx, ny):
    xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, ny))
    grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
    grid[:, :, 0] = grid[:, :, 0] - nx // 2
    a = grid[:, :, 0] / (nx//2)
    b = grid[:, :, 1] / ny
    centerness = (a**2 + b**2) / 2
    centerness = centerness.sqrt() + 1
    return centerness

# def bev_centerness_weight(nx, ny):
#     xs, ys = torch.meshgrid(torch.arange(0, nx), torch.arange(0, nx))
#     grid = torch.cat([xs[:, :, None], ys[:, :, None]], -1)
#     grid = grid - nx//2
#     grid = grid / (nx//2)
#     centerness = (grid[..., 0]**2 + grid[..., 1]**2) / 2
#     centerness = centerness.sqrt() + 1
#     return centerness


class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, loss_weight=1.):
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, smooth=1e-5):
        if self.use_sigmoid:
            inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return self.loss_weight * (1 - dice)