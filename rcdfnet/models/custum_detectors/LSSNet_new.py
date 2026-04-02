# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from ..module.EGA import CBAM
from mmcv.cnn import build_conv_layer
from rcdfnet.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from rcdfnet.ops.voxel_pooling_v2 import voxel_pooling


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class IFSO(nn.Module):
    def __init__(self, in_channels=256, mid_channels=128, context_channels=256, seg_channels=4,
                 camera_aware=False):
        super(IFSO, self).__init__()
        self.camera_aware = camera_aware

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        if self.camera_aware:
            self.bn = nn.BatchNorm1d(12)
            self.depth_mlp = Mlp(12, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_mlp = Mlp(12, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            # ASPP(mid_channels, mid_channels),
            # build_conv_layer(cfg=dict(
            #     type='DCN',
            #     in_channels=mid_channels,
            #     out_channels=mid_channels,
            #     kernel_size=3,
            #     padding=1,
            #     groups=4,
            #     im2col_step=128,
            # )),
            nn.Conv2d(mid_channels,
                      seg_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        x = self.reduce_conv(x)

        if self.camera_aware:
            calib = mats_dict
            batch_size = mats_dict.shape[0]
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            calib[:, ..., 0, 0],
                            calib[:, ..., 1, 1],
                            calib[:, ..., 0, 2],
                            calib[:, ..., 1, 2],
                        ],
                        dim=-1,
                    ),
                ],
                -1,
            )
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x)
            seg = self.depth_conv(x)

        return [context, seg]

class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, seg_channels=2, seg_sup=False,
                 camera_aware=False, use_aspp=False, use_dcn=False):
        super(DepthNet, self).__init__()
        self.camera_aware = camera_aware
        self.seg_sup = seg_sup


        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        if self.seg_sup:
            self.seg_conv_list = [BasicBlock(mid_channels, mid_channels)]
            self.seg_conv_list.append(nn.Conv2d(mid_channels, mid_channels//2, kernel_size=1))
            self.seg_conv_list.append(nn.Conv2d(mid_channels//2, seg_channels, kernel_size=1))
            self.seg_conv = nn.Sequential(*self.seg_conv_list)
        if self.camera_aware:
            self.bn = nn.BatchNorm1d(12)
            self.depth_mlp = Mlp(12, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_mlp = Mlp(12, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            self.depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            self.depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )
                )
            )
        self.depth_conv_list.append(nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0))
        self.depth_conv = nn.Sequential(*self.depth_conv_list)


    def forward(self, x, mats_dict):
        x = self.reduce_conv(x)
        if self.seg_sup:
            seg_output = self.seg_conv(x)

        if self.camera_aware:
            intrins = mats_dict
            batch_size = intrins.shape[0]
            mlp_input = intrins.view(batch_size, -1).contiguous()
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x)
            depth = self.depth_conv(x)

        if self.seg_sup:
            return torch.cat([depth, context], dim=1), seg_output
        else:
            return torch.cat([depth, context], dim=1)


class LSSNet(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, depth_net_conf, depth_channels, return_depth, return_seg=False):

        super(LSSNet, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = 80
        self.depth_channels = depth_channels
        self.camera_aware = False
        self.radar_view_transform = True
        self.return_seg = return_seg
        self.semantic_threshold = 0.5

        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())

        self.depth_channels, _, _, _ = self.frustum.shape
        self.depth_net = self._configure_depth_net(depth_net_conf)
        self.return_depth = return_depth
        # self.cbam = CBAM(self.output_channels)

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            seg_sup=self.return_seg,
            camera_aware=self.camera_aware
        )

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)
        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, calib, z_min=-4, z_max=2):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """

        # undo post-transformation
        # B x D x H x W x 4
        calib = calib.squeeze(dim=1)
        batch_size, _, _ = calib.shape
        depth, height, width, coor = self.frustum.shape
        points = self.frustum.expand(batch_size, depth, height, width, coor)

        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :2] * points[:, :, :, :, 2:3],
             points[:, :, :, :, 2:]), 4)
        padding_values = torch.tensor([0, 0, 0, 1], dtype=calib.dtype, device=calib.device)
        # sensor2ego_mat = torch.cat((calib, padding_values.view(1, 1, 4).expand(calib.size(0), 1, 4)), dim=1)
        sensor2ego_mat = calib
        combine = torch.inverse(sensor2ego_mat.to("cpu")).to(calib.device).double()
        points = combine.view(batch_size, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1).double()).half()

        points = points.squeeze(-1)
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))
        return points[..., :3], points_valid_z

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_voxel_net(self, img_feat_with_depth):
        return img_feat_with_depth

    def forward(self, features, calib, pts_occupancy=None):
        features_depth = features[3]
        assert features_depth.shape[0] == calib.shape[0]
        batch_size = calib.shape[0]
        img_feats = features_depth
        if self.return_seg:
            depth_feature, seg_feature = self._forward_depth_net(img_feats, calib)
            seg_feature = seg_feature.softmax(1)
            img_mask = seg_feature[:,1:2] >= self.semantic_threshold
        else:
            depth_feature = self._forward_depth_net(img_feats, calib)
        depth = depth_feature[:, :self.depth_channels].softmax(1)

        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, self.depth_channels:(
                self.depth_channels + self.output_channels)].unsqueeze(2)
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        geom_xyz, geom_xyz_valid = self.get_geometry(calib)

        if pts_occupancy is not None:
            radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous()
            image_feature_collapsed = (features_depth * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2)
            img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)

            img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)
            img_context = self._forward_view_aggregation_net(img_context)

        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 3, 4, 1)
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous(),
                                    self.voxel_num.cuda())
        # feature_map = self.cbam(feature_map)
        out = [feature_map]
        if self.return_depth:
            out.append(depth)
        if self.return_seg:
            out.append(seg_feature)
        return out



