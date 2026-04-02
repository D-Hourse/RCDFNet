import mmcv
import numpy as np
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
import torch.nn as nn
import random

from rcdfnet.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from rcdfnet.ops import Voxelization
from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from .BEV_deformable_fuser import BEVMulti_Fuser
from ..sparse_det.bev_sparse_recover import BEVSparse1
from ..custum_detectors.DualNet import MS_CAM
from rcdfnet.ops.spconv import *
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32, auto_fp16
from ...models import builder

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return x * scale



class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class PointFusion(BaseModule):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    def __init__(self, num_points_in_pillar=10, embed_dims_img=256, embed_dims_radar=384, bev_size=256, scale_factor=None, region_shape=None, grid_size=None, region_drop_info=None):
        super(PointFusion, self).__init__()

        self.num_points_in_pillar = num_points_in_pillar
        self.bev_size = bev_size
        self.scale_factor = scale_factor
        self.region_shape = region_shape
        self.grid_size = grid_size
        self.region_drop_info = region_drop_info


        self.random_noise = 1.0

        self.embed_dims_img = embed_dims_img
        self.embed_dims_radar = embed_dims_radar
        self.conv_fusion = ConvModule(
            self.embed_dims_img + self.embed_dims_radar,
            self.embed_dims_radar,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        def create_2D_grid(x_size, y_size):
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
            # NOTE: modified
            batch_x, batch_y = torch.meshgrid(
                *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
            )
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
            return coord_base

        self.bev_pos = create_2D_grid(self.bev_size, self.bev_size)

    def img_point_sampling(self, reference_voxel, mlvl_feats, batch_size=4, **kwargs):  # from UVTR

        img_aug_matrix = kwargs.get('img_aug_matrix', None)
        lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        lidar2image = kwargs.get('lidar2img', None)
        image_size = kwargs['img_shape']

        # Transfer to Point cloud range with X,Y,Z
        mask = []
        reference_voxel_cam = []

        for b in range(batch_size):
            cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            if img_aug_matrix is not None:
                cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            if lidar_aug_matrix is not None:
                cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, np.ndarray) else lidar2image[b]

            # inverse aug for pseudo points
            if lidar_aug_matrix is not None:
                cur_coords -= lidar_aug_matrix[b, :3, 3]
                cur_coords = torch.inverse(lidar_aug_matrix[b, :3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )

            # lidar2image
            if isinstance(lidar2image, np.ndarray):
                cur_lidar2image = torch.from_numpy(cur_lidar2image).to(cur_coords.device)
            cur_coords = cur_lidar2image[:3, :3].mm(cur_coords.T)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[2, :].clone()

            if self.grid_size:
                this_mask = (dist > 1e-5)&(dist < self.grid_size[3])
            else:
                this_mask = (dist > 1e-5)

            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            # cur_coords_vis = cur_coords.cpu().detach().numpy()
            # imgaug
            if img_aug_matrix is not None:
                cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:3, 3].reshape(-1, 1)
                cur_coords = cur_coords[:2, :].transpose(0, 1)
            else:
                cur_coords = cur_coords.T


            cur_coords[..., 0] /= image_size[1]
            cur_coords[..., 1] /= image_size[0]
            cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

            this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
                         & (cur_coords[..., 0] < 1.0)
                         & (cur_coords[..., 1] > -1.0)
                         & (cur_coords[..., 1] < 1.0)
                         )

            mask.append(this_mask)
            reference_voxel_cam.append(cur_coords[:, :2])

        # sample img features
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            _, C, H, W = feat.size()
            feat = feat.view(batch_size, C, H, W)
            cam_feats = []
            for b in range(batch_size):
                reference_points_cam_lvl = reference_voxel_cam[b].view(-1, 1, 2)
                sampled_feat = F.grid_sample(feat[b].unsqueeze(dim=0), reference_points_cam_lvl.unsqueeze(dim=0))
                sampled_feat = sampled_feat.view(1, C, -1, 1).squeeze(0).squeeze(-1)
                cam_feats.append(sampled_feat)

            if not len(reference_voxel[0].shape) == 3:
                multi_cam_feats = torch.cat(multi_cam_feats, dim=1)
            sampled_feats.append(cam_feats)

        if not len(reference_voxel[0].shape) == 3:
            sampled_feats = torch.stack(sampled_feats).sum(0).transpose(0, 1)

        return reference_voxel_cam, mask, sampled_feats

    def img_fv_to_bev(self, mlvl_feats, bs, **kwargs):

        pts_metas = kwargs['pts_metas']
        pillars = pts_metas['pillars'][..., :3]
        pillar_coors = pts_metas['pillar_coors']
        num_points_in_pillar = pts_metas['pillars'].shape[1]
        ref_3d = []
        pillar_coors_list = []

        for i in range(bs):
            this_idx = pillar_coors[:, 0] == i
            this_coors = pillar_coors[this_idx]
            pillar_coors_list.append(this_coors)
            ref_3d.append(pillars[this_idx])

        reference_points_cam_stack, ref_mask_stack, ref_img_pillar_cam = self.img_point_sampling(
            ref_3d, mlvl_feats, bs, **kwargs)

        decorated_img_feat = torch.zeros([bs, self.embed_dims_img, self.bev_size, self.bev_size]).type_as(mlvl_feats[0])
        for b in range(bs):
            this_pillar_coors = pillar_coors_list[b]
            output = ref_img_pillar_cam[0][b].reshape(self.embed_dims_img, -1, num_points_in_pillar)
            output = output.sum(dim=2)
            decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output
        if self.scale_factor is None:
            decorated_img_feat = F.max_pool2d(decorated_img_feat, kernel_size=2, stride=2)
        else:
            decorated_img_feat = F.max_pool2d(decorated_img_feat, kernel_size=self.scale_factor, stride=self.scale_factor)


        return decorated_img_feat

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    @force_fp32()
    def forward(self,
                img_mlvl_feats,
                lidar_feats,
                bs,
                **kwargs):
        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats], bs, **kwargs)
        bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats[0]], dim=1))

        return bev_feats


class PointFusion_att(BaseModule):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    def __init__(self, num_points_in_pillar=10, embed_dims_img=256, embed_dims_radar=384, bev_size=256, scale_factor=None, region_shape=None, grid_size=None, region_drop_info=None):
        super(PointFusion_att, self).__init__()

        self.num_points_in_pillar = num_points_in_pillar
        self.bev_size = bev_size
        self.scale_factor = scale_factor
        self.region_shape = region_shape
        self.grid_size = grid_size
        self.region_drop_info = region_drop_info


        self.random_noise = 1.0

        self.embed_dims_img = embed_dims_img
        self.embed_dims_radar = embed_dims_radar
        self.conv_fusion = ConvModule(
            self.embed_dims_img + self.embed_dims_radar,
            self.embed_dims_radar,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.embed_dims_img + self.embed_dims_radar, self.embed_dims_radar, 3, 1, 1),
            nn.BatchNorm2d(self.embed_dims_radar),
            nn.ReLU(inplace=True))

        self.attention = nn.Sequential(
            nn.Conv2d(self.embed_dims_radar, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.cbam = CBAM(self.embed_dims_radar)

        def create_2D_grid(x_size, y_size):
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
            # NOTE: modified
            batch_x, batch_y = torch.meshgrid(
                *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
            )
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
            return coord_base

        self.bev_pos = create_2D_grid(self.bev_size, self.bev_size)

    def img_point_sampling(self, reference_voxel, mlvl_feats, batch_size=4, **kwargs):  # from UVTR

        img_aug_matrix = kwargs.get('img_aug_matrix', None)
        lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        lidar2image = kwargs.get('lidar2img', None)
        image_size = kwargs['img_shape']

        # Transfer to Point cloud range with X,Y,Z
        mask = []
        reference_voxel_cam = []

        for b in range(batch_size):
            cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            if img_aug_matrix is not None:
                cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            if lidar_aug_matrix is not None:
                cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, np.ndarray) else lidar2image[b]

            # inverse aug for pseudo points
            if lidar_aug_matrix is not None:
                cur_coords -= lidar_aug_matrix[b, :3, 3]
                cur_coords = torch.inverse(lidar_aug_matrix[b, :3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )

            # lidar2image
            if isinstance(lidar2image, np.ndarray):
                cur_lidar2image = torch.from_numpy(cur_lidar2image).to(cur_coords.device)
            cur_coords = cur_lidar2image[:3, :3].mm(cur_coords.T)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[2, :].clone()

            if self.grid_size:
                this_mask = (dist > 1e-5)&(dist < self.grid_size[3])
            else:
                this_mask = (dist > 1e-5)

            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            # cur_coords_vis = cur_coords.cpu().detach().numpy()
            # imgaug
            if img_aug_matrix is not None:
                cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:3, 3].reshape(-1, 1)
                cur_coords = cur_coords[:2, :].transpose(0, 1)
            else:
                cur_coords = cur_coords.T


            cur_coords[..., 0] /= image_size[1]
            cur_coords[..., 1] /= image_size[0]
            cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

            this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
                         & (cur_coords[..., 0] < 1.0)
                         & (cur_coords[..., 1] > -1.0)
                         & (cur_coords[..., 1] < 1.0)
                         )

            mask.append(this_mask)
            reference_voxel_cam.append(cur_coords[:, :2])

        # sample img features
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            _, C, H, W = feat.size()
            feat = feat.view(batch_size, C, H, W)
            cam_feats = []
            for b in range(batch_size):
                reference_points_cam_lvl = reference_voxel_cam[b].view(-1, 1, 2)
                sampled_feat = F.grid_sample(feat[b].unsqueeze(dim=0), reference_points_cam_lvl.unsqueeze(dim=0))
                sampled_feat = sampled_feat.view(1, C, -1, 1).squeeze(0).squeeze(-1)
                cam_feats.append(sampled_feat)

            if not len(reference_voxel[0].shape) == 3:
                multi_cam_feats = torch.cat(multi_cam_feats, dim=1)
            sampled_feats.append(cam_feats)

        if not len(reference_voxel[0].shape) == 3:
            sampled_feats = torch.stack(sampled_feats).sum(0).transpose(0, 1)

        return reference_voxel_cam, mask, sampled_feats

    def img_fv_to_bev(self, mlvl_feats, bs, **kwargs):

        pts_metas = kwargs['pts_metas']
        pillars = pts_metas['pillars'][..., :3]
        pillar_coors = pts_metas['pillar_coors']
        num_points_in_pillar = pts_metas['pillars'].shape[1]
        ref_3d = []
        pillar_coors_list = []

        for i in range(bs):
            this_idx = pillar_coors[:, 0] == i
            this_coors = pillar_coors[this_idx]
            pillar_coors_list.append(this_coors)
            ref_3d.append(pillars[this_idx])

        reference_points_cam_stack, ref_mask_stack, ref_img_pillar_cam = self.img_point_sampling(
            ref_3d, mlvl_feats, bs, **kwargs)

        decorated_img_feat = torch.zeros([bs, self.embed_dims_img, self.bev_size, self.bev_size]).type_as(mlvl_feats[0])
        for b in range(bs):
            this_pillar_coors = pillar_coors_list[b]
            output = ref_img_pillar_cam[0][b].reshape(self.embed_dims_img, -1, num_points_in_pillar)
            output = output.sum(dim=2)
            decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output
        if self.scale_factor is None:
            decorated_img_feat = F.max_pool2d(decorated_img_feat, kernel_size=2, stride=2)
        else:
            decorated_img_feat = F.max_pool2d(decorated_img_feat, kernel_size=self.scale_factor, stride=self.scale_factor)


        return decorated_img_feat

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    @force_fp32()
    def forward(self,
                img_mlvl_feats,
                lidar_feats,
                bs,
                **kwargs):
        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats], bs, **kwargs)
        bev_feats = self.fusion_conv(torch.cat([img_bev_feats, lidar_feats[0]], dim=1))
        attention_map = self.attention(bev_feats)
        bev_feats = bev_feats * attention_map
        bev_feats = self.cbam(bev_feats)

        return bev_feats



class PointFusion_sparse(BaseModule):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, num_points_in_pillar=10, embed_dims_img=256, embed_dims_radar=384, bev_size=256,
                 region_shape=None, grid_size=None, region_drop_info=None):
        super(PointFusion_sparse, self).__init__()

        self.num_points_in_pillar = num_points_in_pillar
        self.bev_size = bev_size
        self.region_shape = region_shape
        self.grid_size = grid_size
        self.region_drop_info = region_drop_info

        self.random_noise = 1.0

        self.embed_dims_img = embed_dims_img
        self.embed_dims_radar = embed_dims_radar

        self.bev_sparse_recover = BEVSparse1(dim=embed_dims_img, down_kernel_size=[3, 3, 3], down_stride=[1, 2, 2], num_SBB=[1, 1, 1], indice_key='BEVSlayer1', xy_only=True)
        self.conv_fusion = ConvModule(
            self.embed_dims_img + self.embed_dims_radar,
            self.embed_dims_radar,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )

        self.get_regions = nn.ModuleList()
        self.grid2region_att = nn.ModuleList()

        def create_2D_grid(x_size, y_size):
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
            # NOTE: modified
            batch_x, batch_y = torch.meshgrid(
                *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
            )
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
            return coord_base

        self.bev_pos = create_2D_grid(self.bev_size, self.bev_size)

    def img_point_sampling(self, reference_voxel, mlvl_feats, batch_size=4, **kwargs):  # from UVTR

        img_aug_matrix = kwargs.get('img_aug_matrix', None)
        lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        lidar2image = kwargs.get('lidar2img', None)
        image_size = kwargs['img_metas'][0]['ori_shape']

        # Transfer to Point cloud range with X,Y,Z
        mask = []
        reference_voxel_cam = []

        for b in range(batch_size):
            cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            if img_aug_matrix is not None:
                cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            if lidar_aug_matrix is not None:
                cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else \
                lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, np.ndarray) else lidar2image[b]

            # inverse aug for pseudo points
            if img_aug_matrix is not None:
                cur_coords -= cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )

            # lidar2image
            cur_lidar2image = torch.from_numpy(cur_lidar2image).to(cur_coords.device)
            cur_coords = cur_lidar2image[:3, :3].mm(cur_coords.T)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[2, :].clone()
            this_mask = (dist > 1e-5)

            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            # cur_coords_vis = cur_coords.cpu().detach().numpy()
            # imgaug
            if img_aug_matrix is not None:
                cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:2, :].transpose(1, 2)
            else:
                cur_coords = cur_coords.T

            cur_coords[..., 0] /= image_size[1]
            cur_coords[..., 1] /= image_size[0]
            cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

            this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
                         & (cur_coords[..., 0] < 1.0)
                         & (cur_coords[..., 1] > -1.0)
                         & (cur_coords[..., 1] < 1.0)
                         )

            mask.append(this_mask)
            reference_voxel_cam.append(cur_coords[:, :2])

        # sample img features
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            _, C, H, W = feat.size()
            feat = feat.view(batch_size, C, H, W)
            cam_feats = []
            for b in range(batch_size):
                reference_points_cam_lvl = reference_voxel_cam[b].view(-1, 1, 2)
                sampled_feat = F.grid_sample(feat[b].unsqueeze(dim=0), reference_points_cam_lvl.unsqueeze(dim=0))
                sampled_feat = sampled_feat.view(1, C, -1, 1).squeeze(0).squeeze(-1)
                cam_feats.append(sampled_feat)

            if not len(reference_voxel[0].shape) == 3:
                multi_cam_feats = torch.cat(multi_cam_feats, dim=1)
            sampled_feats.append(cam_feats)

        if not len(reference_voxel[0].shape) == 3:
            sampled_feats = torch.stack(sampled_feats).sum(0).transpose(0, 1)

        return reference_voxel_cam, mask, sampled_feats

    def img_fv_to_bev(self, mlvl_feats, bs, **kwargs):

        pts_metas = kwargs['pts_metas']
        pillars = pts_metas['pillars'][..., :3]
        pillar_coors = pts_metas['pillar_coors']
        num_points_in_pillar = pts_metas['pillars'].shape[1]
        ref_3d = []
        pillar_coors_list = []

        for i in range(bs):
            this_idx = pillar_coors[:, 0] == i
            this_coors = pillar_coors[this_idx]
            pillar_coors_list.append(this_coors)
            ref_3d.append(pillars[this_idx])

        reference_points_cam_stack, ref_mask_stack, ref_img_pillar_cam = self.img_point_sampling(
            ref_3d, mlvl_feats, bs, **kwargs)


        ######################################one scale sparse conv#####################################################
        pillar_coors_sparse = torch.cat((pillar_coors[:, :1], pillar_coors[:, 2:]), dim=1)
        output_list = []
        for b in range(bs):
            output = ref_img_pillar_cam[0][b].reshape(self.embed_dims_img, -1, num_points_in_pillar)
            output = output.sum(dim=2)
            output_list.append(output)
            # decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output
        output_features = torch.cat(output_list, dim=1).T
        spatial_shape = [self.bev_size, self.bev_size]
        sparse_input = SparseConvTensor(output_features, pillar_coors_sparse, spatial_shape, bs)
        sparse_output = self.bev_sparse_recover(sparse_input)
        sparse_output = sparse_output.dense()
        decorated_img_feat = F.max_pool2d(sparse_output, kernel_size=2, stride=2)

        return decorated_img_feat

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    @force_fp32()
    def forward(self,
                img_mlvl_feats,
                lidar_feats,
                bs,
                **kwargs):
        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats], bs, **kwargs)
        bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats[0]], dim=1))

        return bev_feats




class PointFusion_region(BaseModule):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """
    def __init__(self, num_points_in_pillar=10, embed_dims_img=256, embed_dims_radar=384, bev_size=256, region_shape=(6, 6, 1), grid_size=[[160, 160, 1]], region_drop_info=None):
        super(PointFusion_region, self).__init__()

        self.num_points_in_pillar = num_points_in_pillar
        self.bev_size = bev_size
        self.region_shape = region_shape
        self.grid_size = grid_size
        self.region_drop_info = region_drop_info

        self.random_noise = 1.0

        self.embed_dims_img = embed_dims_img
        self.embed_dims_radar = embed_dims_radar
        self.conv_fusion = ConvModule(
            self.embed_dims_img + self.embed_dims_radar,
            self.embed_dims_radar,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            )

        self.get_regions = nn.ModuleList()
        self.grid2region_att = nn.ModuleList()
        for l in range(len(region_shape)):
            this_embed_dim = self.embed_dims_radar
            region_metas = dict(
                type='SSTInputLayerV2',
                window_shape=region_shape[l],
                sparse_shape=grid_size[l],
                shuffle_voxels=True,
                drop_info=region_drop_info[l],
                pos_temperature=1000,
                normalize_pos=False,
                pos_embed=this_embed_dim,
            )
            grid2region_att = dict(
                type='SSTv2',
                d_model=[this_embed_dim,] * 4,
                nhead=[8, ] * 4,
                num_blocks=1,
                dim_feedforward=[this_embed_dim, ] * 4,
                output_shape=grid_size[l][:2],
                in_channel=self.embed_dims_radar if l==0 else None,
            )
            self.get_regions.append(builder.build_middle_encoder(region_metas))
            self.grid2region_att.append(builder.build_backbone(grid2region_att))

        def create_2D_grid(x_size, y_size):
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
            # NOTE: modified
            batch_x, batch_y = torch.meshgrid(
                *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
            )
            batch_x = batch_x + 0.5
            batch_y = batch_y + 0.5
            coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
            coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
            return coord_base

        self.bev_pos = create_2D_grid(self.bev_size, self.bev_size)

    def img_point_sampling(self, reference_voxel, mlvl_feats, batch_size=4, **kwargs):  # from UVTR

        img_aug_matrix = kwargs.get('img_aug_matrix', None)
        lidar_aug_matrix = kwargs.get('lidar_aug_matrix', None)
        lidar2image = kwargs.get('lidar2img', None)
        image_size = kwargs['img_metas'][0]['ori_shape']

        # Transfer to Point cloud range with X,Y,Z
        mask = []
        reference_voxel_cam = []

        for b in range(batch_size):
            cur_coords = reference_voxel[b].reshape(-1, 3)[:, :3].clone()
            if img_aug_matrix is not None:
                cur_img_aug_matrix = img_aug_matrix[b] if not isinstance(img_aug_matrix, list) else img_aug_matrix[0][b]
            if lidar_aug_matrix is not None:
                cur_lidar_aug_matrix = lidar_aug_matrix[b] if not isinstance(lidar_aug_matrix, list) else lidar_aug_matrix[0][b]
            cur_lidar2image = lidar2image[b] if not isinstance(lidar2image, np.ndarray) else lidar2image[b]

            # inverse aug for pseudo points
            if img_aug_matrix is not None:
                cur_coords -= cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                )

            # lidar2image
            cur_lidar2image = torch.from_numpy(cur_lidar2image).to(cur_coords.device)
            cur_coords = cur_lidar2image[:3, :3].mm(cur_coords.T)  # cur_coords: [3, N]
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)

            if self.random_noise is not None and self.training:
                seed = np.random.rand()
                if seed > 0.5:
                    cur_coords += random.uniform(-self.random_noise, self.random_noise)

            # get 2d coords
            dist = cur_coords[2, :].clone()
            this_mask = (dist > 1e-5)

            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]

            cur_coords_vis = cur_coords.cpu().detach().numpy()
            # imgaug
            if img_aug_matrix is not None:
                cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:2, :].transpose(1, 2)
            else:
                cur_coords = cur_coords.T


            cur_coords[..., 0] /= image_size[1]
            cur_coords[..., 1] /= image_size[0]
            cur_coords = (cur_coords - 0.5) * 2  # to [-1, +1]

            this_mask = (this_mask & (cur_coords[..., 0] > -1.0)
                         & (cur_coords[..., 0] < 1.0)
                         & (cur_coords[..., 1] > -1.0)
                         & (cur_coords[..., 1] < 1.0)
                         )

            mask.append(this_mask)
            reference_voxel_cam.append(cur_coords[:, :2])

        # sample img features
        sampled_feats = []
        for lvl, feat in enumerate(mlvl_feats):
            _, C, H, W = feat.size()
            feat = feat.view(batch_size, C, H, W)
            cam_feats = []
            for b in range(batch_size):
                reference_points_cam_lvl = reference_voxel_cam[b].view(-1, 1, 2)
                sampled_feat = F.grid_sample(feat[b].unsqueeze(dim=0), reference_points_cam_lvl.unsqueeze(dim=0))
                sampled_feat = sampled_feat.view(1, C, -1, 1).squeeze(0).squeeze(-1)
                cam_feats.append(sampled_feat)

            if not len(reference_voxel[0].shape) == 3:
                multi_cam_feats = torch.cat(multi_cam_feats, dim=1)
            sampled_feats.append(cam_feats)

        if not len(reference_voxel[0].shape) == 3:
            sampled_feats = torch.stack(sampled_feats).sum(0).transpose(0, 1)

        return reference_voxel_cam, mask, sampled_feats

    def img_fv_to_bev(self, mlvl_feats, bs, **kwargs):

        pts_metas = kwargs['pts_metas']
        pillars = pts_metas['pillars'][..., :3]
        pillar_coors = pts_metas['pillar_coors']
        num_points_in_pillar = pts_metas['pillars'].shape[1]
        ref_3d = []
        pillar_coors_list = []

        for i in range(bs):
            this_idx = pillar_coors[:, 0] == i
            this_coors = pillar_coors[this_idx]
            pillar_coors_list.append(this_coors)
            ref_3d.append(pillars[this_idx])

        reference_points_cam_stack, ref_mask_stack, ref_img_pillar_cam = self.img_point_sampling(
            ref_3d, mlvl_feats, bs, **kwargs)

        decorated_img_feat = torch.zeros([bs, self.embed_dims_img, self.bev_size, self.bev_size]).type_as(mlvl_feats[0])
        for b in range(bs):
            this_pillar_coors = pillar_coors_list[b]
            output = ref_img_pillar_cam[0][b].reshape(self.embed_dims_img, -1, num_points_in_pillar)
            output = output.sum(dim=2)
            decorated_img_feat[b, :, this_pillar_coors[:, 2].long(), this_pillar_coors[:, 3].long()] = output
        decorated_img_feat = F.max_pool2d(decorated_img_feat, kernel_size=2, stride=2)

        return decorated_img_feat

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    @force_fp32()
    def forward(self,
                img_mlvl_feats,
                lidar_feats,
                bs,
                **kwargs):
        img_bev_feats = self.img_fv_to_bev([img_mlvl_feats], bs, **kwargs)

        # kwargs.update(dict(img_bev_feats=img_bev_feats))
        # kwargs.update(dict(lidar_feats=lidar_feats))
        #
        bev_feats = self.conv_fusion(torch.cat([img_bev_feats, lidar_feats[0]], dim=1))
        #
        grid_features = bev_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, bev_feats.shape[1])
        bev_coords = self.create_dense_coord(int(self.bev_size/2), int(self.bev_size/2), bs).type_as(grid_features).int()
        this_coords = []
        for k in range(bs):
            this_coord = bev_coords[k].reshape(4, -1).transpose(1, 0)
            this_coords.append(this_coord)
        grid_coords = torch.cat(this_coords, dim=0)
        #
        # pts_backbone = kwargs.get('pts_backbone', None)
        #
        # ins_hm = None
        return_feats = []
        for i in range(len(self.get_regions)):
            x = self.get_regions[i](grid_features, grid_coords, bs)
            x = self.grid2region_att[i](x)

            # if i == 0:
            #     x[0], ins_hm = self.instance_fusion(bev_feats, x[0], bs, **kwargs)
        #
        #     grid_features, grid_coords, this_feat = pts_backbone(x, 'stage{}'.format(i+1))
        #     return_feats.append(this_feat)

        return bev_feats



