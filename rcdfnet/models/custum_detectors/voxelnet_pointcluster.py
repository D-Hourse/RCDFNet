# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from rcdfnet.core import bbox3d2result, merge_aug_bboxes_3d
from rcdfnet.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector
from ..sparse_det.point_cluster_spconv import Point_Cluster_Conv
from rcdfnet.ops.spconv import *


voxelize1 = 1
voxelize2 = 2

def Density_extract(pts, R):
    pts_return = []
    epsilon = 1e-6
    for pts_per_batch in pts:

        coords = pts_per_batch[:, :3]
        # 计算每个点与其他所有点的欧式距离 (N, N) 形状的矩阵
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, 1, 3) - (1, N, 3)
        dists = torch.sqrt(torch.sum(diff ** 2, dim=2))  # 计算欧式距离，形状 (N, N)

        # 计算高斯核函数
        kernel = (pts_per_batch.unsqueeze(1) - pts_per_batch.squeeze(0))/R
        Gau_kernel = torch.exp(-torch.sum(kernel ** 2, dim=2))

        # 找到在半径 R 内的点 (N, N) 布尔矩阵
        within_radius = dists <= R
        within_radius_sum = torch.sum(within_radius, dim=1)

        # 计算核函数的和
        Gau_kernel_sums = torch.sum(Gau_kernel * within_radius.float(), dim=1) / (within_radius_sum * (R ** 3))  # 形状 (N,)

        # 归一化核函数
        mean = torch.mean(Gau_kernel_sums)
        std = torch.std(Gau_kernel_sums)
        normalized_tensor = (Gau_kernel_sums - mean) / (std+epsilon)
        pts_return.append(torch.cat((pts_per_batch[:, :3], normalized_tensor.unsqueeze(dim=1)), dim=1))

    return pts_return

@DETECTORS.register_module()
class Voxel_ClusterNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 pts_voxel_layer_cluster=None,
                 pts_cluster_spconv=None,
                 spatial_shape=None,
                 output_dim=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(Voxel_ClusterNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.voxel_encoder_cluster = Voxelization(**pts_voxel_layer_cluster)
        self.pts_spconv_R1 = Point_Cluster_Conv(**pts_cluster_spconv)
        self.pts_spconv_R2 = Point_Cluster_Conv(**pts_cluster_spconv)
        self.spatial_shape = spatial_shape
        self.reduce_channel1 = torch.nn.Conv2d(in_channels=int(output_dim * spatial_shape[2]), out_channels=int(output_dim), kernel_size=1)
        self.reduce_channel2 = torch.nn.Conv2d(in_channels=int(output_dim * spatial_shape[2]), out_channels=int(output_dim), kernel_size=1)


    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points, voxelize1)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1

        x = self.middle_encoder(voxel_features, coors, batch_size)

        x = self.backbone(x)

        points_cluster_R1 = Density_extract(points, R=1.5)
        points_cluster_R2 = Density_extract(points, R=4)

        voxels_cluster1, num_points_cluster1, coors_cluster1 = self.voxelize(points_cluster_R1, voxelize2)
        voxels_cluster2, num_points_cluster2, coors_cluster2 = self.voxelize(points_cluster_R2, voxelize2)


        order = torch.tensor([0, 3, 2, 1]).to(coors_cluster1.device)
        sparse_input_R1 = SparseConvTensor(voxels_cluster1.squeeze(dim=1), coors_cluster1.index_select(1, order), self.spatial_shape, batch_size)
        sparse_input_R2 = SparseConvTensor(voxels_cluster2.squeeze(dim=1), coors_cluster2.index_select(1, order), self.spatial_shape, batch_size)
        dense_out_R1 = self.pts_spconv_R1(sparse_input_R1)
        dense_out_R2 = self.pts_spconv_R2(sparse_input_R2)

        # 平均聚合维度
        dense_out_R1 = [torch.max(dense_output, dim=-1).values for dense_output in dense_out_R1]
        dense_out_R2 = [torch.max(dense_output, dim=-1).values for dense_output in dense_out_R2]

        # conv聚合维度
        # batch_size, C, H, W, D = sparse_out_R1.shape
        # sparse_out_R1 = sparse_out_R1.view(batch_size, C * D, H, W)
        # sparse_out_R2 = sparse_out_R2.view(batch_size, C * D, H, W)
        # sparse_out_R1 = self.reduce_channel1(sparse_out_R1)
        # sparse_out_R2 = self.reduce_channel2(sparse_out_R2)

        # sparse_out = torch.cat((sparse_out_R1, sparse_out_R2), dim=1)
        x_temp = ()
        for idx in range(len(x)):
            feature = torch.cat((dense_out_R1[idx], dense_out_R2[idx]), dim=1)
            feature = torch.cat((x[idx], feature), dim=1)
            x_temp = x_temp +(feature,)


        if self.with_neck:
            x = self.neck(x_temp)

        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, pattern):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            if pattern == voxelize1:
                res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            else:
                res_voxels, res_coors, res_num_points = self.voxel_encoder_cluster(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
