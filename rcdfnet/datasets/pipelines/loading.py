# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import cv2
import torch
import os

from rcdfnet.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import math

from rcdfnet.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes,CameraInstance3DBoxes


from numpy.linalg import inv
from mmcv.runner import get_dist_info

from typing import Any, Dict, Tuple
from functools import reduce
from rcdfnet.core.bbox import limit_period

import torch.nn.functional as F
import torchvision.transforms as transforms


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadSegsFromFile(object):
    """ Load an image Segmentation Label from file

    Args:
        kwargs（dict）：Arguments are the same as those in \
            :class:'LoadImageFromFile'.
    """
    def __init__(self,
                 resize=None,
                 file_client_args=dict(backend='disk')
                 ):
        self.resize_size = resize
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_seg(self, seg_filename):
        """Private function to load point clouds data.

        Args:
            seg_filename (str): Filename of segmentation label.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if seg_filename.endswith('.npy'):
            seg_label = np.load(seg_filename)
        else:
            seg_label = np.fromfile(seg_filename, dtype=np.float)
        return seg_label

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        seg_filename = results['seg_filename']
        seg_label = self._load_seg(seg_filename)

        if self.resize_size:
            seg_label = cv2.resize(seg_label, self.resize_size, interpolation=cv2.INTER_LINEAR)

        results['seg_label'] = seg_label

        return results

@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadDepthsFromFile(object):
    """Load Depths From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 3.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=3,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['depth_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['depth'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str



@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None


    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        if not os.path.exists(pts_filename):
            print('path is not exist')
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str

##############################################new#############################################
@PIPELINES.register_module()
class PointToMultiViewDepthVoD(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        # print(depth.max(), depth.min())
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        depth < self.grid_config['depth'][1]) & (
                        depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_img = results['depth'].tensor
        # import pdb;pdb.set_trace()
        # imgs, rots, trans, intrins = results['img_inputs'][:4]
        imgs, sensor2keyegos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        lidar2img_list = []

        cid = 0
        cam2img = np.eye(4, dtype=np.float32)
        cam2img = torch.from_numpy(cam2img)
        # print(intrins.shape)
        cam2img = intrins[cid]
        lidar2cam = torch.inverse(sensor2keyegos[0])
        lidar2img = cam2img.matmul(lidar2cam)

        points_img = points_img.matmul(
            post_rots[cid].T) + post_trans[cid:cid + 1, :]
        depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                         imgs.shape[3])
        depth_map_list.append(depth_map)
        lidar2img_list.append(lidar2img)

        depth_map = torch.stack(depth_map_list)
        lidar2img_list = torch.stack(lidar2img_list)
        results['lidar2img'] = lidar2img_list
        results['gt_depth'] = depth_map
        return results



def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    # mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    # std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img
@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            ego_cam='CAM_FRONT',
            add_adj_bbox=False,
            with_stereo=False,
            with_future_pred=False,
            img_norm_cfg=None,
            ignore=[],
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.with_future_pred = with_future_pred
        self.add_adj_bbox = add_adj_bbox
        self.img_norm_cfg = img_norm_cfg
        self.with_stereo = with_stereo
        self.ignore = ignore
        if len(ignore) > 0:
            print(self.ignore)

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sweep2key_transformation(self,
                                     cam_info,
                                     key_info,
                                     cam_name,
                                     ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepego2keyego = global2keyego @ sweepego2global

        return sweepego2keyego

    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names

        canvas = []
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            results['img_file_paths'][cam_name] = filename  # for vis
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2
            # print(cam_name, self.ignore)
            if cam_name in self.ignore:
                canvas.append(np.zeros_like(np.array(img)))
                imgs.append(torch.zeros_like(self.normalize_img(img)))
            else:
                canvas.append(np.array(img))
                imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    if cam_name in self.ignore:
                        imgs.append(torch.zeros_like(self.normalize_img(img_adjacent)))
                    else:
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
            if self.add_adj_bbox:
                results['adjacent_bboxes'] = self.align_adj_bbox2keyego(results)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['img_shape'] = [(self.data_config['input_size'][0], self.data_config['input_size'][1]) for _ in
                                range(6)]
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_file_paths'] = {}
        if self.add_adj_bbox:
            results['adjacent_bboxes'] = self.get_adjacent_bboxes(results)
        results['img_inputs'] = self.get_inputs(results)
        results['input_shape'] = self.data_config['input_size']  # new
        return results

    def get_adjacent_bboxes(self, results):
        adjacent_bboxes = list()
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt
                break
            adjacent_bboxes.append(adj_info['ann_infos'])
        return adjacent_bboxes

    def align_adj_bbox2keyego(self, results):
        cam_name = self.choose_cams()[0]
        ret_list = []
        for idx, adj_info in enumerate(results['adjacent']):
            if self.with_stereo and idx > 0:  # reference frame不用读gt 也不用align
                break
            sweepego2keyego = self.get_sweep2key_transformation(adj_info,
                                                                results['curr'],
                                                                cam_name,
                                                                self.ego_cam)
            adj_bbox, adj_labels = results['adjacent_bboxes'][idx]
            adj_bbox = torch.Tensor(adj_bbox)
            adj_labels = torch.tensor(adj_labels)
            gt_bbox = adj_bbox
            if len(adj_bbox) == 0:
                adj_bbox = torch.zeros(0, 9)
                ret_list.append((adj_bbox, adj_labels))
                continue
            # center
            homo_sweep_center = torch.cat([gt_bbox[:, :3], torch.ones_like(gt_bbox[:, 0:1])], dim=-1)
            homo_key_center = (sweepego2keyego @ homo_sweep_center.t()).t()  # [4, N]
            # velo
            rot = sweepego2keyego[:3, :3]
            homo_sweep_velo = torch.cat([gt_bbox[:, 7:], torch.zeros_like(gt_bbox[:, 0:1])], dim=-1)
            homo_key_velo = (rot @ homo_sweep_velo.t()).t()

            # yaw
            def get_new_yaw(box_cam, extrinsic):
                corners = box_cam.corners
                cam2lidar_rt = torch.tensor(extrinsic)
                N = corners.shape[0]
                corners = corners.reshape(N * 8, 3)
                extended_xyz = torch.cat(
                    [corners, corners.new_ones(corners.size(0), 1)], dim=-1)
                corners = extended_xyz @ cam2lidar_rt.T
                corners = corners.reshape(N, 8, 4)[:, :, :3]
                yaw = np.arctan2(corners[:, 1, 1] - corners[:, 2, 1], corners[:, 1, 0] - corners[:, 2, 0])

                def limit_period(val, offset=0.5, period=np.pi):
                    """Limit the value into a period for periodic function.

                    Args:
                        val (np.ndarray): The value to be converted.
                        offset (float, optional): Offset to set the value range. \
                            Defaults to 0.5.
                        period (float, optional): Period of the value. Defaults to np.pi.

                    Returns:
                        torch.Tensor: Value in the range of \
                            [-offset * period, (1-offset) * period]
                    """
                    return val - np.floor(val / period + offset) * period

                return limit_period(yaw + (np.pi / 2), period=np.pi * 2)

            new_yaw_sweep = get_new_yaw(LiDARInstance3DBoxes(adj_bbox, box_dim=adj_bbox.shape[-1],
                                                             origin=(0.5, 0.5, 0.5)), sweepego2keyego).reshape(-1, 1)
            adj_bbox = torch.cat([homo_key_center[:, :3], gt_bbox[:, 3:6], new_yaw_sweep, homo_key_velo[:, :2]], dim=-1)
            ret_list.append((adj_bbox, adj_labels))

        return ret_list


@PIPELINES.register_module()
class PrepareImageInputsVODDebug(PrepareImageInputs):
    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            ego_cam='CAM_FRONT',
            add_adj_bbox=False,
            with_stereo=False,
            with_future_pred=False,
            img_norm_cfg=None,
            ignore=[],
            radar=False
    ):
        super(PrepareImageInputsVODDebug, self).__init__(data_config,is_train,sequential,
            ego_cam,add_adj_bbox,with_stereo,with_future_pred,
            img_norm_cfg,ignore)
        self.radar = radar
    def radar_transform(self, radar, post_rots, post_trans):
        radar_points = radar.tensor
        radar_points_coor = radar_points[:, :2].unsqueeze(dim=1)
        radar_points_coor = post_rots.matmul(radar_points_coor.unsqueeze(-1)).squeeze(-1) + post_trans
        radar_points[:, :2] = radar_points_coor.squeeze(dim=1)
        radar.tensor = radar_points
        return radar

    def get_inputs(self, results, flip=None, scale=None):
        assert not self.sequential
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        # cam_names = self.choose_cams()
        # results['cam_names'] = cam_names
        filename = results['img_info']['filename']

        canvas = []
        # for cam_name in cam_names:
        # cam_data = results['curr']['cams'][cam_name]
        # filename = cam_data['data_path']
        img = Image.open(filename)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        intrin = torch.Tensor(results['P2'])[..., :, :]

        # ego2global = torch.inverse(torch.Tensor(results['Trv2c']))
        # sensor2ego = torch.eye(4)
        sensor2ego = torch.inverse(torch.Tensor(results['Trv2c']))
        ego2global = torch.eye(4)
        # image view augmentation (resize, crop, horizontal flip, rotate)
        img_augs = self.sample_augmentation(
            H=img.height, W=img.width, flip=flip, scale=scale)
        resize, resize_dims, crop, flip, rotate = img_augs
        img, post_rot2, post_tran2 = \
            self.img_transform(img, post_rot,
                               post_tran,
                               resize=resize,
                               resize_dims=resize_dims,
                               crop=crop,
                               flip=flip,
                               rotate=rotate)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3)
        post_rot = torch.eye(3)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2
        # print(cam_name, self.ignore)

        canvas.append(np.array(img))
        imgs.append(self.normalize_img(img))

        intrins.append(intrin)
        sensor2egos.append(sensor2ego)
        ego2globals.append(ego2global)
        post_rots.append(post_rot)
        post_trans.append(post_tran)

        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        results['canvas'] = canvas
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

        if self.radar:
            radar_points = results['points']
            radar_points = self.radar_transform(radar_points, post_rot2, post_tran2)
            results['points'] = radar_points

        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes, is_train=True, sequential=False, align_adj_bbox=False, with_hop=False,
                 is_val=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.sequential = sequential
        self.align_adj_bbox = align_adj_bbox
        self.with_hop = with_hop
        self.is_val = is_val  # if is_val then load bbox gt for seg gt

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        fsr_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    fsr_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                    fsr_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, fsr_mat, rot_mat, flip_mat, scale_mat

    def __call__(self, results):
        if self.is_train or self.is_val:
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        else:
            gt_boxes = torch.zeros(0, 9)
            gt_labels = torch.zeros(0, 1)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy  # save

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot, rm, fm, sm = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                           flip_dx, flip_dy)
        bda_mat[:3, :3] = rm
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels

        if 'points' in results:
            points = results['points']
            lidar2ego = results['lidar2ego']
            # points.rotate(lidar2ego[:3, :3].T)
            # points.tensor[:, :3] = points.tensor[:, :3] + lidar2ego[:3, 3]
            points.tensor[:, :3] = (bda_rot @ points.tensor[:, :3].unsqueeze(-1)).squeeze(-1)
            results['points'] = points

            if False:  # for debug vis
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                import random
                print('\n******** BEGIN PRINT GT and LiDAR POINTS **********\n')
                corner = results['gt_bboxes_3d'].corners
                fig = plt.figure(figsize=(16, 16))
                plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
                plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
                for i in range(corner.shape[0]):
                    x1 = corner[i][0][0]
                    y1 = corner[i][0][1]
                    x2 = corner[i][2][0]
                    y2 = corner[i][2][1]
                    x3 = corner[i][6][0]
                    y3 = corner[i][6][1]
                    x4 = corner[i][4][0]
                    y4 = corner[i][4][1]
                    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=1)
                plt.scatter(points.tensor[:, 0].view(-1).numpy(), points.tensor[:, 1].view(-1).numpy(), s=0.5,
                            c='black')
                plt.savefig("/home/xiazhongyu/vis/gt" + str(random.randint(1, 9999)) + ".png")
                print('\n******** END PRINT GT and LiDAR POINTS **********\n')
                exit(0)

        if 'img_inputs' in results:
            imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots,
                                     post_trans, bda_rot)
            ego2img_rts = []
            if not self.sequential and self.with_hop:
                sensor2keyegos = self.get_sensor2keyego_transformation(sensor2egos, ego2globals)
                for sensor2keyego, intrin, post_rot, post_tran in zip(
                        sensor2keyegos, intrins, post_rots, post_trans):
                    rot = sensor2keyego[:3, :3]
                    tran = sensor2keyego[:3, 3]
                    viewpad = torch.eye(3).to(imgs.device)
                    viewpad[:post_rot.shape[0], :post_rot.shape[1]] = \
                        post_rot @ intrin[:post_rot.shape[0], :post_rot.shape[1]]
                    viewpad[:post_tran.shape[0], 2] += post_tran
                    intrinsic = viewpad

                    # need type float
                    ego2img_r = intrinsic.float() @ torch.linalg.inv(rot.float()) @ torch.linalg.inv(bda_rot.float())
                    ego2img_t = -intrinsic.float() @ torch.linalg.inv(rot.float()) @ tran.float()
                    ego2img_rt = torch.eye(4).to(imgs.device)
                    ego2img_rt[:3, :3] = ego2img_r
                    ego2img_rt[:3, 3] = ego2img_t
                    '''
                    X_{3d} = bda * (rots * (intrinsic)^(-1) * X_{img} + trans)
                    bda^(-1) * X_{3d} = rots * (intrinsic)^(-1) * X_{img} + trans
                    bda^(-1) * X_{3d} - trans = rots * (intrinsic)^(-1) * X_{img}
                    intrinsic * rots^(-1) * (bda^(-1) * X_{3d} - trans) = X_{img}
                    intrinsic * rots^(-1) * bda^(-1) * X_{3d} - intrinsic * rots^(-1) * trans = X_{img}
                    rotate = intrinsic * rots^(-1) * bda^(-1)
                    translation = - intrinsic * rots^(-1) * trans
                    '''
                    ego2img_rts.append(ego2img_rt)
                ego2img_rts = torch.stack(ego2img_rts, dim=0)
            if self.align_adj_bbox:
                results = self.align_adj_bbox_bda(results, rotate_bda, scale_bda,
                                                  flip_dx, flip_dy)
            results['lidar2img'] = np.asarray(ego2img_rts)

            if 'img_inputs_lt' in results.keys():
                imgs_lt, rots_lt, trans_lt, intrins_lt = results['img_inputs_lt'][:4]
                post_rots_lt, post_trans_lt = results['img_inputs_lt'][4:]
                results['img_inputs_lt'] = (imgs_lt, rots_lt, trans_lt, intrins_lt,
                                            post_rots_lt, post_trans_lt, bda_rot)

        results["bda_r"] = bda_mat
        results["bda_f"] = (flip_dx, flip_dy)
        results["bda_s"] = scale_bda
        return results

    def get_sensor2keyego_transformation(self, sensor2egos, ego2globals):
        # sensor2ego -> sweep sensor to sweep ego
        # ego2globals -> sweep ego to global
        sensor2keyegos = []
        keyego2global = ego2globals[0]  # assert key ego is frame 0 with CAM_FRONT
        global2keyego = torch.inverse(keyego2global.double())
        for sensor2ego, ego2global in zip(sensor2egos, ego2globals):
            # calculate the transformation from sweep sensor to key ego
            sensor2keyego = global2keyego @ ego2global.double() @ sensor2ego.double()
            sensor2keyegos.append(sensor2keyego)
        return sensor2keyegos

    def align_adj_bbox_bda(self, results, rotate_bda, scale_bda, flip_dx, flip_dy):
        for adjacent_bboxes in results['adjacent_bboxes']:
            adj_bbox, adj_label = adjacent_bboxes
            gt_boxes = adj_bbox
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            gt_boxes, _, _, _, _ = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                                      flip_dx, flip_dy)
            if not 'adj_gt_3d' in results.keys():
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                         origin=(0.5, 0.5, 0.5))
                adj_labels_3d = adj_label
                results['adj_gt_3d'] = [[adj_bboxes_3d, adj_labels_3d]]
            else:
                adj_bboxes_3d = \
                    LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                         origin=(0.5, 0.5, 0.5))
                results['adj_gt_3d'].append([
                    adj_bboxes_3d, adj_label
                ])
        return results

@PIPELINES.register_module()
class LoadAnnotationsBEVDepthVOD(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                         6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            # velocity
            # gt_boxes[:, 7:] = (
            #     rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        # import ipdb; ipdb.set_trace()
        ann_info = results['ann_info']
        # results['ann_info']['gt_bboxes_3d']
        gt_boxes, gt_labels = ann_info['gt_bboxes_3d'], ann_info['gt_labels_3d']
        gt_boxes, gt_labels = gt_boxes.tensor.clone(), torch.tensor(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy  # save
        # print(rotate_bda)
        # print(scale_bda)
        # print(flip_dx, flip_dy)

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 7)
        # results['gt_bboxes_3d'] = CameraInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1])
        #  origin=(0.5, 0.5, 0.))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        return results
