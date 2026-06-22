import os
import argparse
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import pdb


def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4")
    return transform.dot(points.T).T


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    return np.round(uvw[:2].T).astype(np.int32)


def canvas_crop(points, image_size, points_depth=None):
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)
    return idx


def project_pcl_to_image(point_cloud, t_camera_pcl, camera_projection_matrix, image_shape):
    point_homo = np.hstack((point_cloud[:, :3],
                            np.ones((point_cloud.shape[0], 1),
                                    dtype=np.float32)))
    points_camera_frame = homogeneous_transformation(
        point_homo, transform=t_camera_pcl)
    point_depth = points_camera_frame[:, 2]
    uvs = project_3d_to_2d(
        points=points_camera_frame,
        projection_matrix=camera_projection_matrix)
    filtered_idx = canvas_crop(
        points=uvs, image_size=image_shape, points_depth=point_depth)
    return uvs[filtered_idx], point_depth[filtered_idx]

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):
    pc = LidarPointCloud(pc)

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(cam_calibrated_sensor['translation']))
    pc.rotate(Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


def map_pointcloud_to_image2(
    image_info,
    pointcloud,
    depth
):
    # image_path = data_root + '/lidar/' + image_info['image_path']
    image_path = data_root +  image_info['image_path']
    image = cv2.imread(image_path)
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    colormap = plt.get_cmap('jet')
    colors = (colormap(depth_normalized)[:, :3] * 255).astype(np.uint8)

    for (point, color) in zip(pointcloud, colors):
        point = tuple(point)
        color = tuple(map(int, color))
        cv2.circle(image, point, radius=5, color=color, thickness=-1)
    cv2.imshow('Depth Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


data_root = None
INFO_PATHS = None
radar_num_features = None
output_dir = None
overwrite = False

def worker_lidar(info):
    debug = False
    lidar_path = data_root + '/lidar/' + info['point_cloud']['velodyne_path']
    points = np.fromfile(os.path.join(lidar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 4)
    lidar_to_cam_calib = info['calib']['Tr_velo_to_cam']
    cam_intrinsic = info['calib']['P2']
    image_shape = info['image']['image_shape']
    uvs, point_depth = project_pcl_to_image(points, lidar_to_cam_calib, cam_intrinsic, image_shape)
    unique_uvs, indices = np.unique(uvs, axis=0, return_index=True)
    unique_point_depth = point_depth[indices]
    image_name = os.path.basename(info['image']['image_path'])
    # if image_name == '00581.jpg':
    #     print("end")
    #     debug=True

    if debug:
        map_pointcloud_to_image2(info['image'], unique_uvs, unique_point_depth)

    np.concatenate([unique_uvs, unique_point_depth[:, None]], axis=1).astype(np.float32).flatten().tofile(os.path.join(data_root, 'depth_gt', f'{image_name}.bin'))

    # for i, cam_key in enumerate(cam_keys):
    #     cam_calibrated_sensor = info['cam_infos'][cam_key]['calibrated_sensor']
    #     cam_ego_pose = info['cam_infos'][cam_key]['ego_pose']
    #     img = mmcv.imread(
    #         os.path.join(data_root, info['cam_infos'][cam_key]['filename']))
    #     pts_img, depth = map_pointcloud_to_image(
    #         pc.points.copy(), img, cam_calibrated_sensor, cam_ego_pose)
    #     file_name = os.path.split(info['cam_infos'][cam_key]['filename'])[-1]
    #     np.concatenate([pts_img[:2, :].T, depth[:, None]],
    #                    axis=1).astype(np.float32).flatten().tofile(
    #                        os.path.join(data_root, 'depth_gt',
    #                                     f'{file_name}.bin'))
    # plt.savefig(f"{sample_idx}")

def worker_radar(info):
    debug = False
    radar_path = os.path.join(data_root, info['point_cloud']['velodyne_path'])
    points = np.fromfile(os.path.join(radar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, radar_num_features)
    points = points[:, :3]

    radar_to_cam_calib = info['calib']['Tr_velo_to_cam']
    cam_intrinsic = info['calib']['P2']
    image_shape = info['image']['image_shape']
    uvs, point_depth = project_pcl_to_image(points, radar_to_cam_calib, cam_intrinsic, image_shape)
    unique_uvs, indices = np.unique(uvs, axis=0, return_index=True)
    unique_point_depth = point_depth[indices]
    image_name = os.path.basename(info['image']['image_path'])

    if debug:
        map_pointcloud_to_image2(info['image'], unique_uvs, unique_point_depth)

    out_path = os.path.join(output_dir, f'{image_name}.bin')
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f'{out_path} already exists. Remove it or rerun with --overwrite.')
    np.concatenate([unique_uvs, unique_point_depth[:, None]], axis=1).astype(np.float32).flatten().tofile(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate radar depth ground truth.')
    parser.add_argument('--data-root', required=True, help='Dataset root used by the config data_root.')
    parser.add_argument(
        '--info-paths',
        nargs='+',
        required=True,
        help='Generated info pkl files, e.g. train/val/test pkl paths.')
    parser.add_argument(
        '--num-features',
        type=int,
        required=True,
        help='Radar point feature dimension. Use 7 for VoD and 8 for TJ4DRadSet.')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of multiprocessing workers.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files in training/depth_gt_radar.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    INFO_PATHS = args.info_paths
    radar_num_features = args.num_features
    output_dir = os.path.join(data_root, 'training', 'depth_gt_radar')
    overwrite = args.overwrite
    mmcv.mkdir_or_exist(output_dir)
    print("Begin to generate depth_gt, please wait for a moument")
    with Pool(args.workers) as po:
        results = []
        for info_path in INFO_PATHS:
            infos = mmcv.load(info_path)
            for info in infos:
                results.append(po.apply_async(func=worker_radar, args=(info, )))
        for result in results:
            result.get()
    print("Generation finish!!!")
