import os
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) .vod.frame
from transformations import project_pcl_to_image
import pdb

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


####------------------------------VoD----------------------------------------------
# data_root = '/mnt/data1/chengpeifeng/VoD'
# INFO_PATHS = ['/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_train.pkl',
#               '/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_val.pkl',
#               '/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_test.pkl']
# #
# # INFO_PATHS = ['/mnt/data1/chengpeifeng/VoD/lidar/vod_infos_train.pkl',
# #               '/mnt/data1/chengpeifeng/VoD/lidar/vod_infos_val.pkl',
# #               '/mnt/data1/chengpeifeng/VoD/lidar/vod_infos_test.pkl']
#
# lidar_key = 'LIDAR_TOP'

####-----------------------------TJ4D--------------------------------------------------
data_root = '/mnt/data1/chengpeifeng/tj4d/datasets/'
INFO_PATHS = ['/mnt/data1/chengpeifeng/tj4d/datasets/TJ4DRadSet_4DRadar_infos_train.pkl',
              '/mnt/data1/chengpeifeng/tj4d/datasets/TJ4DRadSet_4DRadar_infos_val.pkl',
              '/mnt/data1/chengpeifeng/tj4d/datasets/TJ4DRadSet_4DRadar_infos_test.pkl']
lidar_key = 'LIDAR_TOP'

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
    # radar_path = data_root + '/radar_5frames/' + info['point_cloud']['velodyne_path']
    radar_path = data_root + info['point_cloud']['velodyne_path']
    ####--------------------vod---------------------------
    # points = np.fromfile(os.path.join(radar_path),
    #                      dtype=np.float32,
    #                      count=-1).reshape(-1, 7)
    # points = points[:, :4]
    ####--------------------tj4d---------------------------
    points = np.fromfile(os.path.join(radar_path),
                         dtype=np.float32,
                         count=-1).reshape(-1, 8)
    a = points[:, :3]
    b = points[:, 5]
    b = np.expand_dims(b, axis=1)
    points = np.concatenate((a, b), axis=1)

    radar_to_cam_calib = info['calib']['Tr_velo_to_cam']
    cam_intrinsic = info['calib']['P2']
    image_shape = info['image']['image_shape']
    uvs, point_depth = project_pcl_to_image(points, radar_to_cam_calib, cam_intrinsic, image_shape)
    unique_uvs, indices = np.unique(uvs, axis=0, return_index=True)
    unique_point_depth = point_depth[indices]
    image_name = os.path.basename(info['image']['image_path'])
    if image_name == '100095.png':
        print("end")
        debug=True

    if debug:
        map_pointcloud_to_image2(info['image'], unique_uvs, unique_point_depth)

    np.concatenate([unique_uvs, unique_point_depth[:, None]], axis=1).astype(np.float32).flatten().tofile(os.path.join(data_root, 'depth_gt_radar', f'{image_name}.bin'))


if __name__ == '__main__':
    po = Pool(1)
    mmcv.mkdir_or_exist(os.path.join(data_root, 'depth_gt_radar'))
    print("Begin to generate depth_gt, please wait for a moument")
    for info_path in INFO_PATHS:
        infos = mmcv.load(info_path)
        for info in infos:
            po.apply_async(func=worker_radar, args=(info, ))
    po.close()
    po.join()
    print("Generation finish!!!")
