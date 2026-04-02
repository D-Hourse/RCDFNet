batch_size=4
voxel_size = [0.16, 0.16, 5.76]
voxel_size_cluster = [0.16, 0.16, 0.24]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2.76]
ori_image = (1216, 1936)
img_scale = (1280, 832)
img_scale_flip = (832, 1280)
dataset_type = 'VoDDataset_Seg'
data_root = 'xxx/VoD/radar_5frames'
class_names = ['Pedestrian', 'Cyclist', 'Car']

data_config = {
    'input_size': img_scale_flip,
    'src_size': ori_image,
    # Augmentation
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    # 'crop_h': (0.0, 0.0),
    # 'resize_test': 0.00,
    'resize': (0.00, 0.00),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
bda_aug_conf = dict(
    # rot_lim=(-22.5, 22.5),
    rot_lim=(0, 0),
    # scale_lim=(0.95, 1.05),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.00,
    # flip_dy_ratio=0.5,
    flip_dy_ratio=0
)

grid_config = {  #new
    'x': [0, 51.2, 0.32],
    'y': [-25.6, 25.6, 0.32],
    'z': [-3, 2.76, 5.76],
    'depth': [0, 51.2, 0.32],
}

numC_Trans=80

model = dict(
    type='BEVFusion_vod',
    image_size=[ori_image, img_scale_flip],
    freeze_img=True,
    freeze_radar=False,
    camera_stream=True,
    grid_size=(51.2, 51.2),
    grid_res=0.32,
    isfusion_img_index=1,
    se=True,
    voxel_size=voxel_size,
    return_mask=False,
    imc=80,
    radc=80,
    loss_depth_weight=0.2,
    # img
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    ViewTransform=dict(
        x_bound=grid_config['x'],
        y_bound=grid_config['y'],
        z_bound=grid_config['z'],
        d_bound=grid_config['depth'],
        final_dim=img_scale_flip,
        collapse_z=True,
        downsample_factor=4,
        output_channels=80,
        depth_net_conf=dict(in_channels=256,
                             mid_channels=256,
                             camera_aware=False),
        depth_channels=int((grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]),
        return_depth=True,
        return_mask=False,
        num_height=13,
        pc_range=point_cloud_range,
        gird_config=grid_config,
        dim=0,
    ),

    # pts
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2.76],
        voxel_size=[0.16, 0.16, 5.76],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.16, 0.16, 5.76],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2.76],
        legacy=False,
        with_velocity_snr_center=True),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 320]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    # pointfusion
    point_fusion=None,
    pts_output_channels=dict(
        in_channels=384,
        out_channels=80,
    ),

    # bev
    bev_cross_fusion=dict(
        type='concat'
    ),
    bev_fusion_mask_att=None,
    # img_bev_encoder_backbone=dict(
    #     type='CustomResNet',
    #     numC_input=numC_Trans,
    #     num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    # img_bev_encoder_neck=dict(
    #     type='FPN_LSS',
    #     in_channels=numC_Trans * 8 + numC_Trans * 2,
    #     out_channels=256),
    img_bev_encoder_backbone=None,
    img_bev_encoder_neck=None,

    # head
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -25.6, -0.6, 51.2, 25.6, -0.6],
                    [0, -25.6, -0.6, 51.2, 25.6, -0.6],
                    [0, -25.6, -1.78, 51.2, 25.6, -1.78]],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        assign_per_class=False,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1)
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=4096,
            max_num=500)))
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality = dict(use_lidar=True, use_camera=True)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(
        type='LoadDepthsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='PrepareImageInputsVODDebug',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepthVOD',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='PointToMultiViewDepthVoD', downsample=1, grid_config=grid_config),
    # dict(type='GlobalRotScaleTrans_point'),
    dict(type='LoadSegsFromFile', resize=None),
    dict(type='GetBEVMask', point_cloud_range=point_cloud_range, voxel_size=[0.32,0.32,5.76], downsample_ratio=1.),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='Resize',
    #     img_scale=(1280, 832),
    #     multiscale_mode='value',
    #     keep_ratio=True),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2.76]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2.76]),
    dict(type='PointShuffle'),
    # dict(
    #     type='Normalize',
    #     mean=[103.53, 116.28, 123.675],
    #     std=[1.0, 1.0, 1.0],
    #     to_rgb=False),
    # dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(
        type='Collect3D',
        keys=[
            'points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth',
            'seg_label', 'gt_bev_mask'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(
        type='PrepareImageInputsVODDebug',
        is_train=False,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepthVOD',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2.76]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5]),
    dict(type='LoadImageFromFile'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='VoDDataset_Seg',
            data_root=data_root,
            ann_file=
            '/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_train.pkl',
            split='training',
            pts_prefix='velodyne',
            pipeline=train_pipeline,
            modality=dict(use_lidar=True, use_camera=True),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='VoDDataset_Seg',
        data_root=data_root,
        ann_file='/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=True),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='VoDDataset_Seg',
        data_root=data_root,
        ann_file='/mnt/data1/chengpeifeng/VoD/radar_5frames/vod_infos_val.pkl',
        split='training',
        pts_prefix='velodyne',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=True),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR'))
lr = 0.0003
optimizer = dict(
    type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     step=[40],
# )
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05)
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(5, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.3
# )
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.3
)
# momentum_config = None
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1), ('val', 1)]
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=7,
            use_dim=[0, 1, 2, 3, 5]),
        dict(type='LoadImageFromFile'),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car'],
            with_label=False),
        dict(type='Collect3D', keys=['points', 'img'])
    ])
work_dir = './work_dirs/RCDBF_VoD_BMA_cross_attention_transformer'
resume_from = None
load_from = '/home/chengpeifeng/Documents/VoD_mmdet3d/checkpoints/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth'
# load_img_from = '/home/chengpeifeng/Documents/VoD_mmdet3d/checkpoints/HT_BEVconv_radardepth_ida_epoch_22.pth'
load_radar_from = '/home/chengpeifeng/Documents/VoD_mmdet3d/tools/work_dirs/vod-Radarpillarnet-11.16/checkpoints/epoch_57.pth'
gpu_ids = range(0, 1)
