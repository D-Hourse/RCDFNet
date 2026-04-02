# Copyright (c) OpenMMLab. All rights reserved.

"""Optional ops aggregation.

Some environments may not have all CUDA extensions built. Import failures are
tolerated at package import time. Missing symbols will still fail when called.
"""

__all__ = []

try:
    from mmcv.ops import (RoIAlign, SigmoidFocalLoss, get_compiler_version,
                          get_compiling_cuda_version, nms, roi_align,
                          sigmoid_focal_loss)
    __all__ += [
        'nms', 'RoIAlign', 'roi_align', 'get_compiler_version',
        'get_compiling_cuda_version', 'sigmoid_focal_loss', 'SigmoidFocalLoss'
    ]
except Exception:
    pass

try:
    from .ball_query import ball_query
    __all__.append('ball_query')
except Exception:
    pass

try:
    from .furthest_point_sample import (Points_Sampler, furthest_point_sample,
                                        furthest_point_sample_with_dist)
    __all__ += ['Points_Sampler', 'furthest_point_sample', 'furthest_point_sample_with_dist']
except Exception:
    pass

try:
    from .gather_points import gather_points
    __all__.append('gather_points')
except Exception:
    pass

try:
    from .group_points import (GroupAll, QueryAndGroup, group_points,
                               grouping_operation)
    __all__ += ['GroupAll', 'QueryAndGroup', 'group_points', 'grouping_operation']
except Exception:
    pass

try:
    from .interpolate import three_interpolate, three_nn
    __all__ += ['three_interpolate', 'three_nn']
except Exception:
    pass

try:
    from .knn import knn
    __all__.append('knn')
except Exception:
    pass

from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d
__all__ += ['NaiveSyncBatchNorm1d', 'NaiveSyncBatchNorm2d']

try:
    from .paconv import PAConv, PAConvCUDA, assign_score_withk
    __all__ += ['PAConv', 'PAConvCUDA', 'assign_score_withk']
except Exception:
    pass

try:
    from .pointnet_modules import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
                                   PAConvSAModule, PAConvSAModuleMSG,
                                   PointFPModule, PointSAModule,
                                   PointSAModuleMSG, build_sa_module)
    __all__ += [
        'PAConvCUDASAModule', 'PAConvCUDASAModuleMSG', 'PAConvSAModule',
        'PAConvSAModuleMSG', 'PointFPModule', 'PointSAModule',
        'PointSAModuleMSG', 'build_sa_module'
    ]
except Exception:
    pass

try:
    from .roiaware_pool3d import (RoIAwarePool3d, points_in_boxes_batch,
                                  points_in_boxes_cpu, points_in_boxes_gpu)
    __all__ += ['RoIAwarePool3d', 'points_in_boxes_batch', 'points_in_boxes_cpu', 'points_in_boxes_gpu']
except Exception:
    pass

try:
    from .sparse_block import (SparseBasicBlock, SparseBottleneck,
                               make_sparse_convmodule)
    __all__ += ['SparseBasicBlock', 'SparseBottleneck', 'make_sparse_convmodule']
except Exception:
    pass

try:
    from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
    __all__ += ['DynamicScatter', 'Voxelization', 'dynamic_scatter', 'voxelization']
except Exception:
    pass


def _missing_op(*args, **kwargs):
    raise ImportError(
        'Required CUDA op is not built in current environment. '
        'Please compile corresponding rcdfnet.ops extension or use RCFusion env.'
    )


for _name in [
    'points_in_boxes_batch', 'points_in_boxes_cpu', 'points_in_boxes_gpu',
    'ball_query', 'knn', 'gather_points', 'group_points', 'grouping_operation',
    'furthest_point_sample', 'furthest_point_sample_with_dist',
    'three_interpolate', 'three_nn', 'dynamic_scatter', 'voxelization',
    'Voxelization', 'DynamicScatter',
    'build_sa_module', 'PointSAModule', 'PointSAModuleMSG', 'PointFPModule',
    'PAConv', 'PAConvCUDA', 'assign_score_withk', 'Points_Sampler',
    'PAConvCUDASAModule', 'PAConvCUDASAModuleMSG', 'PAConvSAModule',
    'PAConvSAModuleMSG'
]:
    if _name not in globals():
        globals()[_name] = _missing_op
