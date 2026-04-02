import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Set
import numpy as np
from functools import partial
from rcdfnet.models.sparse_det.hed_utils import post_act_block_sparse_3d
from rcdfnet.models.sparse_det.hed_utils import post_act_block_sparse_2d
from rcdfnet.models.sparse_det.hed_utils import post_act_block_dense_2d
from rcdfnet.models.sparse_det.hed_utils import SparseBasicBlock3D
from rcdfnet.models.sparse_det.hed_utils import SparseBasicBlock2D_RS, SparseBasicBlock2D_SSR
from rcdfnet.models.sparse_det.hed_utils import BasicBlock
from rcdfnet.ops.spconv import *

dense_tensor = torch.randn(4, 256, 160, 160)
nonzero_indices = torch.nonzero(dense_tensor, as_tuple=False)
nonzero_features = dense_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2], nonzero_indices[:, 3]]

indices = nonzero_indices.cpu().numpy()
features = nonzero_features.cpu().numpy()

spatial_shape = [160, 160]

batch_size = 4

sparse_tensor = SparseConvTensor(features, indices, spatial_shape, batch_size)