import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Set
import numpy as np
from functools import partial
from rcdfnet.models.sparse_det.hed_utils import post_act_block_sparse_3d
from rcdfnet.models.sparse_det.hed_utils import post_act_block_sparse_2d
from rcdfnet.models.sparse_det.hed_utils import post_act_block_dense_2d
from rcdfnet.models.sparse_det.hed_utils import SparseBasicBlock3D, SparseBasicBlock3D_RS, SparseBasicBlock3D_SSR
from rcdfnet.models.sparse_det.hed_utils import SparseBasicBlock2D_RS, SparseBasicBlock2D_SSR
from rcdfnet.models.sparse_det.hed_utils import BasicBlock
from rcdfnet.ops.spconv import *

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)



def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

# def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
#     """
#     Finds all spconv keys that need to have weight's transposed
#     """
#     found_keys: Set[str] = set()
#     for name, child in model.named_children():
#         new_prefix = f"{prefix}.{name}" if prefix != "" else name
#
#         if isinstance(child, conv.SparseConvolution):
#             new_prefix = f"{new_prefix}.weight"
#             found_keys.add(new_prefix)
#
#         found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))
#      return found_keys

class Point_Cluster_Conv(SparseModule):
    '''
        Point_Cluster_Conv
    '''
    def __init__(self, in_dim:int, dim: list,  numSB: list, down_stride: list, xy_only=False):
        super().__init__()

        block_SSR = SparseBasicBlock2D_SSR if xy_only else SparseBasicBlock3D_SSR
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList(
            [SparseSequential(*post_act_block_sparse_3d(in_dim, dim[0], 3, conv_type='subm', indice_key='subm_block_1'))]
        )
        self.down_stride_num = len(down_stride)
        self.numSB = numSB
        for idx in range(self.down_stride_num):
            cur_layers = [
                post_act_block(
                    dim[idx], dim[idx+1], 3, down_stride[idx], 1,
                    conv_type='spconv', indice_key=f'spconv_block_{idx}'),
                *[block_SSR(dim[idx+1], indice_key=f"spconv_block_{idx}_{i}") for i in range(self.numSB[idx])]
            ]
            self.encoder.append(SparseSequential(*cur_layers))


    def forward(self, x):
        # for conv in self.encoder:
        #     x = conv(x)
        features = []
        i = 0
        for conv in self.encoder:
            x = conv(x)
            if(i==0):
                i = i+1
                continue
            i = i+1
            features.append(x.dense())
        return features


class BEVSparse2(SparseModule):
    '''
        BEVSparse2-  use two img-scales
    '''
    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, indice_key, xy_only=False):
        super().__init__()

        block_RS = SparseBasicBlock2D_RS if xy_only else SparseBasicBlock3D
        block_SSR = SparseBasicBlock2D_SSR if xy_only else SparseBasicBlock3D
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder1 = nn.ModuleList(
            [SparseSequential(
                *[block_RS(dim, indice_key=f"{indice_key}_0") for _ in range(num_SBB[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [
                post_act_block(
                    dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),

                *[block_SSR(dim, indice_key=f"{indice_key}_{idx}") for _ in range(num_SBB[idx])]
            ]
            self.encoder.append(SparseSequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    dim, dim, down_kernel_size[idx],
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x1, x2):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x



if __name__ == '__main__':
    pts_cluster_spconv = dict(
        in_dim=1,
        dim=[16, 32, 64],
        numSB=[2, 2],
        down_stride=[1, 1],
        xy_only=False
    )
    model = Point_Cluster_Conv(**pts_cluster_spconv).cuda()
    # 创建一个维度为 [2, 256, 160, 160] 的稀疏张量

    features = torch.rand((100, 1)).cuda()  # Example features tensor
    indices = torch.randint(0, 160, (100, 4))  # Example indices tensor
    indices[:50, 0] = torch.randint(0, 1, (50,))
    indices[50:, 0] = torch.randint(1, 2, (50,))
    indices[:50, 3] = torch.randint(0, 1, (50,))
    indices[50:, 3] = torch.randint(1, 2, (50,))
    indices = indices.type(torch.int32).cuda()
    spatial_shape = [160, 160, 2]  # Example spatial shape
    batch_size = 2  # Example batch size

    # Create SparseConvTensor
    sparse_input = SparseConvTensor(features, indices, spatial_shape, batch_size)

    output = model(sparse_input)

    print("input:", sparse_input.dense().shape)
    print("output:", output.dense().shape)