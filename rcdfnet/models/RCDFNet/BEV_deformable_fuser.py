import torch
from torch import nn

from mmcv.runner.base_module import ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding
from mmcv.runner import auto_fp16

from .multimodal_deformable_cross_attention import DeformableCrossAttention, DeformableCrossAttention_Multi


class BEVFuser(nn.Module):
    def __init__(self, bev1_dims=80, bev2_dims=128, embed_dims=256,
                 num_layers=6, num_heads=4, bev_shape=(128, 128)):
        super(BEVFuser, self).__init__()

        self.num_modalities = 2
        self.use_cams_embeds = False
        self.num_heads = num_heads      # num_heads=4

        self.bev1_dims = bev1_dims        # img_dim=80
        self.bev2_dims = bev2_dims        # pts_dim=80
        self.embed_dims = embed_dims    # embed_dim=128
        _pos_dim_ = self.embed_dims//2  # pos_dim=64
        _ffn_dim_ = self.embed_dims*2   # ffn_dim=256

        self.norm_img = build_norm_layer(dict(type='LN'), bev1_dims)[1]      # LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        self.norm_pts = build_norm_layer(dict(type='LN'), bev2_dims)[1]      # LayerNorm((80,), eps=1e-05, elementwise_affine=True)
        self.input_proj = nn.Linear(bev1_dims + bev2_dims, self.embed_dims)   # input_proj-Linear(in_features=160, out_features=128, bias=True)

        self.bev_h, self.bev_w = bev_shape                                  # bev_w-128, bev_h-128

        self.positional_encoding = build_positional_encoding(               # LearnedPositionalEncoding(num_feats=64, row_num_embed=128, col_num_embed=128)
            dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,                                        # the times of 2
                row_num_embed=self.bev_h,
                col_num_embed=self.bev_w,
            ),
        )
        self.register_buffer('ref_2d', self.get_reference_points(self.bev_h, self.bev_w))

        ffn_cfgs = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=_ffn_dim_,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        norm_cfgs = dict(type='LN')

        self.ffn_layers = ModuleList()
        for _ in range(num_layers):                 # num_layers=6
            self.ffn_layers.append(
                build_feedforward_network(ffn_cfgs)
            )
        self.norm_layers1 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers1.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.norm_layers2 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers2.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.attn_layers = ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                DeformableCrossAttention(
                    img_dims=self.bev1_dims,
                    pts_dims=self.bev2_dims,
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    num_modalities=self.num_modalities,
                    num_points=4
                ),
            )

        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableCrossAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @staticmethod
    def get_reference_points(H, W, dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2). (bs-batch_size, num_keys-number of reference point, num_levels-? 2-x and y coordinate)
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.unsqueeze(2).unsqueeze(3)
        return ref_2d

    @auto_fp16(apply_to=('feat_bev1', 'feat_bev2'))
    def forward(self, feat_bev1, feat_bev2):

        bs = feat_bev1.shape[0]              # feat_img-tensor(1,80,128,128)
        ref_2d_stack = self.ref_2d.repeat(bs, 1, 1, self.num_modalities, 1)                                     # ?

        feat_bev1 = self.norm_img(feat_bev1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()       # tensor(1,80,128,128)
        feat_bev2 = self.norm_pts(feat_bev2.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()       # tensor(1,80,128,128)

        feat_flatten = []
        spatial_shapes = []
        for feat in [feat_bev1, feat_bev2]:
            _, _, h, w = feat.shape
            spatial_shape = (h, w)      # spatial_shape==(128, 128)
            feat = feat.flatten(2).permute(0, 2, 1).contiguous()  # feat-tensor(1,16384,80)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(                               # spatial_shapes-list[(128,128),(128,128)]
            spatial_shapes, dtype=torch.long, device=feat_bev1.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(        # level_start_index-tensor[0,16384]
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        bev_queries = torch.cat(feat_flatten, -1)           # feat_flatten-list[2] concatenate img_feat and radar feat  bev_queries-tensor(1,16384,160)
        bev_queries = self.input_proj(bev_queries)          # bev_queries-tensor(1,16384,128)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),            # bev_mask-tensor(1,128,128)
                               device=bev_queries.device).to(feat_bev1.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(feat_bev1.dtype) # bev_pos-tensor(1,128,128,128)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()      # bev_pos-tensor(1,16348,128)

        feat_bev1 = feat_flatten[0]
        feat_bev2 = feat_flatten[1]


        for attn_layer, ffn_layer, norm_layer1, norm_layer2 in \
            zip(self.attn_layers, self.ffn_layers, self.norm_layers1, self.norm_layers2):
            # post norm
            bev_queries = attn_layer(
                bev_queries,
                feat_bev1,
                feat_bev2,
                identity=None,
                query_pos=bev_pos,
                reference_points=ref_2d_stack,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                )
            bev_queries = norm_layer1(bev_queries)                  # bev_queries-tensor(1,16384,128)
            bev_queries = ffn_layer(bev_queries, identity=None)     # bev_queries-tensor(1,16384,128)
            bev_queries = norm_layer2(bev_queries)                  # bev_queries-tensor(1,16384,128)


        output = bev_queries.permute(0, 2, 1).contiguous().reshape(bs, self.embed_dims, h, w)
        return output


class BEVMulti_Fuser(nn.Module):
    def __init__(self, bev_dims=256, embed_dims=256, n_levels=5,
                 num_layers=6, num_heads=4, bev_shape=(160, 160)):
        super(BEVMulti_Fuser, self).__init__()

        self.num_heads = num_heads      # num_heads=4
        self.n_levels = n_levels
        self.bev_dims = bev_dims        # img_dim=256
        self.embed_dims = embed_dims    # embed_dim=128
        _pos_dim_ = self.embed_dims//2  # pos_dim=64
        _ffn_dim_ = self.embed_dims*2   # ffn_dim=256

        self.norm1 = build_norm_layer(dict(type='LN'), bev_dims)[1]      # LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.input_proj = nn.Linear(bev_dims * self.n_levels, self.embed_dims)           # input_proj-Linear(in_features=256, out_features=128, bias=True)
        self.norm_layer0 = ModuleList()
        for _ in range(self.n_levels):
            self.norm_layer0.append(build_norm_layer(dict(type='LN'), bev_dims)[1])

        self.bev_h, self.bev_w = bev_shape                                  # bev_w-128, bev_h-128

        self.positional_encoding = build_positional_encoding(               # LearnedPositionalEncoding(num_feats=64, row_num_embed=128, col_num_embed=128)
            dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,                                        # the times of 2
                row_num_embed=self.bev_h,
                col_num_embed=self.bev_w,
            ),
        )
        self.register_buffer('ref_2d', self.get_reference_points(self.bev_h, self.bev_w))

        ffn_cfgs = dict(
            type='FFN',
            embed_dims=self.embed_dims,
            feedforward_channels=_ffn_dim_,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
        )
        norm_cfgs = dict(type='LN')

        self.ffn_layers = ModuleList()
        for _ in range(num_layers):                 # num_layers=6
            self.ffn_layers.append(
                build_feedforward_network(ffn_cfgs)
            )
        self.norm_layers1 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers1.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.norm_layers2 = ModuleList()
        for _ in range(num_layers):
            self.norm_layers2.append(
                build_norm_layer(norm_cfgs, self.embed_dims)[1],
            )
        self.attn_layers = ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                DeformableCrossAttention_Multi(
                    img_dims=self.bev_dims,
                    embed_dims=self.embed_dims,
                    num_heads=self.num_heads,
                    num_levels=self.n_levels,
                    num_points=4
                ),
            )

        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableCrossAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @staticmethod
    def get_reference_points(H, W, dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2). (bs-batch_size, num_keys-number of reference point, num_levels-? 2-x and y coordinate)
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.unsqueeze(2).unsqueeze(3)
        return ref_2d

    @auto_fp16(apply_to=('feat_bev'))
    def forward(self, feat_bev):

        bs = feat_bev[0].shape[0]              # feat_img-tensor(1,80,128,128)
        ref_2d_stack = self.ref_2d.repeat(bs, 1, 1, self.n_levels, 1)                                     # ?

        # feat_bev1 = self.norm_img(feat_bev1.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()       # tensor(1,80,128,128)
        # feat_bev2 = self.norm_pts(feat_bev2.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()       # tensor(1,80,128,128)

        for i in range(len(feat_bev)):
            feat_bev[i] = self.norm_layer0[i](feat_bev[i].permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

        # feat_bev_list = []
        # for l in range(self.n_levels):
        #     feat_bev = self.input_proj(feat_bev[l])
        #     feat_bev_list.append(feat_bev)

        feat_flatten = []
        spatial_shapes = []
        for feat in feat_bev:
            _, _, h, w = feat.shape
            spatial_shape = (h, w)      # spatial_shape==(128, 128)
            feat = feat.flatten(2).permute(0, 2, 1).contiguous()  # feat-tensor(1,16384,80)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        spatial_shapes = torch.as_tensor(                               # spatial_shapes-list[(128,128),(128,128)]
            spatial_shapes, dtype=torch.long, device=feat_bev[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(        # level_start_index-tensor[0,16384]
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        bev_queries = torch.cat(feat_flatten, dim=-1)           # feat_flatten-list[2] concatenate img_feat and radar feat  bev_queries-tensor(1,16384,160)
        bev_queries = self.input_proj(bev_queries)          # bev_queries-tensor(1,16384,128)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),            # bev_mask-tensor(1,128,128)
                               device=bev_queries.device).to(feat_bev[0].dtype)
        bev_pos = self.positional_encoding(bev_mask).to(feat_bev[0].dtype) # bev_pos-tensor(1,128,128,128)
        bev_pos = bev_pos.flatten(2).permute(0, 2, 1).contiguous()      # bev_pos-tensor(1,16348,128)

        for attn_layer, ffn_layer, norm_layer1, norm_layer2 in \
            zip(self.attn_layers, self.ffn_layers, self.norm_layers1, self.norm_layers2):
            # post norm
            bev_queries = attn_layer(
                bev_queries,
                feat_flatten,
                identity=None,
                query_pos=bev_pos,
                reference_points=ref_2d_stack,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                )
            bev_queries = norm_layer1(bev_queries)                  # bev_queries-tensor(1,16384,128)
            bev_queries = ffn_layer(bev_queries, identity=None)     # bev_queries-tensor(1,16384,128)
            bev_queries = norm_layer2(bev_queries)                  # bev_queries-tensor(1,16384,128)


        output = bev_queries.permute(0, 2, 1).contiguous().reshape(bs, self.embed_dims, h, w)
        return output

