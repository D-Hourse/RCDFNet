import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

class Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3, radc=80, imc=256, fusion=80):
        super(Cross_Modal_Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                imc+radc,
                fusion,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev, radar_bev], dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV

class Cross_Modal_Fusion_test(nn.Module):
    def __init__(self, kernel_size=3, kernel_spatial=7, radc=80, imc=256, fusion=80):
        super(Cross_Modal_Fusion_test, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                radc+imc,
                fusion,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

        self.spatial = nn.Conv2d(2, 1, kernel_spatial, stride=1, padding=(kernel_spatial - 1) // 2)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_mix_out = img_avg_out + img_max_out
        img_avg_max = torch.cat([img_avg_out, img_max_out, img_mix_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_mix_out = radar_avg_out + radar_max_out
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out, radar_mix_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev,radar_bev],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        fusion_compress = torch.cat((torch.max(fusion_BEV, 1)[0].unsqueeze(1), torch.mean(fusion_BEV, 1).unsqueeze(1)), dim=1)
        fusion_out = self.spatial(fusion_compress)
        scale = torch.sigmoid(fusion_out)
        return fusion_BEV * scale

class Not_Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3):
        super(Not_Cross_Modal_Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                256+384,
                384,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev = img_bev * img_att
        radar_bev = radar_bev * radar_att
        fusion_BEV = torch.cat([img_bev,radar_bev],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV

class Cross_Modal_Fusion_spatial_original(nn.Module):  ##spatial+original
    def __init__(self, kernel_size=3):
        super(Cross_Modal_Fusion_spatial_original, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                256*2+384*2,
                384,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)

        img_bev2 = img_bev * radar_att

        radar_bev2 = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev,img_bev2,radar_bev,radar_bev2],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV


class Cross_Modal_Fusion_spatial_concatoriginal(nn.Module):  ##spatial+original
    def __init__(self, kernel_size=3):
        super(Cross_Modal_Fusion_spatial_concatoriginal, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                256*2+384*2,
                384,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)

        img_bev2 = img_bev * radar_att

        radar_bev2 = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev,img_bev2,radar_bev,radar_bev2],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV



class Cross_Modal_Fusion_spatial_concat(nn.Module):
    def __init__(self, kernel_size=3):
        super(Cross_Modal_Fusion_spatial_concat, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                256*2+384*2,
                384,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev1 = img_bev * img_att
        img_bev2 = img_bev * radar_att
        radar_bev1 = radar_bev * radar_att
        radar_bev2 = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev1,img_bev2,radar_bev1,radar_bev2],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV

class Cross_Modal_Fusion_spatial_channel_concat(nn.Module):
    def __init__(self, kernel_size=3):
        super(Cross_Modal_Fusion, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        self.reduce_mixBEV = ConvModule(
                256*2+384*2,
                384,
                1,
                padding=0,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(384, 384//16, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384//16, 384, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, img_bev, radar_bev):
        img_avg_out = torch.mean(img_bev, dim=1, keepdim=True)
        img_max_out, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_avg_max = torch.cat([img_avg_out, img_max_out], dim=1)
        img_att = self.att_img(img_avg_max)
        radar_avg_out = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max_out, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_avg_max = torch.cat([radar_avg_out, radar_max_out], dim=1)
        radar_att = self.att_radar(radar_avg_max)
        img_bev1 = img_bev * img_att
        img_bev2 = img_bev * radar_att
        radar_bev1 = radar_bev * radar_att
        radar_bev2 = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev1,img_bev2,radar_bev1,radar_bev2],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        avg_out = self.shared_MLP(self.avg_pool(fusion_BEV))
        max_out = self.shared_MLP(self.max_pool(fusion_BEV))
        channel_att = self.sigmoid(avg_out + max_out)
        fusion_BEV = fusion_BEV * channel_att
        return fusion_BEV



class Cross_Modal_FusionV3(nn.Module):
    def __init__(self, img_dim=256, radar_dim=384):
        super(Cross_Modal_Fusion, self).__init__()
        self.mixconv = nn.Conv2d(img_dim+radar_dim,radar_dim,kernel_size=1, stride=1)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img_dim+2*radar_dim, img_dim+2*radar_dim, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.bev_encoder = nn.Conv2d(img_dim+2*radar_dim,radar_dim,kernel_size=1, stride=1)
    def forward(self, img_bev, radar_bev):
        mix_bev = self.mixconv(torch.cat([img_bev,radar_bev],dim=1))

        final_bev = torch.cat([img_bev,mix_bev,radar_bev],dim=1)
        final_bev = final_bev * self.att(final_bev)
        final_bev = self.bev_encoder(final_bev)
        return final_bev
class Transformer_cross_attention(nn.Module):        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/crossvit.py
    def __init__(
            self,
            input_dim=80,
            output_dim=256,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.wk = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.wv = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.conv = nn.Conv1d(in_channels=1, out_channels=784, kernel_size=1)  # 输入通道为1，输出通道为784，核大小为1         Add it myself
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, output_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, img_bev, radar_bev):
        B, C, W, H = img_bev.shape
        img_bev = img_bev.permute(0, 2, 3, 1).reshape(B, W * H, C)  # (B,W*H,C)
        radar_bev = radar_bev.permute(0, 2, 3, 1).reshape(B, W * H, C)  # (B,W*H,C)
        assert img_bev.shape == radar_bev.shape


        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(radar_bev[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(radar_bev).reshape(B, W*H, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(img_bev).reshape(B, W*H, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)



        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)



        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C

        # x = self.conv(x)           # Add it myself

        x = self.proj(x)

        x = self.proj_drop(x)

        return x

class Cross_Modal_FusionV2(nn.Module):
    def __init__(self, dim=256):
        super(Cross_Modal_Fusion, self).__init__()
        self.att_img = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.att_radar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.bev_encoder = nn.Conv2d(dim,dim,kernel_size=1, stride=1)
    def forward(self, img_bev, radar_bev):
        atten_img_bev = img_bev * self.att_radar(radar_bev)
        atten_radar_bev = radar_bev * self.att_img(img_bev)
        #final_bev = img_bev + radar_bev + atten_img_bev + atten_radar_bev
        final_bev = atten_img_bev + atten_radar_bev
        final_bev = self.bev_encoder(final_bev)
        return final_bev

if __name__=="__main__":
    img_bev = torch.randn((2, 256, 124, 108))
    radar_bev = torch.randn((2, 384, 124, 108))
    cross_attention = Cross_Modal_Fusion(kernel_size=3)
    out = cross_attention(img_bev,radar_bev)
    print(out.shape)