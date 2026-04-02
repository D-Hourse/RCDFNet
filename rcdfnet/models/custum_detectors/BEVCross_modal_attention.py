import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

class Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3, radc=384, imc=256):
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
                radc,
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
    def __init__(self, kernel_size=3):
        super(Cross_Modal_Fusion_test, self).__init__()

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
                384+384,
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
        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fusion_BEV = torch.cat([img_bev,radar_bev],dim=1)
        fusion_BEV = self.reduce_mixBEV(fusion_BEV)
        return fusion_BEV

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

class Transformer_cross_attention(nn.Module):
    def __init__(self,dim=256, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv_img = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_radar = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_img = nn.Linear(dim, dim // 2)
        self.proj_radar = nn.Linear(dim, dim//2)

    def forward(self, img_bev, radar_bev):
        B, C, W, H = img_bev.shape
        img_bev = img_bev.permute(0, 2, 3, 1).reshape(B,W*H,C) #(B,W*H,C)
        radar_bev = radar_bev.permute(0, 2, 3, 1).reshape(B,W*H,C) #(B,W*H,C)
        assert img_bev.shape == radar_bev.shape
        qkv_img = self.qkv_img(img_bev).reshape(B, W*H, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q_img, k_img, v_img = qkv_img[0], qkv_img[1], qkv_img[2]   #(B,h, W*H,c/h)
        qkv_radar = self.qkv_radar(radar_bev).reshape(B, W*H, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q_radar, k_radar, v_radar = qkv_radar[0], qkv_radar[1], qkv_radar[2] #(B,h, W*H,c/h)
        q_img = q_img * self.scale
        q_radar = q_radar * self.scale
        attn_img = (q_img @ k_radar.transpose(-2,-1))   #(B,h,W*H,W*H)
        attn_radar = (q_radar @ k_img.transpose(-2,-1))  #(B,h,W*H,W*H)
        attn_img = self.softmax(attn_img)
        attn_radar = self.softmax(attn_radar)
        img_bev = (attn_radar @ v_img).transpose(1,2).reshape(B, W*H, C)
        img_bev = self.proj_img(img_bev).reshape(B, W, H, C//2).permute(0, 3, 1, 2) #(B,C/2,W,H)
        radar_bev = (attn_img @ v_radar).transpose(1, 2).reshape(B, W * H, C)
        radar_bev = self.proj_radar(radar_bev).reshape(B, W, H, C//2).permute(0, 3, 1, 2)
        output = torch.cat([img_bev,radar_bev],dim=1)
        return output

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