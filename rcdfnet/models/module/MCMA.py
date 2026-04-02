import torch
from torch import nn
from torch.nn import functional as F

"""
图像特征融合模块
它通过采用跨模态注意力机制，帮助深度学习模型关注重要特征.
MCAM模块实现了SAR(合成孔径雷达)和Optical图像(通过光学传感器捕获的图像)特征之间的有效融合，利用跨模态注意力机制提取和利用两种模态之间的互补信息，这对于提高多模态数据处理任务的性能至关重要。
这种机制不仅增强了模型对特征的表达能力，还增强了模型在复杂环境下的适应性和鲁棒性。
"""

class MCAM(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,    # 子采样
                 bn_layer=True):     # 批量归一化
        super(MCAM, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g_sar = conv_nd(in_channels=self.in_channels,out_channels=self.inter_channels,kernel_size=1,stride=1,padding=0)

        self.g_opt = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta_sar = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.theta_opt = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi_sar = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.phi_opt = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        if sub_sample:
            self.g_sar = nn.Sequential(self.g_sar, max_pool_layer)
            self.g_opt = nn.Sequential(self.g_opt, max_pool_layer)
            self.phi_sar = nn.Sequential(self.phi_sar, max_pool_layer)
            self.phi_opt = nn.Sequential(self.phi_opt, max_pool_layer)

    def forward(self, sar, opt):

        batch_size = sar.size(0)

        g_x = self.g_sar(sar).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta_sar(sar).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi_sar(sar).view(batch_size, self.inter_channels, -1)

        f_x = torch.matmul(theta_x, phi_x)
        f_div_C_x = F.softmax(f_x, dim=-1)

        g_y = self.g_opt(opt).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_y = g_y.permute(0, 2, 1)

        theta_y = self.theta_opt(opt).view(batch_size, self.inter_channels, -1)
        theta_y = theta_y.permute(0, 2, 1)

        phi_y = self.phi_opt(opt).view(batch_size, self.inter_channels, -1)

        f_y = torch.matmul(theta_y, phi_y)
        f_div_C_y = F.softmax(f_y, dim=-1)
        y = torch.einsum('ijk,ijk->ijk', [f_div_C_x, f_div_C_y])
        y_x = torch.matmul(y, g_x)
        y_y = torch.matmul(y, g_y)
        y = y_x * y_y
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *sar.size()[2:])
        y = self.W(y)
        return y

if __name__ == '__main__':
    model = MCAM(in_channels=256).cuda()
    sar = torch.randn(2, 256, 64, 64).cuda()
    opt = torch.randn(2, 256, 64, 64).cuda()
    print("input:", sar.shape, opt.shape)
    print("output:", model(sar, opt).shape)