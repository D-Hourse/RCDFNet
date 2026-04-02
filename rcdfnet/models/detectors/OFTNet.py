import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-6

def perspective(matrix, vector):   ##这个函数写一写,用lidar2img直接到像素坐标系
    """
    Applies perspective projection to a vector using projection matrix
    """
    vector = vector.unsqueeze(-1)

    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]

#--------y不变，X和Z离散，起点边缘--------------
def make_grid(grid_size, grid_offset, grid_res, grid_z_min, grid_z_max):  #定义在雷达坐标系下,返回3D grid
    """   （69.12，79.36）    （0，-39.68, 0），0.32,   -3,    2.76
    self.grid = utils.make_grid(grid_size, (-grid_size[0]/2., y_offset, 0.), grid_res)
    make_grid((80,80),(-40,1.74,0),0.5)
    Constructs an array representing the corners of an orthographic grid
    """
    depth, width = grid_size
    xoff, yoff, zoff = grid_offset

    xcoords = torch.range(0., depth, grid_res) + xoff  #216+1
    ycoords = torch.range(width, 0, -grid_res) + yoff  #248+1

    xx, yy = torch.meshgrid(xcoords, ycoords)
    z_corners = torch.range(grid_z_max, grid_z_min, -grid_res)  ##[-3,2.76] 18个数+1
    z_corners = F.pad(z_corners.view(-1, 1, 1, 1), [2, 0])  ### 最内层常数0填充左边填充两个
    xxyy = torch.stack([xx, yy, torch.full_like(xx, zoff)], dim=-1).unsqueeze(0)
    xxyyzz = xxyy + z_corners
    corners = xxyyzz.unsqueeze(0)
    return corners

class Attention_Block(nn.Module):
    def __init__(self, c):
        super(Attention_Block, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c,c,kernel_size=1,stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

class OFT(nn.Module):

    def __init__(self, channels, input_yxchannels, scale=1):  ##第二个参数为y方向格子Xchannels，即18*256，为了折叠y轴
        super().__init__()



        # self.conv3d = nn.Conv2d((len(y_corners)-1) * channels, channels,1)
        self.conv3d = nn.Linear(input_yxchannels, channels)  ##此处要根据实际改
        self.scale = scale

    def forward(self, features, calib, corners_radar):   ##corners_radar(1,18,216,248,3)
        ##grid  corners  norm_corners  bbox_corners都没有梯度
        # Expand the grid in the y dimension
        assert features.shape[0] == calib.shape[0]
        B = calib.shape[0]
        corners = corners_radar.repeat(B, 1, 1, 1, 1)  ##[b, 18, 216, 248, 3]

        # Project grid corners to image plane and normalize to [-1, 1]，calib（b,1,1,1,3,4）
        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)  # b,18,216,248,2）

        # Normalize to [-1, 1]
        img_height, img_width = features.size()[2:]  ## 特征图大小
        img_size = corners.new([img_width, img_height]) / self.scale  ##原图大小，hw换过来
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)  ##中点是0左右-1

        # Get top-left and bottom-right coordinates of voxel bounding boxes
        bbox_corners = torch.cat([
            torch.min(norm_corners[:, :-1, :-1, :-1],  # 这里索引的是前四维，先取小uv再取大
                      norm_corners[:, :-1, 1:, :-1]),
            torch.max(norm_corners[:, 1:, 1:, 1:],
                      norm_corners[:, 1:, :-1, 1:])
        ], dim=-1)
        batch, _, depth, width, _ = bbox_corners.size()  ##（B,7,159,159,4）
        bbox_corners = bbox_corners.flatten(2, 3)  ###按层展开（B,7,25281,4）

        # Compute the area of each bounding box
        area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) \
                * img_height * img_width * 0.25 + EPSILON).unsqueeze(1)
        visible = (area > EPSILON)  ## 此处投影在图像外面的都会为epsilon，即内部乘积是0

        # Sample integral image at bounding box locations
        integral_img = integral_image(features)
        top_left = F.grid_sample(integral_img, bbox_corners[..., [0, 1]],align_corners=False)
        btm_right = F.grid_sample(integral_img, bbox_corners[..., [2, 3]],align_corners=False)
        top_right = F.grid_sample(integral_img, bbox_corners[..., [2, 1]],align_corners=False)
        btm_left = F.grid_sample(integral_img, bbox_corners[..., [0, 3]],align_corners=False)

        # Compute voxel features (ignore features which are not visible)
        vox_feats = (top_left + btm_right - top_right - btm_left) / area
        vox_feats = vox_feats * visible.float()
        # vox_feats = vox_feats.view(batch, -1, depth, width) # （b,256,7,25281）
        vox_feats = vox_feats.permute(0, 3, 1, 2).flatten(0, 1).flatten(1, 2)  # (b*25281,1792)

        # Flatten to orthographic feature map (b,159,159,256)
        ortho_feats = self.conv3d(vox_feats).view(batch, depth, width, -1)
        ortho_feats = F.relu(ortho_feats.permute(0, 3, 1, 2), inplace=True)
        # ortho_feats = F.relu(self.conv3d(vox_feats))
        ortho_feats = ortho_feats.permute(0,1,3,2)
        # Block gradients to pixels which are not visible in the image

        return ortho_feats  ##(b,256,248,216)


def integral_image(features):
    return torch.cumsum(torch.cumsum(features, dim=-1), dim=-2)
###积分图

## 输入FPN出来的feature，tuple类型，共五层特征，每层256维度，（b,256,img_h/4,img_w/4）,(b,256,img_h/8,img_w/8),......

class OftNet(nn.Module):  #（B,256,216,248）

    def __init__(self, img_bev_harf=True, grid_size=(69.12,79.36), grid_offset=(0,-39.68, 0), grid_res=0.32, grid_z_min=-3, grid_z_max=2.76):
        super().__init__()
        corners_radar = make_grid(grid_size, grid_offset, grid_res, grid_z_min, grid_z_max)
        self.register_buffer('corners_radar', corners_radar)  ##(1,18,216,248,3)
        # Orthographic feature transforms
        input_yxchannels = int((grid_z_max-grid_z_min)/grid_res*256)

        self.oft4 = OFT(256, input_yxchannels, 1 / 4.)
        self.oft8 = OFT(256, input_yxchannels,1 / 8.)
        self.oft16 = OFT(256, input_yxchannels, 1 / 16.)
        self.oft32 = OFT(256, input_yxchannels, 1 / 32.)
        self.oft64 = OFT(256,  input_yxchannels,1 / 64.)
        #-------------
        if img_bev_harf:
            self.bevencoder = nn.Sequential(
                nn.Conv2d(256*5,256,kernel_size=3,stride=2,padding=1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        else:
            self.bevencoder = nn.Sequential(
                nn.Conv2d(256*5,256,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        self.attention_block = Attention_Block(256)
        #---------------------------
    def forward(self, img_feats, calib):
        # Normalize by mean and std-dev


        # Run frontend network
        feats4, feats8, feats16, feats32, feats64 = img_feats

        # Apply lateral layers to convert image features to common feature size

        # Apply OFT and sum
        ortho4 = self.oft4(feats4, calib, self.corners_radar)
        ortho8 = self.oft8(feats8, calib, self.corners_radar)
        ortho16 = self.oft16(feats16, calib, self.corners_radar)
        ortho32 = self.oft32(feats32, calib, self.corners_radar)
        ortho64 = self.oft64(feats64, calib, self.corners_radar)

        #----------------------
        ortho = torch.cat([self.attention_block(ortho4),self.attention_block(ortho8),self.attention_block(ortho16), self.attention_block(ortho32),self.attention_block(ortho64)],dim=1)
        ortho = self.bevencoder(ortho)
        #--------------------
# #------------------------xiaorong-------------------
#         ortho = ortho4+ortho8+ortho16+ortho32+ortho64
        return ortho  ###(B, 256, 124, 108)

if __name__ == '__main__':
    ortho1 = torch.randn((8,256,218,232))
    atten = Attention_Block(256)
    x1 = atten(ortho1)
    print(x1.shape)


    #img_feats = (torch.randn((8,256,240,960), requires_grad=True), torch.randn((8,256,120,480),requires_grad=True),torch.randn((8,256,60,240),requires_grad=True),torch.randn((8,256,30,120),requires_grad=True),torch.randn((8,256,15,60),requires_grad=True))
    # calib = torch.tensor([[0, -1, 0, 0.003],
    #                       [0, 0, -1, 1.334],
    #                       [1, 0, 0, 2.875]]).repeat(8,1,1)
    #
    # BEVencoder = OftNet()
    # out_bev = BEVencoder(img_feats,calib)
    # print(out_bev.shape)

    # img_corners = make_grid((69.12,79.36),(0,-39.68, 0),0.32,-3,2.76)
    # print(img_corners.shape)
    # img_size = img_corners.new([1280, 960])
    # norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)
    #
    # bbox_corners = torch.cat([
    #     torch.min(norm_corners[:, :-1, :-1, :-1],  # 这里索引的是前四维，先取小uv再取大
    #               norm_corners[:, :-1, 1:, :-1]),
    #     torch.max(norm_corners[:, 1:, 1:, 1:],
    #               norm_corners[:, 1:, :-1, 1:])
    # ], dim=-1)
    # bbox_corners = bbox_corners.flatten(2, 3)
    # print(bbox_corners.shape)
    # area = ((bbox_corners[..., 2:] - bbox_corners[..., :2]).prod(dim=-1) \
    #         * 360 * 1080 * 0.25*0.125*0.5 + EPSILON).unsqueeze(1)
    # print(area)
    # print(area.shape)