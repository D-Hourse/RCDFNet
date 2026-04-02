# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .second import SECOND
from .resnet import CustomResNet

try:
    from .pointnet2_sa_msg import PointNet2SAMSG
    from .pointnet2_sa_ssg import PointNet2SASSG
except Exception:
    PointNet2SAMSG = None
    PointNet2SASSG = None

try:
    from .sst_v2 import SSTv2
except Exception:
    SSTv2 = None

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'MultiBackbone',
    #######################custom#########################################
    'CustomResNet'
]

if PointNet2SASSG is not None:
    __all__.append('PointNet2SASSG')
if PointNet2SAMSG is not None:
    __all__.append('PointNet2SAMSG')
if SSTv2 is not None:
    __all__.append('SSTv2')
