# Copyright (c) OpenMMLab. All rights reserved.

"""Minimal detector exports for standalone RCDFNet.

该独立子工程先保留 RCDFNet 直接依赖的检测器实现，
将强依赖可视化/外部工具的大量实验性 detector 设为可选。
"""

from .base import Base3DDetector
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN, RC_backbone
from .mvx_two_stage import MVXTwoStageDetector

__all__ = [
    'Base3DDetector',
    'DynamicMVXFasterRCNN',
    'MVXFasterRCNN',
    'RC_backbone',
    'MVXTwoStageDetector',
]

