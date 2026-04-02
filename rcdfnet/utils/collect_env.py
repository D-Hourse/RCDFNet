# Copyright (c) OpenMMLab. All rights reserved.
import platform

import mmcv
import torch
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdet
import rcdfnet
import mmseg


def collect_env():
    """Collect the information of the running environments."""
    try:
        env_info = collect_base_env()
    except Exception as exc:  # pragma: no cover
        env_info = {
            'Platform': platform.platform(),
            'Python': platform.python_version(),
            'PyTorch': torch.__version__,
            'MMCV': mmcv.__version__,
            'MMCV Compiler': f'unavailable ({exc})',
            'MMCV CUDA Compiler': 'unavailable',
        }
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = mmseg.__version__
    env_info['RCDFNet3D'] = rcdfnet.__version__ + '+' + get_git_hash()[:7]

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
