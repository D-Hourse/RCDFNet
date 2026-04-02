from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='iou3d_cuda',
    ext_modules=[
        CUDAExtension(
            name='iou3d_cuda',
            sources=['src/iou3d.cpp', 'src/iou3d_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
