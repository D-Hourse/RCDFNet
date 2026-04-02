import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = os.path.dirname(os.path.abspath(__file__))


setup(
    name='sparse_conv_ext',
    ext_modules=[
        CUDAExtension(
            name='sparse_conv_ext',
            sources=[
                os.path.join('src', 'all.cc'),
                os.path.join('src', 'reordering.cc'),
                os.path.join('src', 'reordering_cuda.cu'),
                os.path.join('src', 'indice.cc'),
                os.path.join('src', 'indice_cuda.cu'),
                os.path.join('src', 'maxpool.cc'),
                os.path.join('src', 'maxpool_cuda.cu'),
            ],
            include_dirs=[os.path.join(ROOT, 'include')],
            extra_compile_args={
                'cxx': ['-w', '-std=c++14'],
                'nvcc': [
                    '-w',
                    '-std=c++14',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ],
            },
            define_macros=[('WITH_CUDA', None)],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
