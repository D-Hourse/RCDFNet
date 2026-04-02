from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='bev_pool_v2_ext',
    ext_modules=[
        CUDAExtension(
            name='bev_pool_v2_ext',
            sources=['src/bev_pool.cpp', 'src/bev_pool_cuda.cu'],
            extra_compile_args={
                'cxx': ['-std=c++14'],
                'nvcc': [
                    '-std=c++14',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
