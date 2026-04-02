from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='voxel_layer',
    ext_modules=[
        CUDAExtension(
            name='voxel_layer',
            sources=[
                'src/voxelization.cpp',
                'src/scatter_points_cpu.cpp',
                'src/scatter_points_cuda.cu',
                'src/voxelization_cpu.cpp',
                'src/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)],
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
