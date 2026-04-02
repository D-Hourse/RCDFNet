from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths
from pathlib import Path


ROOT = Path(__file__).resolve().parent
local_src = ROOT / 'third_party' / 'torchex' / 'src' / 'ingroup_inds'
legacy_src = ROOT.parent / 'TorchEx-main' / 'torchex' / 'src' / 'ingroup_inds'

src_dir = local_src if local_src.exists() else legacy_src
cpp_file = src_dir / 'ingroup_inds.cpp'
cu_file = src_dir / 'ingroup_inds_kernel.cu'

if not (cpp_file.exists() and cu_file.exists()):
    raise FileNotFoundError(
        'Cannot find ingroup_indices sources. Expected files:\n'
        f'- {cpp_file}\n'
        f'- {cu_file}\n'
        'Please vendor TorchEx sources into RCDFNet-main/third_party/torchex/src/ingroup_inds.'
    )


setup(
    name='rcdfnet.ops.ingroup_indices',
    ext_modules=[
        CUDAExtension(
            'rcdfnet.ops.ingroup_indices',
            [
                str(cpp_file),
                str(cu_file),
            ],
            extra_compile_args={
                'cxx': ['-std=c++14'],
                'nvcc': [
                    '-std=c++14',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ],
            },
            runtime_library_dirs=[p for p in library_paths(cuda=True) if Path(p).is_absolute()],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
