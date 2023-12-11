from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bm3d_cuda',
    ext_modules=[
        CUDAExtension('bm3d_cuda', [
            'cuda/bm3d_cuda.cpp',
            'cuda/filtering.cu',
            'cuda/blockmatching.cu',
            'cuda/dct8x8.cu',
        ],
        libraries=['cufft', 'cudart', 'png']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
