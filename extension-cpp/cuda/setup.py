from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vadd_cuda',
    ext_modules=[
        CUDAExtension('vadd_cuda', [
            'vadd_cuda.cpp',
            'vadd_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
