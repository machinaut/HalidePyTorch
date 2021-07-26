from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vadd',
    ext_modules=[
        CUDAExtension('vadd', [
            'vadd.cpp',
            'vadd_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
