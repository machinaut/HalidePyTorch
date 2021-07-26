from torch.utils.cpp_extension import load
vadd_cuda = load(
    'vadd_cuda', ['vadd_cuda.cpp', 'vadd_cuda_kernel.cu'], verbose=True)
help(vadd_cuda)
