#!/usr/bin/env python


# %%
import os
from ctypes import CDLL, c_void_p

import torch
import triton
import triton.language as tl

# %%
path = os.path.join(os.environ['PWD'], 'cuda_ctypes')
vadd_so = os.path.join(path, 'vadd.so')
vadd = CDLL(vadd_so)


# %%
@triton.jit
def _add(
    X,  # *Pointer* to first input vector
    Y,  # *Pointer* to second input vector
    Z,  # *Pointer* to output vector
    N,  # Size of the vector
    **meta  # Optional meta-parameters for the kernel
):
    pid = tl.program_id(0)
    # Create an offset for the blocks of pointers to be
    # processed by this program instance
    offsets = pid * meta['BLOCK'] + tl.arange(0, meta['BLOCK'])
    # Create a mask to guard memory operations against
    # out-of-bounds accesses
    mask = offsets < N
    # Load x
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    # Write back x + y
    z = x + y
    tl.store(Z + offsets, z)

def add(x, y):
    z = torch.empty_like(x)
    N = z.shape[0]
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']), )
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments
    _add[grid](x, y, z, N, BLOCK=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return z

# %%
def add_cuda(x, y, threads=1024):
    z = torch.empty_like(x)
    N = z.shape[0]
    # Get pointers to the data
    xp = c_void_p(x.data_ptr())
    yp = c_void_p(y.data_ptr())
    zp = c_void_p(z.data_ptr())
    # Run the cuda kernel
    vadd.vadd(xp, yp, zp, N, threads)
    return z


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)
za = x + y
zb = add_cuda(x, y)
print(za)
print(zb)
print(f'The maximum difference between torch and cuda is ' f'{torch.max(torch.abs(za - zb))}')

# %%
# triton PTX
size = 98432
x = torch.rand(size, device='cuda', dtype=torch.float32)
y = torch.rand(size, device='cuda', dtype=torch.float32)
z = torch.empty_like(x)
grid = (triton.cdiv(size, 1024),)
pgm = _add[grid](x, y, z, size, BLOCK=1024)
print(pgm.asm('ptx'))
pgm

# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'cuda'],  # possible values for `line_arg`
        line_names=["Triton", "Torch", "CUDA"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="vector-add-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={}  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y))
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_cuda(x, y))
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)

# %%
