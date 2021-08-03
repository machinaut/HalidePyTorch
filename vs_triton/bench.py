#!/usr/bin/env python


# %%
import os
from ctypes import CDLL, c_void_p

import torch
import triton
import triton.language as tl
from itertools import product

# %%
vadds = {}
thread_range = [32, 64, 128, 256, 512, 1024]
vectorize_range = [1, 2, 3, 4, 8, 16, 32]
unroll_range = [1, 2, 3, 4, 8, 16, 32]
for threads, vectorize, unroll in product(thread_range, vectorize_range, unroll_range):
    path = f'/tmp/vadd/vadd_{threads}_{vectorize}_{unroll}.so'
    vadds[(threads, vectorize, unroll)] = CDLL(path)



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
def add_cuda(x, y, tup):
    z = torch.empty_like(x)
    N = z.shape[0]
    # Get pointers to the data
    xp = c_void_p(x.data_ptr())
    yp = c_void_p(y.data_ptr())
    zp = c_void_p(z.data_ptr())
    # Run the cuda kernel
    vadds[tup].vadd(xp, yp, zp, N)
    # max_diff = torch.max(torch.abs(z - (x + y)))
    # if max_diff != 0.0:
    #     # print the indexes that are different
    #     print(f'{tup}')
    #     print(f'{max_diff}')
    #     result = x + y
    #     # print the indexes where result is different than z
    #     print(f'{torch.nonzero(torch.abs(result - z) > 1e-10)}')
    #     assert torch.allclose(result, z)
    # assert max_diff == 0.0, f'Max diff: {max_diff}, {tup}, {x.shape}'
    return z


for tup in product(thread_range, vectorize_range, unroll_range):
    torch.manual_seed(hash(tup))
    size = 4096
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    za = x + y
    zb = add_cuda(x, y, tup)
    max_diff = torch.max(torch.abs(za - zb))
    assert max_diff == 0.0, f'Max diff: {max_diff}, {tup}'

# # %%
# # triton PTX
# size = 98432
# x = torch.rand(size, device='cuda', dtype=torch.float32)
# y = torch.rand(size, device='cuda', dtype=torch.float32)
# z = torch.empty_like(x)
# grid = (triton.cdiv(size, 1024),)
# pgm = _add[grid](x, y, z, size, BLOCK=1024)
# print(pgm.asm('ptx'))
# pgm

# %%
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i - 100 for i in range(12, 28, 1)],  # different possible values for `x_name`
        x_log=True,  # x axis is logarithmic
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'cuda', 'cuda2'],  # possible values for `line_arg`
        line_names=["Triton", "Torch", "CUDA", "CUDA2"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('red', '-')],  # line styles
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
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_cuda(x, y, (512, 1, 2)))
    if provider == 'cuda2':  # This seems like the best
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_cuda(x, y, (1024, 2, 1)))
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)

# %%
# vector-add-performance:
#            size      Triton       Torch        CUDA       CUDA2
# 0        3996.0    9.365625    9.365625    9.365625   11.707031
# 1        8092.0   18.965625   23.707031   18.965625   18.965625
# 2       16284.0   38.165626   38.165626   37.694445   38.165626
# 3       32668.0   63.804685   76.565627   63.804685   63.804685
# 4       65436.0  127.804683  127.804683  109.546873  109.546873
# 5      130972.0  191.853516  191.853516  170.536452  170.536452
# 6      262044.0  255.902335  255.902335  255.902335  255.902335
# 7      524188.0  323.306758  323.306758  307.141414  307.141414
# 8     1048476.0  351.052223  361.377291  351.052223  351.052223
# 9     2097052.0  390.076640  390.076640  383.981690  383.981690
# 10    4194204.0  402.875648  399.600212  399.600212  402.875648
# 11    8388508.0  411.308903  409.595128  409.595128  409.595128
# 12   16777116.0  413.909105  414.782337  413.909105  414.782337
# 13   33554332.0  416.100357  416.541119  416.541119  416.982841
# 14   67108764.0  417.426124  417.647807  417.647807  417.869725
# 15  134217628.0  417.981085  418.092192  418.314583  418.425867

# Notes:
# Triton wins on sizes: 8388508
# Torch wins on sizes: 8092, 32668, 1048476
# Both win on sizes: 65436, 524188, 130972, 2097052