// For the CUDA runtime routines (prefixed with "cuda_")
// #include <cuda.h>
#include <cuda_runtime.h>

namespace
{
    __global__ void _vadd(const float *A, const float *B, float *C, int n)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n)
        {
            C[i] = A[i] + B[i];
        }
    }
}

extern "C" void vadd(const float *A, const float *B, float *C, int n, int threads)
{
    const int blocks = (n + threads - 1) / threads;
    _vadd<<<blocks, threads>>>(A, B, C, n);
}
