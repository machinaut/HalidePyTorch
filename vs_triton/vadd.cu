#include <cuda_runtime.h>

__global__ void _vadd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

extern "C" void vadd(const float *A, const float *B, float *C, int n)
{
    const int blocks = (n + 64 - 1) / 64;
    _vadd<<<blocks, 64>>>(A, B, C, n);
}
