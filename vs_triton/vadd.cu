// For the CUDA runtime routines (prefixed with "cuda_")
// #include <cuda.h>
#include <cuda_runtime.h>

#ifndef THREADS
#error "THREADS must be defined"
#endif
#ifndef VECTORIZE
#error "VECTORIZE must be defined"
#endif
#ifndef UNROLL
#error "UNROLL must be defined"
#endif

#define BLOCK_SIZE (THREADS * VECTORIZE * UNROLL)

namespace
{
    __global__ void vadd(const float *A, const float *B, float *C, int n)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        for (int j = 0; j < UNROLL; j++)
        {
            for (int k = 0; k < VECTORIZE; k++)
            {
                if (i >= n)
                    return;
                C[i] = A[i] + B[i];
                i++;
            }
            i += BLOCK_SIZE;
        }
    }
}

extern "C" void vadd(const float *A, const float *B, float *C, int n);
{
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    _vadd<<<blocks, THREADS>>>(A, B, C, n);
}
