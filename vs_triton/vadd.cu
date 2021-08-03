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
#define INNER_SIZE (THREADS * VECTORIZE)

namespace
{
    __global__ void _vadd(const float *A, const float *B, float *C, int n)
    {
        const int start = BLOCK_SIZE * blockIdx.x + VECTORIZE * threadIdx.x;
        for (int j = 0; j < UNROLL; j++)
        {
            const int sub = start + j * INNER_SIZE;
            for (int k = 0; k < VECTORIZE; k++)
            {
                const int i = sub + k;
                if (i >= n)
                    return;
                C[i] = A[i] + B[i];
            }
        }
    }
}

extern "C" void vadd(const float *A, const float *B, float *C, int n)
{
    const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    _vadd<<<blocks, THREADS>>>(A, B, C, n);
}
