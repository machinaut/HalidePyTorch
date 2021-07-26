#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace
{
    __global__ void _zero(float *A, int n)
    {
        printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n)
        {
            printf("Writing to i=%d.\n", i);
            printf("A[i] = %f.\n", A[i]);
            A[i] = 0;
        }
        else
        {
            printf("Skipping index %d\n", i);
        }
    }
}

extern "C" void zero(float *A, int n, int threads)
{
    printf("Got A pointer %p n %d threads %d\n", A, n, threads);
    const char *res_str;
    CUdeviceptr pbase;
    size_t psize;
    CUresult res = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)A);
    cuGetErrorName(res, &res_str);
    if (res != CUDA_SUCCESS)
        printf("Error getting A mem address range: (%d) %s\n", res, res_str);
    else
        printf("A mem address range: 0x%p - 0x%p, size: %zu\n", (void *)pbase, (void *)pbase + psize, psize);

    printf("Running the kernel\n");
    const int blocks = (n + threads - 1) / threads;
    _zero<<<blocks, threads>>>(A, n);
    printf("Finished running the kernel\n");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Failed to launch zero kernel (error code %s)!\n", cudaGetErrorString(err));
    else
        printf("Successfully launched zero kernel!\n");
}
