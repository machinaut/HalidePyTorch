// For the CUDA runtime routines (prefixed with "cuda_")
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace
{
    __global__ void _vadd(const float *A, const float *B, float *C, int n)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
        if (i < n)
        {
            printf("Writing to i=%d.\n", i);
            printf("A[i] = %f\n", A[i]);
            printf("B[i] = %f\n", B[i]);
            C[i] = A[i] + B[i];
            printf("C[i] = %f\n", C[i]);
        }
        else
        {
            printf("Skipping index %d\n", i);
        }
    }
}

extern "C" void vadd(const float *A, const float *B, float *C, int n, int threads)
{
    cudaError_t err;
    // CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )
    CUresult res;
    const char *res_str;
    CUdeviceptr pbase;
    size_t psize;
    res = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)A);
    if (res != CUDA_SUCCESS)
    {
        cuGetErrorName(res, &res_str);
        printf("Error getting A mem address range: (%d) %s\n", res, res_str);
    }
    else
    {
        printf("A mem address range: 0x%p - 0x%p, size: %zu\n", (void *)pbase, (void *)pbase + psize, psize);
    }
    res = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)B);
    if (res != CUDA_SUCCESS)
    {
        cuGetErrorName(res, &res_str);
        printf("Error getting B mem address range: %s\n", res_str);
    }
    else
    {
        printf("B mem address range: 0x%p - 0x%p, size: %zu\n", (void *)pbase, (void *)pbase + psize, psize);
    }
    res = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)C);
    if (res != CUDA_SUCCESS)
    {
        cuGetErrorName(res, &res_str);
        printf("Error getting C mem address range: %s\n", res_str);
    }
    else
    {
        printf("C mem address range: 0x%p - 0x%p, size: %zu\n", (void *)pbase, (void *)pbase + psize, psize);
    }

    const int blocks = (n + threads - 1) / threads;
    _vadd<<<blocks, threads>>>(A, B, C, n);
    printf("kernel over\n");
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        printf("Failed to launch vadd kernel (error code %s)!\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Successfully launched vadd kernel!\n");
    }
}
