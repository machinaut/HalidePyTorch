
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

#define numElements (50000)
#define size (numElements * sizeof(float))
#define threadsPerBlock (256)
#define blocksPerGrid ((numElements + threadsPerBlock - 1) / threadsPerBlock)

static float h_A[numElements];
static float h_B[numElements];
static float h_C[numElements];

static float *d_A = 0;
static float *d_B = 0;
static float *d_C = 0;

int main(void)
{
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; ++i)
    {
        if (h_C[i] != i * 3)
        {
            return 1;
        }
    }
    return 0;
}
