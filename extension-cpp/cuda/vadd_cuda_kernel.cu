#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace
{
  template <typename scalar_t>
  __global__ void vadd_cuda_forward_kernel(
      const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> A,
      const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> B,
      torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> C)
  {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < A.size(0))
    {
      C[i] = A[i] + B[i];
    }
  }
}

std::vector<torch::Tensor> vadd_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    int threads)
{
  auto C = torch::zeros_like(A);

  const int blocks = (A.size(0) + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(
      gates.type(),
      "vadd_forward_cuda",
      ([&]
       { vadd_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
             A.packed_accessor<scalar_t, , torch::RestrictPtrTraits, size_t>(),
             B.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
             C.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()); }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}
