#include <torch/extension.h>

// CUDA forward declarations

torch::Tensor vadd_cuda_forward(torch::Tensor A, torch::Tensor B, int threads);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor vadd_forward(
    torch::Tensor A,
    torch::Tensor B,
    int threads)
{
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    return vadd_cuda_forward(A, B, threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &vadd_forward, "VAdd forward (CUDA)");
}
