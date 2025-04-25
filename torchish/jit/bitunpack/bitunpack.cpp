
#include "bitunpack.hpp"
#include "check_macros.hpp"

torch::Tensor bitunpack_2d(
    torch::Tensor bitpacked_tensor, // [N*, K]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel)
{
    CHECK_CONTIGUOUS(bitpacked_tensor);

    if (bitpacked_tensor.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        return bitunpack_2d_CUDA(bitpacked_tensor, N, M, dtype, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators().op("cuda::bitunpack_2d", &bitunpack_2d);