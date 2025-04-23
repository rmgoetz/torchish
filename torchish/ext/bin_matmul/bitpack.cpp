
#include "../check_macros.hpp"
#include "bitpack.hpp"

torch::Tensor bitpack_2d(torch::Tensor input, int64_t kernel)
{
    CHECK_CONTIGUOUS(input);

    if (input.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        return bitpack_2d_CUDA(input, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators("cuda::bitpack_2d", &bitpack_2d);