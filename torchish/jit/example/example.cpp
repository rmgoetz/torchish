
#include "example.hpp"

torch::Tensor example(torch::Tensor input)
{
    if (input.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA
        return example_CUDA(input);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators("cuda::example", &example);