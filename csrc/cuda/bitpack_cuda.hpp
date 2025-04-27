
#pragma once

#include <torch/torch.h>

torch::Tensor bitpack_2d_CUDA(
    torch::Tensor input,
    int64_t kernel);