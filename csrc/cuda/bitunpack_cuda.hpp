
#pragma once

#include <torch/torch.h>

torch::Tensor bitunpack_2d_CUDA(
    torch::Tensor bitpacked, // [P, K]
    int64_t N,
    int64_t M,
    int64_t kernel);