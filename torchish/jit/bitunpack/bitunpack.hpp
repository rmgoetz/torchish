
#pragma once

#include <torch/script.h>

torch::Tensor bitunpack_2d(
    torch::Tensor bitpacked, // [P, K]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel);

torch::Tensor bitunpack_2d_CUDA(
    torch::Tensor bitpacked, // [P, K]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel);
