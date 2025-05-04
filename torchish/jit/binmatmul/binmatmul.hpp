
#pragma once

#include <torch/script.h>

torch::Tensor binmatmul(
    torch::Tensor A, // [N, K]
    torch::Tensor B, // [K, M]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel);

torch::Tensor binmatmul_CUDA(
    torch::Tensor A, // [N, K]
    torch::Tensor B, // [K, M]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel);
