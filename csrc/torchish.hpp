#pragma once

#include "api_macros.hpp"
#include <torch/torch.h>

namespace torchish
{
    TORCHISH_API int64_t cuda_version() noexcept;

    namespace detail
    {
        TORCHISH_INLINE_VARIABLE int64_t _cuda_version = cuda_version();
    } // namespace detail
} // namespace torchish

TORCHISH_API std::vector<torch::Tensor> raycast(
    torch::Tensor origins,      // [B, R, 3 (x, y, z)]
    torch::Tensor directions,   // [B, R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    torch::Tensor vertex_batch, // [V] consecutive and sorted
    int64_t kernel);

TORCHISH_API std::vector<torch::Tensor> raycast_nb(
    torch::Tensor origins,    // [R, 3 (x, y, z)]
    torch::Tensor directions, // [R, 3 (x, y, z)]
    torch::Tensor vertices,   // [V, 3 (x, y, z)]
    torch::Tensor faces,      // [F, 3 (v0, v1, v2)]
    int64_t kernel);

TORCHISH_API torch::Tensor bitpack_2d(
    torch::Tensor input, // [N, M]
    int64_t kernel);

TORCHISH_API torch::Tensor bitunpack_2d(
    torch::Tensor bitpacked, // [P, K]
    int64_t N,
    int64_t M,
    int64_t kernel);