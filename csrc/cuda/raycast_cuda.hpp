
#pragma once

#include <torch/torch.h>

std::vector<torch::Tensor> raycast_CUDA(
    torch::Tensor origins,      // [B, R, 3 (x, y, z)]
    torch::Tensor directions,   // [B, R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    torch::Tensor vertex_batch, // [V] consecutive and sorted
    int64_t kernel);

std::vector<torch::Tensor> raycast_CUDA_nb(
    torch::Tensor origins,    // [R, 3 (x, y, z)]
    torch::Tensor directions, // [R, 3 (x, y, z)]
    torch::Tensor vertices,   // [V, 3 (x, y, z)]
    torch::Tensor faces,      // [F, 3 (v0, v1, v2)]
    int64_t kernel);