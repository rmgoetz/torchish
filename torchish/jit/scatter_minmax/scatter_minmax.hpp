
#ifndef CUDEXT_SCATTER_MINMAX
#define CUDEXT_SCATTER_MINMAX

#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> scatter_min(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out);

std::tuple<torch::Tensor, torch::Tensor> scatter_max(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out);

std::tuple<torch::Tensor, torch::Tensor> scatter_min_CUDA(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out);

std::tuple<torch::Tensor, torch::Tensor> scatter_max_CUDA(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out);

#endif