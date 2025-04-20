
#ifndef CUDEXT_COMPACTIFY_2D
#define CUDEXT_COMPACTIFY_2D

#include <torch/script.h>

torch::Tensor bitpack_2d(
    torch::Tensor input,
    int64_t kernel = 0);

torch::Tensor bitpack_2d_CUDA(
    torch::Tensor input,
    int64_t kernel);

#endif