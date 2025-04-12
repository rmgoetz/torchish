
#include "scatter_minmax.hpp"
#include "reducer.cuh"
#include <tuple>


template <typename scalar_t, ReductionType REDUCE>
__global__ void scatter_kernel(
    const scalar_t *src,
    const at::cuda::detail::TensorInfo<int64_t, int> index_info,
    scalar_t *out

){

}

template <typename zipped_scalar_t, typename scalar_t, ReductionType REDUCE>
__global__ void unzip_kernel(
    const zipped_scalar_t *vals_and_index,
    scalar_t *vals,
    int64_t *index
){

    

}


std::tuple<torch::Tensor, torch::Tensor> scatter_min_CUDA(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out
){

}