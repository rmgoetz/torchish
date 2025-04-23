
#include "scatter_minmax.hpp"
#include "reducer.cuh"
#include <tuple>

// template <typename scalar_t, ReductionType REDUCE>
// __global__ void scatter_kernel(
//     const scalar_t *src,
//     const at::cuda::detail::TensorInfo<int64_t, int> index_info,
//     scalar_t *out

// ){

// }

// template <typename zipped_scalar_t, typename scalar_t, ReductionType REDUCE>
// __global__ void unzip_kernel(
//     const zipped_scalar_t *vals_and_index,
//     scalar_t *vals,
//     int64_t *index
// ){

// }

__global__ void scatter_min_kernel(
    const float *src,
    const int32_t *index,
    double *out)
{
    return;
}

__global__ void unzip_kernel(
    const uint64_t *vals_and_index,
    float *vals,
    int32_t *argmin,
    const int threads_per_block,
    const int blockX,
    const int blockY,
    const int blockZ
){
    // The index for this thread within the block
    const uint32_t thread_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

    //
    const uint32_t flat_index = 0;

    //
    uint64_t val_and_ind;

    //
    float val = static_cast<float>(val_and_ind >> 32);
    int32_t argmin = static_cast<uint32_t>(val_and_ind & 0xFFFFFFFF);

    // *vals_and_index;
    
}

std::tuple<torch::Tensor, torch::Tensor> scatter_min_CUDA(
    torch::Tensor src,
    torch::Tensor index,
    int64_t dim,
    std::optional<torch::Tensor> out)
{
    using namespace torch::indexing;

    torch::Tensor vals_and_index;   // [*]
    int64_t out_elements = vals_and_index.numel();


}