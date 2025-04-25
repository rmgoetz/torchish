
#include "example.hpp"

__global__ void example_kernel()
{
    return;
}

torch::Tensor example_CUDA(torch::Tensor input)
{
    dim3 threads(256, 1, 1);
    dim3 blocks(256, 1, 1);
    example_kernel<<<blocks, threads>>>();

    return input;
}