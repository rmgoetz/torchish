
#include "../check_macros.hpp"
#include "raycast.hpp"
#include <c10/cuda/CUDAGuard.h>

std::vector<torch::Tensor> raycast(
    torch::Tensor origins,      // [B, R, 3 (x, y, z)]
    torch::Tensor directions,   // [B, R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    torch::Tensor vertex_batch) // [V] consecutive and sorted
{
    CHECK_CONTIGUOUS(origins);
    CHECK_CONTIGUOUS(directions);
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(faces);
    CHECK_CONTIGUOUS(vertex_batch);

    if (origins.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        CHECK_CUDA(directions);
        CHECK_CUDA(vertices);
        CHECK_CUDA(faces);
        CHECK_CUDA(vertex_batch);

        const at::cuda::OptionalCUDAGuard device_guard(device_of(origins));

        return raycast_CUDA(origins, directions, vertices, faces, vertex_batch);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators("cuda::raycast", &raycast);