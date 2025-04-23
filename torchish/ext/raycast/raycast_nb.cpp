
#include "../check_macros.hpp"
#include "raycast.hpp"
#include <c10/cuda/CUDAGuard.h>

std::vector<torch::Tensor> raycast_nb(
    torch::Tensor origins,      // [R, 3 (x, y, z)]
    torch::Tensor directions,   // [R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    int64_t kernel = 0)
{
    CHECK_CONTIGUOUS(origins);
    CHECK_CONTIGUOUS(directions);
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(faces);

    if (origins.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        CHECK_CUDA(directions);
        CHECK_CUDA(vertices);
        CHECK_CUDA(faces);

        const at::cuda::OptionalCUDAGuard device_guard(device_of(origins));

        return raycast_CUDA_nb(origins, directions, vertices, faces, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators("cuda::raycast_nb", &raycast_nb);