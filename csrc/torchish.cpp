#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <torch/script.h>

#include "api_macros.hpp"
#include "check_macros.hpp"

#ifdef COMPILED_WITH_CUDA
#include "cuda/raycast_cuda.hpp"
#include "cuda/bitpack_cuda.hpp"
#include "cuda/bitunpack_cuda.hpp"
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef COMPILED_WITH_CUDA
PyMODINIT_FUNC PyInit__torchish_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__torchish_cpu(void) { return NULL; }
#endif
#endif
#endif

TORCHISH_API std::vector<torch::Tensor> raycast(
    torch::Tensor origins,      // [B, R, 3 (x, y, z)]
    torch::Tensor directions,   // [B, R, 3 (x, y, z)]
    torch::Tensor vertices,     // [V, 3 (x, y, z)]
    torch::Tensor faces,        // [F, 3 (v0, v1, v2)]
    torch::Tensor vertex_batch, // [V] consecutive and sorted
    int64_t kernel)
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

        return raycast_CUDA(origins, directions, vertices, faces, vertex_batch, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

TORCHISH_API std::vector<torch::Tensor> raycast_nb(
    torch::Tensor origins,    // [R, 3 (x, y, z)]
    torch::Tensor directions, // [R, 3 (x, y, z)]
    torch::Tensor vertices,   // [V, 3 (x, y, z)]
    torch::Tensor faces,      // [F, 3 (v0, v1, v2)]
    int64_t kernel)
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

TORCHISH_API torch::Tensor bitpack_2d(torch::Tensor input, int64_t kernel)
{
    CHECK_CONTIGUOUS(input);

    if (input.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        return bitpack_2d_CUDA(input, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

TORCHISH_API torch::Tensor bitunpack_2d(
    torch::Tensor bitpacked_tensor, // [N*, K]
    int64_t N,
    int64_t M,
    at::ScalarType dtype,
    int64_t kernel)
{
    CHECK_CONTIGUOUS(bitpacked_tensor);

    if (bitpacked_tensor.is_cuda())
    {
#ifdef COMPILED_WITH_CUDA

        return bitunpack_2d_CUDA(bitpacked_tensor, N, M, dtype, kernel);

#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    else
    {
        AT_ERROR("No CPU implementation.");
    }
}

static auto registry = torch::RegisterOperators()
                           .op("torchish::raycast", &raycast)
                           .op("torchish::raycast_nb", &raycast_nb)
                           .op("torchish::bitpack_2d", &bitpack_2d)
                           .op("torchish::bitunpack_2d", &bitunpack_2d);