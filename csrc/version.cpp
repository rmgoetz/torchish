#ifdef WITH_PYTHON
#include <Python.h>
#endif

#include <torch/script.h>
#include "torchish.hpp"
#include "api_macros.hpp"

#ifdef COMPILED_WITH_CUDA
#ifdef USE_ROCM
#include <hip/hip_version.h>
#else
#include <cuda.h>
#endif
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef COMPILED_WITH_CUDA
PyMODINIT_FUNC PyInit__version_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__version_cpu(void) { return NULL; }
#endif
#endif
#endif

namespace torchish
{
    TORCHISH_API int64_t cuda_version() noexcept
    {
#ifdef COMPILED_WITH_CUDA
#ifdef USE_ROCM
        return HIP_VERSION;
#else
        return CUDA_VERSION;
#endif
#else
        return -1;
#endif
    }
} // namespace torchish

static auto registry = torch::RegisterOperators().op(
    "torchish::cuda_version", []
    { return torchish::cuda_version(); });