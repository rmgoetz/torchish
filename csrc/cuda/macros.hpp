
#pragma once

#include <torch/torch.h>

#define AT_DISPATCH_OUTPUT_BOOL_OR_UINT8(dtype, ...)                 \
    do                                                               \
    {                                                                \
        switch (dtype)                                               \
        {                                                            \
        case torch::kBool:                                           \
        {                                                            \
            using _TYPE = bool;                                      \
            static constexpr at::ScalarType PT_DTYPE = torch::kBool; \
            __VA_ARGS__();                                           \
            break;                                                   \
        }                                                            \
        case torch::kUInt8:                                          \
        {                                                            \
            using _TYPE = uint8_t;                                   \
            constexpr at::ScalarType PT_DTYPE = torch::kUInt8;       \
            __VA_ARGS__();                                           \
            break;                                                   \
        }                                                            \
        default:                                                     \
            throw std::invalid_argument("Unsupported dtype");        \
        }                                                            \
    } while (0)