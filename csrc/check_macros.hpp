/*
A module defining the macros to validate the inputs of our C++ extension functions
*/

#pragma once

#include <torch/script.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) \
    CHECK_CUDA(x); \
    CHECK_CONTIGUOUS(x);