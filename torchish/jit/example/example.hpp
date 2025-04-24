
#pragma once

#include <torch/script.h>

torch::Tensor example(torch::Tensor input);

torch::Tensor example_CUDA(torch::Tensor input);
