import os
import sys
import torch
from torch.utils import cpp_extension

'''
A template for testing kernels.
'''

__all__ = ['example']

FILE_DIR = os.path.dirname(__file__)
BUILD_DIR = os.path.join(FILE_DIR, "build")

if not os.path.exists(BUILD_DIR):
    os.makedirs(BUILD_DIR)

if sys.platform.startswith("win"):
    extra_flags = ["/openmp"]
    extra_ldflags = []
else:
    extra_flags = []
    extra_ldflags = []

if torch.cuda.is_available():
    cpp_extension.load(
        name = "cuda_example",
        sources = [os.path.join(FILE_DIR, "example.cpp"),
                   os.path.join(FILE_DIR, "example.cu")],
        is_python_module = False,
        build_directory = BUILD_DIR,
        extra_cflags = extra_flags + ["-DCOMPILED_WITH_CUDA"],
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = ["-lineinfo"]
    )
else:
    raise Exception("CPU implementation not yet supported.")


def example(input: torch.Tensor) -> torch.Tensor:
    return torch.ops.cuda.example(input)

