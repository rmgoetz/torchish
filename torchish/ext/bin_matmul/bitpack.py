import os
import sys
import torch
from torch.utils import cpp_extension

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
        name = "cuda_bitpack",
        sources = [os.path.join(FILE_DIR, "bitpack.cpp"),
                   os.path.join(FILE_DIR, "bitpack.cu")],
        is_python_module = False,
        build_directory = BUILD_DIR,
        extra_cflags = extra_flags + ["-DCOMPILED_WITH_CUDA"],
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = ["-lineinfo"]
    )
else:
    raise Exception("CPU implementation not yet supported.")

def bitpack_2d(input: torch.Tensor, kernel: int = 0) -> torch.Tensor:
    """Compactifies 2D boolean tensors along the row dimension.

    Args:
        input (torch.Tensor): A [N, M] boolean tensor.

    Returns:
        torch.Tensor: A [N, K] tensor with dtype uint8
    """

    return torch.ops.cuda.bitpack_2d(input, kernel) # [K, M]  
