from typing import Tuple
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
        name = "cuda_scatter",
        sources = [os.path.join(FILE_DIR, "scatter.cpp"),
                   os.path.join(FILE_DIR, "scatter_cuda.cu")],
        is_python_module = False,
        build_directory = BUILD_DIR,
        extra_cflags = extra_flags + ["-DCOMPILED_WITH_CUDA"],
        extra_ldflags = extra_ldflags,
        extra_cuda_cflags = ["-lineinfo"]
    )
else:
    raise Exception("CPU implementation not yet supported.")


def scatter_min(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int,
        optional_out: torch.Tensor,
        dim_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        src (torch.Tensor): _description_
        index (torch.Tensor): _description_
        dim (int): _description_
        optional_out (torch.Tensor): _description_
        dim_size (int): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """

    return torch.ops.cuda.scatter_min(
        src,
        index,
        dim,
        optional_out,
        dim_size
    )

def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int,
        optional_out: torch.Tensor,
        dim_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        src (torch.Tensor): _description_
        index (torch.Tensor): _description_
        dim (int): _description_
        optional_out (torch.Tensor): _description_
        dim_size (int): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """

    return torch.ops.cuda.scatter_max(
        src,
        index,
        dim,
        optional_out,
        dim_size
    )
