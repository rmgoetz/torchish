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
        name = "cuda_raycast",
        sources = [os.path.join(FILE_DIR, "scatter_minmax.cpp"),
                   os.path.join(FILE_DIR, "scatter_minmax.cu")],
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
    dim: int = 0,
    kernel: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Casts rays onto a collection of triangulated facets, returning the distance to the first collision, and the normal
    of the facet hit.

    Args:
        origins: A [B, R, 3 (x, y, z)] or [R, 3 (x, y, z)] of ray origin points
        directions: A [B, R, 3 (x, y, z)] or [R, 3 (x, y, z)]  of ray direction vectors
        vertices: A [V, 3 (x, y, z)] of vertex locations
        faces: A [F, 3 (v0, v1, v2)] of triangulated facets
        vertex_batch: An optional [V] corresponding each vertex to a batch (B), must be consecutive and sorted. Defaults 
                      to None.
        kernel: An option to choose the computation kernel. Defaults to 0.

    Returns:
        torch.Tensor: A [B, R] or [R] of distances from origins to nearest facet along ray direction. Values are infinite 
                      for cases of no coliision.
        torch.Tensor: A [B, R, 3 (x, y, z)] or [R, 3 (x, y, z)] of normal vectors of the facets hit by each ray. Values are infinite for cases
                      of no collision.
    """
  
