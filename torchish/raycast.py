from typing import Tuple
import torch

__all__ = [
    'raycast'
]

def raycast(
    origins: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_batch: torch.Tensor = None,
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

    if vertex_batch is not None:
        return torch.ops.torchish.raycast(
            origins,
            directions,
            vertices,
            faces,
            vertex_batch,
            kernel
        )   # [B, R], [B, R, 3 (x, y, z)]
    else:
        return torch.ops.torchish.raycast_nb(
            origins,
            directions,
            vertices,
            faces,
            kernel
        )   # [R], [R, 3 (x, y, z)]      
