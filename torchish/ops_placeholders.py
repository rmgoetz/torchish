from typing import Tuple
import torch

def cuda_version_placeholder() -> int:
    return -1

def raycast_placeholder(
    origins: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_batch: torch.Tensor,
    kernel: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ImportError
\
def raycast_nb_placeholder(
    origins: torch.Tensor,
    directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    kernel: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ImportError

def bitpack_2d_placeholder(input: torch.Tensor, kernel: int) -> torch.Tensor:
    raise ImportError
