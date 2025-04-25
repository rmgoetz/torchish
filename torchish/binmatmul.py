import torch

__all__ = [
    'bitpack_2d'
]

def bitpack_2d(input: torch.Tensor, kernel: int = 0) -> torch.Tensor:
    """Compactifies 2D boolean tensors along the row dimension, with 
    zero padding up to the nearest multiple of 8.

    Args:
        input: A [N, M] boolean tensor.

    Returns:
        torch.Tensor: A [N, K] tensor with dtype uint8
    """

    return torch.ops.torchish.bitpack_2d(input, kernel) # [K, M]  
