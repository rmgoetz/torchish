import torch

__all__ = [
    'bitpack_2d',
    'bitunpack_2d'
]

def bitpack_2d(input: torch.Tensor, kernel: int = 0) -> torch.Tensor:
    """Compactifies 2D boolean tensors along the row dimension, with 
    zero padding up to the nearest multiple of 8.

    Args:
        input: A [N, M] boolean tensor (or uint8 tensor of 0s and 1s).

    Returns:
        torch.Tensor: A [N, K] tensor with dtype uint8.
    """
    return torch.ops.torchish.bitpack_2d(input, kernel) # [K, M]  

def bitunpack_2d(packed: torch.Tensor, N: int, M: int, dtype: torch.dtype = torch.bool, kernel: int = 0) -> torch.Tensor:
    """Unpacks a 2D bit-packed bool-like tensor of uint8 into a proper boolean tensor.

    Args:
        packed: A 2D bitpacked tensor of dtype uint8.
        N: The desired row dimension of the output (first N rows are kept, rest are dropped).
        M: The desired column dimension of the output (first M columns are kept, rest are dropped).
        dtype: The desired datatype of the output tensor, either torch.bool or torch.uint8. Defaults to torch.bool.
        kernel: An optional argument to select the kernel used in computation. Defaults to 0.

    Returns:
        torch.Tensor: A [N, M] boolean tensor.
    """
    return torch.ops.torchish.bitunpack_2d(packed, N, M, dtype, kernel)