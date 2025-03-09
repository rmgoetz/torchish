import torch
from torchish.utils import left_broadcast_shape, interpret_as_tensor


def outer_prod(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Computes the outer product of vectors in a batched fashion.

    Args:
        x (torch.Tensor): A [*, D1] tensor of D1-dimensional vectors, where * is an arbitrary sequence of leading dimensions.
        y (torch.Tensor): A [**, D2] tensor of D2-dimensional vectors, where ** is an arbitrary sequence of leading dimensions.

    Returns:
        torch.Tensor: A [***, D1, D2] tensor of the outer product of x with y, where *** is the resultant shape of left-broadcasting
                      the shapes * and **.
    """

    D1 = x.shape[-1]
    D2 = y.shape[-1]

    # Determine the broadcast shape of x with y
    broadcast_shape = left_broadcast_shape(
        x = x.detach()[..., 0:1],
        y = y.detach()[..., 0:1]
    )   # [ndims]
    broadcast_shape[-1] = -1

    # Reshape the inputs and make contiguous in memory
    c1 = interpret_as_tensor(X = x, target_shape = broadcast_shape, contiguous = True)   # [***, D1]
    c2 = interpret_as_tensor(X = y, target_shape = broadcast_shape, contiguous = True)   # [***, D2]

    # Flatten to build the outer product
    c1 = c1.view(-1, D1)   # [N, D1]
    c2 = c2.view(-1, D2)   # [N, D1]
    outer = torch.einsum('bi, bj -> bij', (c1, c2))   # [B, D1, D2]

    # Unflatten back to the broadcast shape
    outer = outer.reshape(*broadcast_shape[:-1], D1, D2)   # [***, D1, D2]

    return outer   # [***, D1, D2]


def cross_prod(x: torch.Tensor) -> torch.Tensor:
    """Computed the generalized cross product for dimensions greate than or equal to 2.

    Args:
        x: A [*, D - 1, D] tensor of (D - 1)-many D-dimensional vectors for which to evaluate the cross product. Here *
           denotes an arbitrary sequence of leading dimensions.

    Returns:
        torch.Tensor: A [*, D] tensor of D-dimensional cross product results.
    """

    D = x.shape[-1]

    # Add a dummy for convenience if it doesn't already exist
    if x.ndim == 1:
        x = x[None, :]   # [1, D]

    # Construct the tensor of minors
    minors = torch.stack([x[..., [i for i in range(D) if i != d]]
                          for d in range(D)],
                          dim = -3)   # [*, D, D - 1, D - 1]
    
    # The cross product (up to sign) is the determinant of the minors
    cross = torch.linalg.det(minors)   # [*, D]

    # Account for the sign to retrieve the appropriate cross product
    signs = torch.tensor([(-1)**(D + i - 1) for i in range(D)], device = x.device)   # [D]
    signs = interpret_as_tensor(X = signs, target_shape = (*x.shape[:-2], -1))   # [*, D]
    cross = signs * cross   # [*, D
    
    return cross   # [*, D]