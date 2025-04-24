from typing import Union
import torch

__all__ = [
    'interpret_as_tensor',
    'left_broadcast_shape',
    'consecutive_index_remap'
]

def interpret_as_tensor(
    X: Union[int, float, tuple, list, torch.Tensor],
    target_shape: Union[tuple, torch.Tensor],
    contiguous: bool = False
) -> torch.Tensor:
    """Attempts to interpret/convert an input to a torch Tensor of a specified shape. The general paradigm is to work 
    from trailing dimensions to leading dimensions of the input X. As an example, for inputs such that:

        X.shape -> (3, 10, 2)
        target_shape -> (3, 10, -1, 2)

    the returned tensor will have the shape (3, 10, 10, 2), whereas for inputs with:

        X.shape -> (3, 10, 2)
        target_shape -> (3, -1, -1, 2)

    the returned tensor will have the shape (3, 3, 10, 2).


    Args:
        X: An object to be interpreted as a tensor.
        target_shape: The target shape of the return tensor
        contiguous: If True, the output is guaranteed to be contiguous. Defaults to False.

    Returns:
        torch.Tensor: 
    """

    # If the target shape was accidentally given as an integer instead of a tuple (remember commas!)
    if isinstance(target_shape, int):
        target_shape = torch.tensor([target_shape])   # [1]

    # Convert the target shape to a tensor if it isn't already
    if not isinstance(target_shape, torch.Tensor):
        target_shape = torch.tensor(target_shape)   # [target ndim]

    # The number of target dimensions
    target_ndim = len(target_shape)

    # Handle the special case where we actually want a 0-dimensional tensor
    if len(target_shape) == 1 and target_shape.item() == 0:
        target_ndim = 0

    # If X is a list or tuple, attempt to convert to tensor
    if isinstance(X, (list, tuple)):
        try:
            X = torch.tensor(X)
        except TypeError:
            raise Exception(f"Input {X} cannot be converted to a torch tensor.")
        
    # If X is a number, convert to tensor
    if isinstance(X, (float, int)):
        if target_ndim == 0:
            converted_tensor = torch.tensor(X)   # [0]
        else:
            target_shape = torch.where(target_shape == -1, 1, target_shape)   # [target ndim]
            converted_tensor = torch.tensor([X]).expand(*target_shape)   # [*target shape]

    # If X was not given as a number (i.e. list, tuple, or tensor)
    elif isinstance(X, torch.Tensor):

        # The number of dimensions of the input
        source_ndim = X.ndim

        # We now consider three cases:
        #   1) The source ndim exceeds that of the target ndim. For this we raise an exception except when the product of
        #      the source dimensions is one.
        if source_ndim > target_ndim:
            if torch.tensor(X.size()).prod() == 1:
                if target_ndim == 0:
                    converted_tensor = torch.tensor(X.item())   # [0]
                else:
                    converted_tensor = torch.tensor([X.item()]).expand(*target_shape)   # [*target shape]
            else:
                raise Exception(f"Source with {source_ndim} dimensions cannot be converted to a tensor of {target_ndim} dimensions.")

        #   2) The source ndim is equal to the target ndim. For this case we attempt to expand X into the target shape.
        elif source_ndim == target_ndim:
            if target_ndim == 0:
                converted_tensor = X   # [*target shape]
            else:
                converted_tensor = X.expand(*target_shape)   # [*target shape]
            
        #   3) The source ndim is less than the target ndim. For this we consider two subcases:
        #       a) The source ndim is 0
        elif source_ndim == 0:
            target_shape = torch.where(target_shape == -1, 1, target_shape)   # [target ndim]
            converted_tensor = X.expand(*target_shape)   # [*target shape]

        #       3b) The source ndim is not 0
        else:

            # Costruct a matrix which tells us which dimension sizes of the source and target shapes match
            target_column = target_shape.view(-1, 1)   # [target ndim, 1]
            source_row = torch.tensor(X.shape).view(1, -1)   # [1, source ndim]
            comparison_matrix = (target_column == source_row) + (target_column == -1)   # [target ndim, source ndim]

            # Go through the matrix line by line and reduce it according to the last indices for both the target and source
            max_idx = source_ndim - 1
            for target_idx in range(target_ndim - 1, -1, -1):
                row_zero = torch.zeros((source_ndim), dtype = bool)   # [source ndim]
                row_zero[max_idx] = True
                comparison_matrix[target_idx, :] *= row_zero
                if comparison_matrix[target_idx, max_idx]:
                    col_zero = torch.zeros((target_ndim), dtype = bool)   # [target ndim]
                    col_zero[target_idx] = True
                    comparison_matrix[:, max_idx] *= col_zero
                    max_idx -= 1
            
            # If our checksum doesn't work out we can't convert to the target shape
            if comparison_matrix.sum() != source_ndim:
                raise Exception(f"Input of shape {X.shape} cannot be converted to tensor of shape {target_shape}.")
            
            # Determine the intermediate shape to pass to reshape()
            view_shape = (comparison_matrix * source_row).sum(dim = -1)   # [target ndim]
            view_shape = torch.where(view_shape == 0, 1, view_shape)   # [target ndim]

            # Reshape and expand to the target shape
            converted_tensor = X.reshape(*view_shape).expand(*target_shape)   # [*target shape]

    # If X is not a supported input type
    else:
        raise TypeError(f"Invalid input type for `X`: {type(X)}")
    
    # Make contiguous if desired
    converted_tensor = converted_tensor.contiguous() if contiguous else converted_tensor   # [*target shape]

    return converted_tensor   # [*target shape]


def left_broadcast_shape(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """Determine the shape tensor for the broadcast of two inputs under the convention that dimensions are added "on the
    left" to make the number of dimensions agree.

    Args:
        x (torch.Tensor): A [*] shaped tensor
        y (torch.Tensor): A [**] shaped tensor

    Returns:
        torch.Tensor: A [ndims] tensor
    """

    xsize = torch.as_tensor(x.size())   # [x ndims]
    ysize = torch.as_tensor(y.size())   # [y ndims]

    dim_diff = x.ndim - y.ndim

    # If x has fewer dimensions than y
    if dim_diff < 0:
        xsize = torch.cat([torch.tensor((-dim_diff)*[1]), xsize])   # [ndims]
    # If y has fewer dimensions than x
    elif dim_diff > 0:
        ysize = torch.cat([torch.tensor(dim_diff*[1]), ysize])   # [ndims]

    broadcast_shape = torch.stack([xsize, ysize], dim = 0).max(dim = 0)[0]   # [ndims]

    return broadcast_shape   # [ndims]


def consecutive_index_remap(X: torch.Tensor) -> torch.Tensor:
    """Maps a tensor of integers to a corresponding tensor of consecutive integers beginning at zero, preserving the 
    relative ordering of the input across the last dimenion. As an example, the tensor:

    [[-1,  2, 27,  4,  4],
     [-3, -3,  2, 80,  6],
     [ 0,  0,  0,  0,  1],
     [ 6, 32,  1,  3,  7]]

    will be mapped to:

    [[ 0,  1,  3,  2,  2],
     [ 0,  0,  1,  3,  2],
     [ 0,  0,  0,  0,  1],
     [ 2,  4,  0,  1,  3]]

    Args:
        X (torch.Tensor): A [*, P] tensor of integer types, where * denotes arbitrary leading dimensions.

    Returns:
        A [*, P] tensor of remapped integers.
    """

    # Flatten all leading dimensions of X
    input_shape = X.shape
    Xf = X.reshape(-1, input_shape[-1])   # [N, P]

    # Shift values so that the lowest value along any last dimension is zero
    Xshift = Xf - Xf.min(dim = -1, keepdim = True).values   # [N, P]

    # Determine offsets
    offsets = 1 + Xshift.max(dim = -1, keepdim = True).values   # [N, 1]
    offsets = offsets.roll(shifts = (1,), dims = 0)   # [N, 1]
    offsets = offsets.cumsum(dim = 0)

    Xshift = Xshift + offsets   # [N, P]

    # Find the unique inverse indices and reshape back to input shape
    Xconsecutive: torch.Tensor = Xshift.view(-1).unique(return_inverse = True)[1].view(input_shape)   # [*, P]

    # Subtract away minimum so as to start values at 0
    Xconsecutive = Xconsecutive - Xconsecutive.min(dim = -1, keepdim = True).values   # [*, P]

    return Xconsecutive   # [*, P]
