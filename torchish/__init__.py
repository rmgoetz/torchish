import importlib
import os
import os.path as osp
from pathlib import Path
import torch

version_file = os.path.join(Path(__file__).parents[1], "VERSION")
__version__ = open(version_file).read()

for library in ['_version', '_torchish']:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    elif os.getenv('BUILD_DOCS', '0') != '1':  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")
    else:  # pragma: no cover
        from .ops_placeholders import cuda_version_placeholder
        torch.ops.torchish.cuda_version = cuda_version_placeholder

        from .ops_placeholders import raycast_placeholder, raycast_nb_placeholder
        torch.ops.torchish.raycast = raycast_placeholder
        torch.ops.torchish.raycast_nb = raycast_nb_placeholder

        from .ops_placeholders import bitpack_2d_placeholder
        torch.ops.torchish.bitpack_2d = bitpack_2d_placeholder


cuda_version = torch.ops.torchish.cuda_version()
is_not_hip = torch.version.hip is None
is_cuda = torch.version.cuda is not None
if is_not_hip and is_cuda and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torchish were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torchish has CUDA version '
            f'{major}.{minor}. Please reinstall the torchish that '
            f'matches your PyTorch install.')

from .utils import __all__ as utils_all
from .linalg import __all__ as linalg_all
from .raycast import __all__ as raycast_all
from .binmatmul import __all__ as binmatmul_all
from .utils import *
from .linalg import *
from .raycast import *
from .binmatmul import *

__all__ = utils_all + linalg_all + raycast_all + binmatmul_all
