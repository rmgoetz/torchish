# torchish
a repo for useful PyTorch based functions/extensions


## Installation
To build from source with CUDA support you will need to ensure that your `$PATH` environment variable contains the path to the binaries file containing the CUDA compiler (NVCC), and your `$CPATH` environment variable points to the CUDA include headers folder. Then one can clone the repository and simply install with pip:
```
pip install .
```

For editable installs, it is advisable to first manually install the `setuptools` and `torch` dependencies into your active python environment, then remove these from the `pyproject.toml` requirements list, and run:
```shell
pip install --no-build-isolation -e .
```