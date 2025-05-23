import glob
import os
import os.path as osp
import platform
import sys
import fnmatch
from itertools import product
import torch
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_original
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

__version__ = open('VERSION').read()
URL = 'https://github.com/rmgoetz/torchish'

COMPILED_WITH_CUDA = False
if torch.cuda.is_available():
    COMPILED_WITH_CUDA = CUDA_HOME is not None or torch.version.hip
suffices = ['cpu', 'cuda'] if COMPILED_WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda', 'cpu']
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    suffices = ['cuda']
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    suffices = ['cpu']

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'
WITH_SYMBOLS = os.getenv('WITH_SYMBOLS', '0') == '1'

class build_py(build_py_original):
    '''A hack to work around exclude_package_data not being properly understood by build
    '''
    def find_package_modules(self, package, package_dir):
        excluded_extensions = ["*_nb.py", "*.pyc"]
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat = pattern) for pattern in excluded_extensions)
        ]


def get_extensions():
    '''Manage the C++/CUDA extensions
    '''
    extensions = []

    extensions_dir = osp.join('csrc')

    # main files will be cpp files in the top level of csrc
    main_files = glob.glob(osp.join(extensions_dir, '*.cpp'))

    # remove generated 'hip' files, in case of rebuilds
    main_files = [path for path in main_files if 'hip' not in path]

    for main, suffix in product(main_files, suffices):
        define_macros = [('WITH_PYTHON', None)]
        undef_macros = []

        if sys.platform == 'win32':
            define_macros += [('torchish_EXPORTS', None)]

        extra_compile_args = {'cxx': ['-O2']}

        extra_link_args = [] if WITH_SYMBOLS else ['-s']

        info = parallel_info()
        if ('backend: OpenMP' in info and 'OpenMP not found' not in info
                and sys.platform != 'darwin'):
            extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
            if sys.platform == 'win32':
                extra_compile_args['cxx'] += ['/openmp']
            else:
                extra_compile_args['cxx'] += ['-fopenmp']
        else:
            print('Compiling without OpenMP...')

        # Compile for mac arm64
        if sys.platform == 'darwin':
            extra_compile_args['cxx'] += ['-D_LIBCPP_DISABLE_AVAILABILITY']
            if platform.machine == 'arm64':
                extra_compile_args['cxx'] += ['-arch', 'arm64']
                extra_link_args += ['-arch', 'arm64']

        if suffix == 'cuda':
            define_macros += [('COMPILED_WITH_CUDA', None)]
            nvcc_flags = os.getenv('NVCC_FLAGS', '')
            nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
            nvcc_flags += ['-O2']
            if torch.version.hip:
                # USE_ROCM was added to later versions of PyTorch.
                # Define here to support older PyTorch versions as well:
                define_macros += [('USE_ROCM', None)]
                undef_macros += ['__HIP_NO_HALF_CONVERSIONS__']
            else:
                nvcc_flags += ['--expt-relaxed-constexpr']
            extra_compile_args['nvcc'] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        directory = osp.join(extensions_dir, suffix)
        file_extension = ".cu" if suffix == "cuda" else ".cpp"
        if osp.isdir(directory) and name != "version":
            for file in os.listdir(directory):
                if osp.isfile(osp.join(directory, file)) and file.endswith(file_extension):
                    sources += [osp.join(directory, file)]

        Extension = CppExtension if suffix == 'cpu' else CUDAExtension
        extension = Extension(
            f'torchish._{name}_{suffix}',
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions += [extension]

    return extensions


install_requires = []


# work-around hipify abs paths
include_package_data = True
if torch.cuda.is_available() and torch.version.hip:
    include_package_data = False

setup(
    name='torchish',
    version=__version__,
    description='Ecclectic PyTorch Extension Library',
    author='Ryan Goetz',
    url=URL,
    python_requires='>=3.8',
    install_requires=install_requires,
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_py' : build_py,
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(exclude = ["torchish.jit.*"]),
    include_package_data=include_package_data,
)