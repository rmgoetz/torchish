from typing import List, Tuple
import os
import re
import fnmatch
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig

VERSION = open('VERSION').read()
exluded = ['*_nb.py', '*.pyc']
ABS_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT_DIR = os.path.normpath(os.path.join(ABS_DIR, "torchish/ext"))

class build_py(build_py_orig):
    """A hack to work around the fact that exclude package data is not understood properly by build.
    """
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in exluded)
        ]

def ext_data_files() -> List[Tuple[str, List[str]]]:
    """Generates a data file list for the extensions subdirectory.
    """
    ext_dirs = [EXT_ROOT_DIR] + list_subdirectories(EXT_ROOT_DIR)
    ext_import_names = convert_to_module_import(ext_dirs)
    data_files = []
    for dirpath, imp_name in zip(ext_dirs, ext_import_names):
        cpp_cuda_files = grab_cpp_cuda_files(dirpath)
        data_files.append((imp_name, cpp_cuda_files))
    return data_files

def list_subdirectories(root_dir: str) -> List[str]:
    """List all subdirectories in a root directory
    """
    subdirectories = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            subdirectories.append(os.path.join(dirpath, dirname))
    return subdirectories

def grab_cpp_cuda_files(dirpath: str) -> List[str]:
    """Lists all C++/CUDA related files in a directory
    """
    # glob is straight trash, so please excuse this
    file_extensions = ["*.cpp", "*.cu", "*.hpp", "*.cuh"]
    file_list = []
    for ext in file_extensions:
        files = list(Path(dirpath).glob(ext))
        files = [os.path.normpath(os.path.join(str(f))) for f in files]
        file_list += files
    return file_list

def convert_to_module_import(paths: List[str]) -> List[str]:
    """Converts an absolute directory path to an import path for this torchish module.
    """
    period_ABS_DIR = ABS_DIR.replace("\\", ".").replace("/", ".")
    import_names = []
    for path in paths:
        period_path = path.replace("\\", ".").replace("/", ".")
        imp_name = re.sub(f'^{period_ABS_DIR}', '', period_path)[1:]
        import_names.append(imp_name)
    return import_names


setup(
    name = "torchish",
    author = "Ryan Goetz",
    version = VERSION,
    packages = find_packages(exclude = []),
    include_package_data = True,
    cmdclass = {"build_py" : build_py},
    data_files = ext_data_files()
)