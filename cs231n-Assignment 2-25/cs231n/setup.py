from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import shutil
import sys


if sys.platform == "win32" and shutil.which("gcc"):
    has_build_ext = "build_ext" in sys.argv
    has_compiler = any(arg.startswith("--compiler") for arg in sys.argv)
    if has_build_ext and not has_compiler:
        sys.argv.append("--compiler=mingw32")

extensions = [
    Extension(
        "im2col_cython", ["im2col_cython.pyx"], include_dirs=[numpy.get_include()]
    ),
]

setup(ext_modules=cythonize(extensions),)
