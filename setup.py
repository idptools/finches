"""
FINCHES
Package for computing chemical specificity between disordered regions.
"""

import os

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

cython_file = os.path.join("finches", "utils", "matrix_manipulation.pyx")

# pyx_path = "finches/utils/matrix_manipulation.pyx"
print("Absolute path:", os.path.abspath(cython_file))

extensions = [
    Extension(
        "finches.utils.matrix_manipulation",
        [cython_file],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
        ],
    )
]


setup(
    name="finches",
    include_package_data=True,
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
