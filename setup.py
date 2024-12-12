"""
FINCHES
Package for computing chemical specificity between disordered regions.
"""
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
import os


cython_file = os.path.join("finches", "utils", "matrix_manipulation.pyx")

#pyx_path = "finches/utils/matrix_manipulation.pyx"
print("Absolute path:", os.path.abspath(cython_file))

extensions = [
    Extension(
        "finches.utils.matrix_manipulation",
        [cython_file],
        include_dirs=[numpy.get_include()], 
    )]


setup(
    name='finches',
    include_package_data=True,
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
    zip_safe=False,
)
    
