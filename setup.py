"""
sparrow
Next generation package for sequence parameter calculation
"""
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy


extensions = [
    Extension(
        "finches.utils.matrix_manipulation",
        ["finches/utils/matrix_manipulation.pyx"],
        include_dirs=[numpy.get_include()], 
    )]


setup(
    name='finches',
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
    zip_safe=False,
)
    
