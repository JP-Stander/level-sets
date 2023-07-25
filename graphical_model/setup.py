from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "spatio_distance", 
        ["spatio_distance.pyx"],
        include_dirs=[numpy.get_include()]  
    )
]

setup(
    ext_modules = cythonize(extensions)
)
