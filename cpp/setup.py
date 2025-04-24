from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


c_ext = Extension("dsi",
                  sources=["simu.cpp"],
                  language="c++",
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['/std:c++17'])

setup(
    name='dsi',
    version='1.1',
    description='DVS simu',
    ext_modules=cythonize([c_ext])
)
