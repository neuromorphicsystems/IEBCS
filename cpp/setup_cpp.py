from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("dsi",
                  sources=["simu_cpp.cpp"],
                  language = "c++",
                  include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
                  extra_compile_args=['-std=c++17'])

setup(
    name='dsi',
    version='1.0',
    description='DVS simu',
    ext_modules=[c_ext],
)
