#!/bin/sh
rm dsi.cpython-37m-x86_64-linux-gnu.so
python setup_cpp.py build_ext --inplace
python setup_cpp.py install
python test_setup_cpp.py 
 
