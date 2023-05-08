#!/bin/sh
rm dsi.cpython*
conda activate testSimu
python setup_cpp.py build_ext --inplace
python setup_cpp.py install 
python test_setup_cpp.py 
 
