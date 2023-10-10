#!/bin/sh
conda activate IEBCS
python -m build .
pip install .
python test_setup_cpp.py 
 
