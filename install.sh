#!/bin/bash
work_path=$(dirname $(readlink -f $0))
cd ${work_path}/pytorch_bm3d/
python setup.py develop | grep "error"
cd ../
python setup.py develop