#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
python data/create_data.py --root-path data/scannet --out-dir data/scannet --tag scannet
python data/create_data.py --root-path data/scannet --out-dir data/scannet --tag scannet200
