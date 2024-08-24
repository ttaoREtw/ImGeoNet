#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
python data/create_data.py --root-path data/arkit --out-dir data/arkit --tag arkit
