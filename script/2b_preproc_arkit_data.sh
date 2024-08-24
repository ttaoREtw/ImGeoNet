#!/bin/bash

ARKIT_ROOT="data/arkit/downloaded/3dod"

TRAIN_DIR="${ARKIT_ROOT}/Training"
VAL_DIR="${ARKIT_ROOT}/Validation"

TRAIN_SAMPLES=50
VAL_SAMPLES=100
OUT_DIR="data/arkit/processed/3dod"

python data/arkit/process.py --train-dir ${TRAIN_DIR} \
                             --train-num-samples ${TRAIN_SAMPLES} \
                             --val-dir ${VAL_DIR} \
                             --val-num-samples ${VAL_SAMPLES} \
                             --output-dir ${OUT_DIR} \
                             --nproc 8
