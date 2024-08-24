#!/bin/bash

cd data/arkit

python download_data.py 3dod --split Training --download_dir downloaded --video_id_csv 3dod_train_val_splits.csv
python download_data.py 3dod --split Validation --download_dir downloaded --video_id_csv 3dod_train_val_splits.csv
