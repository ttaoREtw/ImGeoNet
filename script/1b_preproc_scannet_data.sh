#!/bin/bash

cd data/scannet

# For ScanNet
python batch_load_scannet_data.py --max_num_point 50000 \
                                  --output_folder ./scannet_instance_data

# For ScanNet200
python batch_load_scannet_data.py --max_num_point 50000 \
                                  --data_name scannet200 \
                                  --output_folder ./scannet200_instance_data
