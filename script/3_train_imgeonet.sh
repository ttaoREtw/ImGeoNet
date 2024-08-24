#!/bin/bash

cd mmdetection3d

# Train ImGeoNet on ScanNet
# For the baseline ImVoxelNet, use the config `configs/imgeonet/imvoxelnet_scannet.py`
# Multi-GPU
# bash tools/dist_train.sh configs/imgeonet/imgeonet_scannet.py ${NUM_GPU} --work-dir work_dir/imgeonet_scannet
# Single GPU
python tools/train.py configs/imgeonet/imgeonet_scannet.py --work-dir work_dir/imgeonet_scannet

# Train ImGeoNet on ScanNet200
# For the baseline ImVoxelNet, use the config `configs/imgeonet/imvoxelnet_scannet200_vx808032.py`
# Multi-GPU
# bash tools/dist_train.sh configs/imgeonet/imgeonet_scannet200_vx808032.py ${NUM_GPU} --work-dir work_dir/imgeonet_scannet200_vx808032
# Single GPU
python tools/train.py configs/imgeonet/imgeonet_scannet200_vx808032.py --work-dir work_dir/imgeonet_scannet200_vx808032

# Train ImGeoNet on ARKitScenes
# For the baseline ImVoxelNet, use the config `configs/imgeonet/imvoxelnet_arkit.py`
# Multi-GPU
# bash tools/dist_train.sh configs/imgeonet/imgeonet_arkit.py ${NUM_GPU} --work-dir work_dir/imgeonet_arkit
# Single GPU:
python tools/train.py configs/imgeonet/imgeonet_arkit.py --work-dir work_dir/imgeonet_arkit

