#!/bin/bash

pip install Cython==0.29.35
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html 
pip install mmdet==2.10.0
pip install numpy==1.23.0

cd mmdetection3d; pip install --no-cache-dir -e .
cd mmdet3d/ops/rotated_iou/cuda_op; python setup.py install

pip uninstall pycocotools --no-cache-dir -y
pip install mmpycocotools==12.0.3 --no-cache-dir --force --no-deps
pip install numba==0.48.0
pip install yapf==0.40.1
pip install protobuf==3.20.3

# Used for arkit preprocess
pip install open3d==0.18.0 imageio==2.35.1 loguru==0.7.2
