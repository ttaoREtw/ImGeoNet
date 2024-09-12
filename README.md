# ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection

Paper link: https://arxiv.org/abs/2308.09098  
Project page: https://ttaoretw.github.io/imgeonet/  

## Performance
| Dataset     | mAP@0.25 | mAP@0.5  | Log |
| --------    | :------: | :------: | --- |
| ScanNet     | 54.57    | 28.94    | [link](logs/scannet.txt) |
| ScanNet200  | 22.38    | 9.67     | [link](logs/scannet200.txt) |
| ARKitScenes | 59.82    | 42.76    | [link](logs/arkit.txt) |

> Performance may vary slightly depending on the number of GPUs.


## Environment

```sh
# Create conda virtual environment
conda create -n imgeonet python=3.8
conda activate imgeonet

# Clone repo
git clone https://github.com/ttaoREtw/ImGeoNet.git
cd ImGeoNet

# Setup virtual environment
bash script/0_install_env.sh
```


## Data
### ScanNet & ScanNet200
Download [ScanNet](http://www.scan-net.org/) data and link `scans` folder under `data/scannet`, then run

```sh
# Warning: this step requires a lot of disk space
# Extract frame data: rgb, depth, intrinsic, pose, axis matrix
bash script/1a_extract_scannet_posed_data.sh

# Process scannet as in VoteNet
bash script/1b_preproc_scannet_data.sh

# Convert to mmdet3d's format
bash script/1c_convert_scannet_data.sh
```

### ARKitScenes
```sh
# Download ARKitScenes data - 3D detection part
bash script/2a_download_arkit.sh

# Extract frame data: rgb, depth, intrinsic, pose
bash script/2b_preproc_arkit_data.sh

# Convert to mmdet3d's format
bash script/2c_convert_arkit_data.sh

```


## Training
```sh
# Train on ScanNet, ScanNet200, and ARKitScenes
bash script/3_train_imgeonet.sh
```

## Test
```sh
cd mmdetection3d
# config in `configs/`
# checkpoint in `work_dir/.../latest.pth`
python tools/test.py $config $checkpoint --eval mAP
```

## Citation
```
@inproceedings{tu2023imgeonet,
  title     = {ImGeoNet: Image-induced Geometry-aware Voxel Representation for Multi-view 3D Object Detection},
  author    = {Tu, Tao and Chuang, Shun-Po and Liu, Yu-Lun and Sun, Cheng and Zhang, Ke and Roy, Donna and Kuo, Cheng-Hao and Sun, Min},
  booktitle = {Proceedings of the IEEE international conference on computer vision},
  year      = {2023},
}
```

## Acknowledgement
This project is built upon various open-source projects.
If your work involves components related to these projects, you may have to consider citing them.  
* [ARKitScenes](https://github.com/apple/ARKitScenes)
* [ScanNet](https://github.com/ScanNet/ScanNet)
* [VoteNet](https://github.com/facebookresearch/votenet)
* [ImVoxelNet](https://github.com/SamsungLabs/imvoxelnet)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [Atlas](https://github.com/magicleap/Atlas)
