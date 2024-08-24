# Modified from OpenMMLab.

import argparse
import os
from os import path as osp

import mmcv
import numpy as np

from data.scannet_data_utils import ScanNetData
from data.arkit_data_utils import ARKitData


def data_prep(root_path, tag, save_path, workers):
    assert os.path.exists(root_path)
    assert tag in ['scannet', 'scannet200', 'arkit'], \
        f'unsupported indoor dataset {tag}'
    save_path = root_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    train_filename = os.path.join(save_path, f'{tag}_infos_train.pkl')
    val_filename = os.path.join(save_path, f'{tag}_infos_val.pkl')
    if tag in ['scannet', 'scannet200']:
        train_dataset = ScanNetData(root_path=root_path, split='train', data_name=tag)
        val_dataset = ScanNetData(root_path=root_path, split='val', data_name=tag)
    elif tag == 'arkit':
        train_dataset = ARKitData(root_path=root_path, split='train')
        val_dataset = ARKitData(root_path=root_path, split='val')
    infos_train = train_dataset.get_infos(
        num_workers=workers, has_label=True)
    mmcv.dump(infos_train, train_filename, 'pkl')
    print(f'{tag} info train file is saved to {train_filename}')

    infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
    mmcv.dump(infos_val, val_filename, 'pkl')
    print(f'{tag} info val file is saved to {val_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/scannet',
        help='specify the root path of dataset')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/scannet',
        required=False,
        help='name of info pkl')
    parser.add_argument('--tag', type=str, default='scannet')
    parser.add_argument(
        '--workers', type=int, default=4, help='number of threads to be used')
    args = parser.parse_args()

    data_prep(
        root_path=args.root_path,
        tag=args.tag,
        save_path=args.out_dir,
        workers=args.workers)
