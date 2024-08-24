# Modified from OpenMMLab.

import os
from concurrent import futures as futures
from os import path as osp

import mmcv
import numpy as np

from data.scannet import scannet200_splits


# Number images for train & validation
TRAIN_IMAGES = 50
VALID_IMAGES = 100


def linear_sample(arr, num_samples):
    if num_samples >= len(arr):
        return arr
    indices = np.linspace(0, len(arr)-1, num_samples, dtype=int)
    return [arr[i] for i in indices]

def get_class_info(data_name):
    if data_name == "scannet":
        classes = [
            'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
            'bookshelf', 'picture', 'counter', 'desk', 'curtain',
            'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
            'garbagebin'
            ]
        ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
            )
    elif data_name == "scannet200":
        classes = scannet200_splits.VALID_CLASS_IDS_200_VALIDATION
        ids = np.array(scannet200_splits.CLASS_LABELS_200_VALIDATION)
    else:
        raise NotImplementedError(f"Not support data name: {data_name}")
    assert len(classes) == len(ids)
    return classes, ids


class ScanNetData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train', data_name='scannet'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes, self.cat_ids = get_class_info(data_name)
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            cat_id: i
            for i, cat_id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = sorted(mmcv.list_from_file(split_file))
        self.test_mode = (split == 'test')
        self.data_name = data_name

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, f'{self.data_name}_instance_data',
                            f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = osp.join(self.root_dir, f'{self.data_name}_instance_data',
                            f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = osp.join(self.root_dir, f'{self.data_name}_instance_data',
                               f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_frame_data(self, idx):
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        paths = linear_sample(paths, TRAIN_IMAGES if self.split == 'train' else VALID_IMAGES)

        img_paths = paths
        dep_paths = [path.replace('.jpg', '.png') for path in paths]
        exts = [np.loadtxt(osp.join(self.root_dir, path.replace('.jpg', '.txt'))) for path in paths]
        return (img_paths, dep_paths, exts)

    def get_intrinsics(self, idx):
        matrix_file = osp.join(self.root_dir, 'posed_images', idx,
                               'intrinsic.txt')
        mmcv.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, f'{self.data_name}_instance_data',
                                    f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            points.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')

            # update with RGB image paths if exist
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_img_paths, all_dep_paths, all_extrinsics = self.get_frame_data(sample_idx)
                # some poses in ScanNet are invalid
                extrinsics, dep_paths, img_paths = [], [], []
                all_img_paths, all_dep_paths, all_extrinsics = self.get_frame_data(sample_idx)
                for extrinsic, dep_path, img_path in zip(all_extrinsics, all_dep_paths, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        dep_paths.append(dep_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['depth_paths'] = dep_paths
                info['img_paths'] = img_paths

            if not self.test_mode:
                pts_instance_mask_path = osp.join(
                    self.root_dir, f'{self.data_name}_instance_data',
                    f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(
                    self.root_dir, f'{self.data_name}_instance_data',
                    f'{sample_idx}_sem_label.npy')

                pts_instance_mask = np.load(pts_instance_mask_path).astype(
                    np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(
                    np.int64)

                mmcv.mkdir_or_exist(
                    osp.join(self.root_dir, 'instance_mask'))
                mmcv.mkdir_or_exist(
                    osp.join(self.root_dir, 'semantic_mask'))

                pts_instance_mask.tofile(
                    osp.join(self.root_dir, 'instance_mask',
                             f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(
                    osp.join(self.root_dir, 'semantic_mask',
                             f'{sample_idx}.bin'))

                info['pts_instance_mask_path'] = osp.join(
                    'instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join(
                    'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                # box is of shape [k, 6 + class]
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]  # k, 6
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]  # k
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    # default names are given to aligned bbox for compatibility
                    # we also save unaligned bbox info with marked names
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(
                        annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix  # 4x4
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
