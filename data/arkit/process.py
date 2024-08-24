# Author: Tao Tu
# Modified from ARKitScenes: https://github.com/apple/ARKitScenes

import argparse
import pathlib
import pickle
import shutil
import sys
import multiprocessing
import json

import cv2
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

import point_utils

from loguru import logger


CLASS_NAMES = [
    "cabinet",
    "refrigerator",
    "shelf",
    "stove",
    "bed",  # 0..5
    "sink",
    "washer",
    "toilet",
    "bathtub",
    "oven",  # 5..10
    "dishwasher",
    "fireplace",
    "stool",
    "chair",
    "table",  # 10..15
    "tv_monitor",
    "sofa",  # 15..17
]


def is_equal(*arg):
    return len(set([x for x in arg])) == 1


def is_sync(timestamps1, timestamps2, tol=None):
    is_successful = True
    if len(timestamps1) != len(timestamps2):
        is_successful = False
    else:
        check_fn = (
            lambda x, y: x == y
            if tol is None
            else lambda x, y: abs(float(x) - float(y)) <= tol
        )
        for ts1, ts2 in zip(timestamps1, timestamps2):
            if not check_fn(ts1, ts2):
                is_successful = False
                break
    return is_successful


def compute_yaw_from_raw(size, center, rotate_mat):
    # The rotation can be recovered by R.from_euler('z', yaw).as_matrix()
    hf_dx, hf_dy, hf_dz = size / 2

    pts = np.array([[hf_dx, hf_dy, hf_dz], [-hf_dx, hf_dy, hf_dz]])
    pts = pts @ rotate_mat
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts[:, 2] += center[2]

    dx = pts[0, 0] - pts[1, 0]
    dy = pts[0, 1] - pts[1, 1]
    yaw_angle = np.arctan2(dy, dx)
    return yaw_angle


def compute_cam2world_matrix(raw_pose):
    rotate_vec = np.array(raw_pose[:3])
    trans_vec = np.array(raw_pose[3:])
    rotate_mat, _ = cv2.Rodrigues(rotate_vec)

    extrinsic = np.eye(4, 4)
    extrinsic[:3, :3] = rotate_mat
    extrinsic[:3, -1] = trans_vec

    pose = np.linalg.inv(extrinsic)
    return pose


def linear_sample(arr, num_samples, return_indices=False):
    num_samples = min(num_samples, len(arr))
    indices = np.linspace(0, len(arr) - 1, num_samples, dtype=int)
    sampled_arr = [arr[i] for i in indices]
    if return_indices:
        return sampled_arr, indices
    else:
        return sampled_arr


def generate_point_clouds(color_paths, depth_paths, poses, intrinsic, chunk=50):
    depth_shift = 1000.  # mm to meter
    num_frames = len(poses)
    pts_all = []
    rgb_all = []

    num_iters = num_frames // chunk
    num_remain = num_frames % chunk
    for i in range(num_iters):
        images = []
        depths = []
        for j in range(chunk):
            images.append(imageio.imread(color_paths[i * chunk + j]))
            depths.append(imageio.imread(depth_paths[i * chunk + j]) / depth_shift)
        poses_this_iter = poses[i * chunk:(i+1) * chunk]
        pts, rgb = point_utils.gen_points(images, depths, poses_this_iter, intrinsic)
        pts_all.append(pts)
        rgb_all.append(rgb)
    
    if num_remain > 0:
        images = []
        depths = []
        for color_path, depth_path in zip(color_paths[-num_remain:], depth_paths[-num_remain:]):
            images.append(imageio.imread(color_path))
            depths.append(imageio.imread(depth_path) / depth_shift)
        poses_this_iter = poses[-num_remain:]
        pts, rgb = point_utils.gen_points(images, depths, poses_this_iter, intrinsic)
        pts_all.append(pts)
        rgb_all.append(rgb)

    pts_all = np.concatenate(pts_all, axis=0)
    rgb_all = np.concatenate(rgb_all, axis=0)

    logger.info(f"Number of points from {num_frames} frames: {len(pts_all)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(rgb_all / 255.0)

    voxel_size = 0.02
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    logger.info(f"Downsampled to {len(pcd.points)}")
    return pcd


class SceneConfig:
    def __init__(self, scene_path):
        if not isinstance(scene_path, pathlib.Path):
            scene_path = pathlib.Path(scene_path)
        self.scene_path = scene_path

    @property
    def video_id(self):
        return self.scene_path.name

    @property
    def annotation_path(self):
        return self.scene_path / f"{self.video_id}_3dod_annotation.json"

    @property
    def mesh_path(self):
        return self.scene_path / f"{self.video_id}_3dod_mesh.json"

    @property
    def pose_path(self):
        return self.scene_path / f"{self.video_id}_frames" / "lowres_wide.traj"

    @property
    def colors_dir(self):
        return self.scene_path / f"{self.video_id}_frames" / "lowres_wide"

    @property
    def depths_dir(self):
        return self.scene_path / f"{self.video_id}_frames" / "lowres_depth"

    @property
    def intrinsics_dir(self):
        return self.scene_path / f"{self.video_id}_frames" / "lowres_wide_intrinsics"


class SceneProcessor:
    def __init__(self, data_path, pcd_mode=False):
        self.cfg = SceneConfig(data_path)
        self.data = None
        self.pcd_mode = pcd_mode

    def load(self, num_samples):
        ts_poses, poses = self._load_poses()
        ts_colors, color_paths = self._load_color_paths()
        ts_depths, depth_paths = self._load_depth_paths()
        intrinsic, _, _ = self._load_intrinsic()
        annotations = self._load_ann()

        assert is_equal(len(poses), len(color_paths), len(depth_paths))
        assert is_sync(ts_colors, ts_depths)
        assert is_sync(ts_colors, ts_poses, tol=2e-3)

        poses, indices = linear_sample(poses, num_samples, return_indices=True)
        color_paths = [color_paths[i] for i in indices]
        depth_paths = [depth_paths[i] for i in indices]

        if self.pcd_mode:
            pcd = generate_point_clouds(color_paths, depth_paths, poses, intrinsic)
            self.data = pcd, annotations
        else:
            self.data = poses, color_paths, depth_paths, intrinsic, annotations
        return self

    def save(self, output_dir):
        assert self.data is not None

        output_dir.mkdir(exist_ok=True, parents=True)

        if self.pcd_mode:
            pcd, annotations = self.data
            # o3d.io.write_point_cloud(str(output_dir / "points.ply"), pcd)
            pts = np.asarray(pcd.points).astype(np.float32)
            rgb = np.asarray(pcd.colors).astype(np.float32)
            to_save = np.concatenate([pts, rgb], axis=-1)
            to_save.tofile(output_dir / "points.bin")
        else:
            poses, color_paths, depth_paths, intrinsic, annotations = self.data
            for pose, color_path, depth_path in zip(poses, color_paths, depth_paths):
                frame_name = color_path.stem

                np.save(output_dir / (frame_name + "_pose.npy"), pose)
                shutil.copyfile(color_path, output_dir / (frame_name + "_color.png"))
                shutil.copyfile(depth_path, output_dir / (frame_name + "_depth.png"))

            np.save(output_dir / "intrinsic.npy", intrinsic)
        
        with open(output_dir / "annotations.pkl", "wb") as f:
            pickle.dump(annotations, f)

    def _load_ann(self):
        path = self.cfg.annotation_path
        with open(path, "r") as f:
            ann = json.load(f)

        skipped = ann["skipped"]
        if skipped:
            logger.error(f"Skip {path}")
            return None, None

        label_list = []
        box3d_list = []
        for data in ann["data"]:
            _label = data["label"]
            _special_symbols = [" ", "-", "/"]
            for ch in _special_symbols:
                label = _label.replace(ch, "_")
            if label not in CLASS_NAMES:
                logger.warning(f"Ignore unknown category: {label}")
                continue
            label_list.append(label)

            box_data = data["segments"]["obbAligned"]
            rotate_mat = np.array(box_data["normalizedAxes"]).reshape(3, 3)
            center = np.array(box_data["centroid"]).reshape(3)
            size = np.array(box_data["axesLengths"]).reshape(3)

            yaw_angle = compute_yaw_from_raw(size, center, rotate_mat)
            box3d = np.concatenate([center, size, np.array([yaw_angle])], axis=0)
            box3d_list.append(box3d)
        return {"label": label_list, "bbox": box3d_list}

    def _load_poses(self):
        path = self.cfg.pose_path
        with open(path) as f:
            raw = f.readlines()

        timestamps = []
        poses = []
        for line in raw:
            parts = line.strip().split(" ")
            timestamp = parts[0]
            pose_raw = parts[1:]
            pose_raw = list(map(float, pose_raw))
            pose_mat = compute_cam2world_matrix(pose_raw)
            timestamps.append(timestamp)
            poses.append(pose_mat)
        return timestamps, poses

    def _load_intrinsic(self):
        paths = sorted(list(self.cfg.intrinsics_dir.glob("*.pincam")))

        data = None
        for path in paths:
            _data = np.loadtxt(path)
            if data is None:
                data = _data
            else:
                assert all(
                    [x == y for x, y in zip(data, _data)]
                ), f"1st: {data}, curr: {_data}"

        width, height, fx, fy, cx, cy = data
        intrisic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrisic, int(height), int(width)

    def _load_color_paths(self):
        paths = sorted(list(self.cfg.colors_dir.glob("*.png")))
        timestamps = [path.stem.split("_")[-1] for path in paths]
        return timestamps, paths

    def _load_depth_paths(self):
        paths = sorted(list(self.cfg.depths_dir.glob("*.png")))
        timestamps = [path.stem.split("_")[-1] for path in paths]
        return timestamps, paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=pathlib.Path,
        help="Path to training directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=pathlib.Path,
        help="Path to validation directory.",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=50,
        help="Number images for training.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=100,
        help="Number images for testing.",
    )
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--output-dir", type=pathlib.Path, help="Output Path.")
    parser.add_argument(
        "--log", type=pathlib.Path, default="run_process.log", help="Log path."
    )
    parser.add_argument(
        "--pcd-mode",
        action="store_true",
        help="Point cloud mode.",
    )
    return parser.parse_args()


def setup_logger(log_path):
    logger.remove()
    logger.add(log_path)
    logger.add(sys.stdout)


def worker(scene_dir, num_samples, output_dir, pcd_mode):
    logger.info(f"{scene_dir} starts.")
    processor = SceneProcessor(scene_dir, pcd_mode)
    processor.load(num_samples)
    processor.save(output_dir)
    logger.success(f"{scene_dir} done.")


def main(args):
    works = []
    for scene_dir in sorted(list(args.train_dir.iterdir())):
        works.append(
            (
                scene_dir,
                args.train_num_samples,
                args.output_dir / "training" / scene_dir.name,
                args.pcd_mode,
            )
        )

    for scene_dir in sorted(list(args.val_dir.iterdir())):
        works.append(
            (
                scene_dir,
                args.val_num_samples,
                args.output_dir / "validation" / scene_dir.name,
                args.pcd_mode,
            )
        )

    with multiprocessing.Pool(args.nproc) as pool:
        pool.starmap(worker, works)


if __name__ == "__main__":
    args = parse_args()
    setup_logger(args.log)
    main(args)