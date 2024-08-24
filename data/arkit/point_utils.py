import argparse
import pathlib

import open3d as o3d
import numpy as np

import imageio.v2 as imageio


def gen_points_single(image, depth_map, pose, intrinsic, stride=1):
    height, width, _ = image.shape

    u, v = np.meshgrid(range(0, width, stride), range(0, height, stride), indexing="ij")

    rgb = image[v, u]
    depth = depth_map[v, u]
    mask = depth != 0

    u = u[mask]
    v = v[mask]
    depth = depth[mask]
    rgb = rgb[mask]

    # Pixel coord.
    pts = np.stack([u * depth, v * depth, depth, np.ones_like(u)])

    K = np.eye(4)
    K[:3, :3] = intrinsic

    # World coord.
    pts = pose @ np.linalg.inv(K) @ pts
    pts[:3] /= pts[3]
    pts = pts[:3].T
    return pts, rgb


def gen_points(images, depth_maps, poses, intrinsic, stride=1):
    pts_all = []
    rgb_all = []
    for image, depth_map, pose in zip(images, depth_maps, poses):
        pts, rgb = gen_points_single(image, depth_map, pose, intrinsic, stride)
        pts_all.append(pts)
        rgb_all.append(rgb)
    pts_all = np.concatenate(pts_all, axis=0)
    rgb_all = np.concatenate(rgb_all, axis=0)
    return pts_all, rgb_all


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-dir",
        type=pathlib.Path,
        help="Path to scene directory.",
    )
    parser.add_argument("--output", type=pathlib.Path, help="Output Path.")
    parser.add_argument(
        "--log", type=pathlib.Path, default="compute_trans.log", help="Log path."
    )
    return parser.parse_args()


def load_file(path):
    ext = path.suffix
    if ext == ".png" or ext == ".jpg":
        ret = imageio.imread(path)
    elif ext == ".npy":
        ret = np.load(path)
    else:
        raise NotImplementedError(f"Not support: {ext}")
    return ret


def load_files(directory, pattern, return_array=False):
    objs = [load_file(path) for path in sorted(list(directory.glob(pattern)))]
    if return_array:
        if len(objs) > 0 and isinstance(objs[0], np.ndarray):
            objs = np.stack(objs)
        else:
            objs = np.array(objs)
    return objs


def save_pcd(fpath, pts, rgb):
    rgb = rgb.astype(np.float) / 255.0  # Scale to [0, 1]
    ext = fpath.suffix
    if ext == ".ply":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        o3d.io.write_point_cloud(str(fpath), pcd)
    elif ext == ".npy":
        np.save(fpath, np.concatenate([pts, rgb], axis=-1))
    else:
        raise NotImplementedError(f"Not support {ext} format.")


def main(args):
    scene_dir = args.scene_dir
    images = load_files(scene_dir, "*_color.png", return_array=True)
    depth_maps = load_files(scene_dir, "*_depth.png", return_array=True)
    poses = load_files(scene_dir, "*_pose.npy", return_array=True)
    intrinsic = load_file(scene_dir / "intrinsic.npy")

    depth_shift = 1000.  # mm to meter
    pts, rgb = gen_points(images, depth_maps / depth_shift, poses, intrinsic)
    save_pcd(args.output, pts, rgb)


if __name__ == "__main__":
    main(parse_args())