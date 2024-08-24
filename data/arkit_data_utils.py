import mmcv
import numpy as np
from concurrent import futures as futures
import pathlib
import pickle


class ARKitData(object):
    """ARKitScenes data.
    """

    def __init__(self, root_path, split="train"):
        assert split in ["train", "val"]
        self.split = split
        self.root_dir = pathlib.Path(root_path)
        self.data_dir = self.root_dir / "processed" / "3dod" / ("training" if split == "train" else "validation")
        self.sample_id_list = sorted([scene_dir.name for scene_dir in self.data_dir.iterdir()])
        self.classes = [
            "cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", 
            "toilet", "bathtub", "oven", "dishwasher", "fireplace", "stool",
            "chair", "table", "tv_monitor", "sofa"
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}

    def __len__(self):
        return len(self.sample_id_list)

    def get_annotation(self, idx):
        path = self.data_dir / idx / "annotations.pkl"
        bboxs = None
        labels = None
        if path.exists():
            with open(path, "rb") as f:
                ann = pickle.load(f)
            if len(ann["bbox"]) != 0:
                bboxs = ann["bbox"]
                labels = ann["label"]
        return bboxs, labels

    def get_align_matrix(self, idx):
        # Not used anymore
        return np.eye(4)
    
    def get_intrinsic(self, idx):
        path = self.data_dir / idx / "intrinsic.npy"
        mat = None
        if path.exists():
            mat = np.load(path)
        return mat

    def get_frame_data(self, idx):
        paths = sorted([
            path.relative_to(self.root_dir)
            for path in (self.data_dir / idx).glob("*_color.png")
            ])
        rgb_paths = [str(path) for path in paths]
        dep_paths = [path.replace("color", "depth") for path in rgb_paths]
        pose_paths = [path.replace("color", "pose").replace(".png", ".npy") for path in rgb_paths]
        poses = [np.load(self.root_dir / path) for path in pose_paths]
        return (rgb_paths, dep_paths, poses)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.
        This method gets information from the raw data.
        Args:
            num_workers (int): Number of threads to be used. Default: 4.
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.
        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f"{self.split} sample_idx: {sample_idx}")

            rgb_paths, dep_paths, poses = self.get_frame_data(sample_idx)
            intrinsic = self.get_intrinsic(sample_idx)
            bboxs, labels = self.get_annotation(sample_idx)
            align_mat = self.get_align_matrix(sample_idx)
            if any(x is None for x in (rgb_paths, dep_paths, poses, intrinsic,
                                       bboxs, labels, align_mat)):
                return None

            finite_indices = [i for i, pose in enumerate(poses) if np.all(np.isfinite(pose))]

            if len(finite_indices) == 0:
                return None            

            rgb_paths = [rgb_paths[i] for i in finite_indices]
            dep_paths = [dep_paths[i] for i in finite_indices]
            poses = [poses[i] for i in finite_indices]

            info = dict()
            info["intrinsic"] = intrinsic
            info["img_paths"] = rgb_paths
            info["depth_paths"] = dep_paths
            info["poses"] = poses

            annotations = {}
            bboxs = np.stack(bboxs)
            annotations["gt_num"] = bboxs.shape[0]
            # annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)  # ?
            # annotations["location"] = bboxs[:, :3]  # ?
            # annotations["dimensions"] = bboxs[:, 3:6]  # ?
            # annotations["name"] = np.array(labels)  # ?
            annotations["gt_boxes_upright_depth"] = bboxs
            annotations["align_matrix"] = align_mat
            annotations["class"] = np.array([self.cat2label[cat] for cat in labels])
            info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return [info for info in infos if info is not None]
