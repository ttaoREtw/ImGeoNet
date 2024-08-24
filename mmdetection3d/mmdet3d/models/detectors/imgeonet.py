import numpy as np
import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result


@DETECTORS.register_module()
class ImGeoNet(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 voxel_size,
                 occ_head=None,
                 use_gt_occ=False,
                 head_2d=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 depth_cast_margin=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.voxel_size = voxel_size
        self.head_2d = build_head(head_2d) if head_2d is not None else None
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.depth_cast_margin = depth_cast_margin
        self.occ_head = build_head(occ_head) if occ_head is not None else None
        self.use_gt_occ = use_gt_occ
        self.init_weights(pretrained=pretrained)
        print(f"model size: {self.compute_model_size():.3f}MB")

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.neck_3d.init_weights()
        self.bbox_head.init_weights()
        if self.head_2d is not None:
            self.head_2d.init_weights()
        if self.occ_head is not None:
            self.occ_head.init_weights()

    def compute_model_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f"param_size: {param_size/1024**2}, buffer_size: {buffer_size/1024**2}")
        return size_all_mb

    def build_volume(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)
        features_2d = self.head_2d.forward(x[-1], img_metas) if self.head_2d is not None else None
        x = self.neck(x)[0]
        x = x.reshape([batch_size, -1] + list(x.shape[1:]))

        stride = img.shape[-1] / x.shape[-1]
        assert stride == 4  # may be removed in the future
        stride = int(stride)

        avg_vols, var_vols, valids = [], [], []
        for feature, img_meta in zip(x, img_metas):
            # use predicted pitch and roll for SUNRGBDTotal test
            angles = features_2d[0] if features_2d is not None and mode == 'test' else None
            # projection = self._compute_projection(img_meta, stride, angles).to(x.device)
            points = get_points(
                n_voxels=torch.tensor(self.n_voxels),
                voxel_size=torch.tensor(self.voxel_size),
                origin=torch.tensor(img_meta['lidar2img']['origin'])
            ).to(x.device)
            height = img_meta['img_shape'][0] // stride
            width = img_meta['img_shape'][1] // stride
            volume, valid = backproject(
                feature[:, :, :height, :width],
                points,
                img_meta,
                stride,
                )

            # Cost volume
            valid_count = valid.sum(dim=0)
            valid_mask = valid_count > 0
            invalid_voxel_mask = ~valid_mask[0]

            vol = volume.sum(dim=0)
            vol2 = volume.pow(2).sum(dim=0)
            avg_vol = vol / valid_count
            avg_vol2 = vol2 / valid_count
            avg_vol[:, invalid_voxel_mask] = .0
            avg_vol2[:, invalid_voxel_mask] = .0
            # Var = E[X^2] - E[X]^2
            var_vol = avg_vol2 - avg_vol.pow(2)

            avg_vols.append(avg_vol)
            var_vols.append(var_vol)
            valids.append(valid_mask)

        avg_vols = torch.stack(avg_vols)
        var_vols = torch.stack(var_vols)
        valids = torch.stack(valids)
        return avg_vols, var_vols, valids, features_2d

    def extract_feat(self, volumes):
        return self.neck_3d(volumes)

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        avg_vols, var_vols, valids, features_2d = self.build_volume(img, img_metas, 'train')

        losses = {}

        occ = 1
        if self.occ_head is not None:
            depth_maps = kwargs['depth_maps']
            depth_masks = kwargs['depth_masks']
            target_occ = self.compute_target_occ(img_metas, depth_maps, depth_masks)
            occ, occ_loss = self.occ_head.forward_train(
                torch.cat([avg_vols, var_vols], dim=1), valids, target_occ)
            losses.update(occ_loss)
        elif self.use_gt_occ:
            depth_maps = kwargs['depth_maps']
            depth_masks = kwargs['depth_masks']
            target_occ = self.compute_target_occ(img_metas, depth_maps, depth_masks)
            occ = target_occ.float()
        x = self.extract_feat(avg_vols * occ)

        losses.update(
            self.bbox_head.forward_train(x, valids.float(), img_metas, gt_bboxes_3d, gt_labels_3d))
        if self.head_2d is not None:
            losses.update(
                self.head_2d.loss(*features_2d, img_metas))
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        avg_vols, var_vols, valids, features_2d = self.build_volume(img, img_metas, 'test')

        occ = 1
        if self.occ_head is not None:
            occ = self.occ_head.forward_test(
                torch.cat([avg_vols, var_vols], dim=1), valids)
        elif self.use_gt_occ:
            depth_maps = kwargs['depth_maps']
            depth_masks = kwargs['depth_masks']
            target_occ = self.compute_target_occ(img_metas, depth_maps, depth_masks)
            occ = target_occ.float()
        x = self.extract_feat(avg_vols * occ)
        x = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*x, valids.float(), img_metas)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        if self.head_2d is not None:
            angles, layouts = self.head_2d.get_bboxes(*features_2d, img_metas)
            for i in range(len(img)):
                bbox_results[i]['angles'] = angles[i]
                bbox_results[i]['layout'] = layouts[i]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass

    def compute_target_occ(self, img_metas, depth_maps, depth_masks):
        with torch.no_grad():
            device = depth_maps.device
            target_occ = []
            for img_meta, dep_maps, dep_masks in zip(img_metas, depth_maps, depth_masks):
                points = get_points(
                    n_voxels=torch.tensor(self.n_voxels),
                    voxel_size=torch.tensor(self.voxel_size),
                    origin=torch.tensor(img_meta['lidar2img']['origin'])
                ).to(device)
                tgt_occ = compute_target_occ_single(
                    points, img_meta, dep_maps, dep_masks,
                    self.voxel_size, self.depth_cast_margin)
                target_occ.append(tgt_occ)
        return torch.stack(target_occ)
            

    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predicted pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]),
        torch.arange(n_voxels[1]),
        torch.arange(n_voxels[2])
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def compute_projection(img_meta, stride=1):
    projection = []
    intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
    ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
    intrinsic[:2] /= ratio
    extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
    for extrinsic in extrinsics:
        projection.append(intrinsic @ extrinsic[:3])
    return torch.stack(projection)


def compute_target_occ_single(points, img_meta, depth_maps, depth_masks, voxel_size, depth_cast_margin):
    device = points.device
    n_images = len(img_meta['lidar2img']['extrinsic'])
    H, W = depth_maps.shape[1], depth_maps.shape[2]
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # (num_images, 3+1, num_voxels)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # (num_images, 3, num_voxels)
    points_2d = torch.bmm(compute_projection(img_meta).to(device), points)
    # (num_images, num_voxels)
    z = points_2d[:, 2]
    x = (points_2d[:, 0] / z).round().long()
    y = (points_2d[:, 1] / z).round().long()
    
    valid = (x >= 0) & (y >= 0) & (x < W) & (y < H) & (z > 0)

    n_voxels = points.shape[-1]
    gt_depth = torch.zeros((n_images, n_voxels), device=device)
    for i in range(n_images):
        valid[i, valid[i]] = valid[i, valid[i]] & depth_masks[i, y[i, valid[i]], x[i, valid[i]]]
        gt_depth[i, valid[i]] = depth_maps[i, y[i, valid[i]], x[i, valid[i]]]

    extrinsic = torch.tensor(np.stack(img_meta['lidar2img']['extrinsic'])).to(device)
    # Shape: (num_images, 3+1, num_voxels)
    points_cam = torch.bmm(extrinsic, points)
    # Shape: (num_images, num_voxels)
    vx_depth = points_cam[:, 2] / points_cam[:, 3]
    margin = voxel_size[2] * (depth_cast_margin * 0.5)
    for i in range(n_images):
        gt_dep = gt_depth[i, valid[i]]
        vx_dep = vx_depth[i, valid[i]]
        valid[i, valid[i]] = valid[i, valid[i]] & ((gt_dep <= vx_dep + margin) & \
                                                   (vx_dep - margin <= gt_dep))
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    target_occ = (valid.sum(dim=0) > 0)
    return target_occ

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, img_meta, feat_stride):
    device = features.device
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # Shape: (num_images, 3+1, num_voxels)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # Shape: (num_images, 3, num_voxels)
    points_2d_feat = torch.bmm(compute_projection(img_meta, feat_stride).to(device), points)
    x_feat = (points_2d_feat[:, 0] / points_2d_feat[:, 2]).round().long()
    y_feat = (points_2d_feat[:, 1] / points_2d_feat[:, 2]).round().long()
    z_feat = points_2d_feat[:, 2]
    valid = (x_feat >= 0) & (y_feat >= 0) & (x_feat < width) & (y_feat < height) & (z_feat > 0)

    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y_feat[i, valid[i]], x_feat[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic
