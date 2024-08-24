model = dict(
    type='ImGeoNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=128,
        num_outs=4),
    neck_3d=dict(
        type='FastIndoorImVoxelNeck',
        in_channels=128,
        out_channels=128,
        n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ScanNetImVoxelHeadV2',
        loss_bbox=dict(type='AxisAlignedIoULoss', loss_weight=1.0),
        n_classes=189,
        n_channels=128,
        n_reg_outs=6,
        n_scales=3,
        limit=27,
        centerness_topk=18),
    voxel_size=(.08, .08, .08),
    n_voxels=(80, 80, 32))
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    iou_thr=.25,
    score_thr=.01)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'ScanNetMultiViewDataset'
data_root = '../data/scannet/'
class_names = [
            'wall', 'chair', 'floor', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk',
            'office chair', 'bed', 'pillow', 'sink', 'picture', 'window', 'toilet', 'bookshelf',
            'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box', 'refrigerator', 'lamp',
            'kitchen cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool',
            'cushion', 'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard', 'bag',
            'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard', 'blanket', 'shower curtain',
            'trash can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer tower', 'bottle',
            'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier', 'basket',
            'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
            'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano',
            'suitcase', 'rail', 'radiator', 'recycling bin', 'container', 'wardrobe', 'soap dispenser',
            'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer',
            'guitar', 'toilet paper holder', 'seat', 'speaker', 'column', 'ladder', 'bathroom stall',
            'shower wall', 'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher',
            'paper towel roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board',
            'ironing board', 'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
            'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball', 'hat',
            'shower curtain rod', 'water cooler', 'paper cutter', 'tray', 'shower door', 'pillar',
            'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'furniture', 'cart',
            'scale', 'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign',
            'projector', 'closet door', 'vacuum cleaner', 'plunger', 'stuffed animal', 'headphones',
            'dish rack', 'broom', 'range hood', 'dustpan', 'hair dryer', 'water bottle', 'handicap bar',
            'vent', 'shower floor', 'water pitcher', 'mailbox', 'bowl', 'paper bag', 'projector screen',
            'divider', 'laundry detergent', 'bathroom counter', 'object', 'bathroom vanity',
            'closet wall', 'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin',
            'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'closet rod', 'coffee kettle',
            'shower head', 'keyboard piano', 'case of water bottles', 'coat rack', 'folded chair',
            'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', 'mattress']

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=20,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(480, 640))
        ]),
    dict(type='RandomShiftOrigin', std=(.7, .7, .0)),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        n_images=50,
        sample_method="linear",
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(480, 640))
        ]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet200_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet200_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet200_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
total_epochs = 30
lr_config = dict(policy='step', step=[8, 29])

checkpoint_config = dict(interval=1, max_keep_ckpts=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
