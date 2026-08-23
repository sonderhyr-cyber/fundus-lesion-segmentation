_base_ = [
    '../_base_/models/hrdecoder_fcn_hr48.py',
    '../_base_/datasets/hr_ddr_2048.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10warm.py',
]

# 板块二 ②: baseline + SE copy-paste augmentation (custom extension,
# NOT part of the original paper method). Fixed 6:1:3 split, seed 42.

model = dict(
    type='HRDecoder',
    hr_settings=dict(
        hr_scale=(1024, 1024),
        scale_ratio=(0.75, 1.25),
        divisible=8,
        lr_loss_weight=0,
        hr_loss_weight=0.1,
        fuse_mode='simple',
        crop_num=4,
    ),
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=True)),
    decode_head=dict(norm_cfg=dict(type='BN', requires_grad=True)),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='SECopyPaste',
         library_path='/root/HRDecoder/data/DDR_split_613/se_lesion_library.pkl',
         prob=0.5,
         max_patches=2),
    dict(type='Resize', img_scale=(2048, 2048), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(2048, 2048), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', prob=1.0, pad_val=0, seg_pad_val=0,
         degree=(-45, 45), auto_bound=False),
    dict(type='Normalize',
         mean=[81.205, 50.636, 21.216],
         std=[76.252, 48.798, 21.625],
         to_rgb=True),
    dict(type='Pad', size=(2048, 2048), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline),
    val=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/val',
        ann_dir='labels/val'),
    test=dict(
        data_root='./data/DDR_split_613',
        img_dir='images/test',
        ann_dir='labels/test'),
)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)

seed = 42
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU', priority='LOW', save_best='mIoU')
