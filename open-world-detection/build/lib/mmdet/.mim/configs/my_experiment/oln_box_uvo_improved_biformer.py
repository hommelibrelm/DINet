_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]
custom_imports = dict(
    imports=['mmdet.core.optimizers.adan_t'],
    allow_failed_imports=False)
# model settings
pretrained = 'trained_weights/biformer_small_best.pth'
model = dict(
    type='FasterRCNN',
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='BiFormer_mm',
        #--------------------------
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        mlp_ratios=[3, 3, 3, 3],
        n_win=8, # training resolution is 512
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        #--------------------------
        # it seems whole_eval takes raw-resolution input
        # use auto_pad to allow any-size input
        auto_pad=True,
        # use grad ckpt to save memory on old gpus
        use_checkpoint_stages=[],
        drop_path_rate=0.15),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OlnRPNDecoupledHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            # Use a single anchor per location.
            scales=[8],
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64]),
        adapt_cfg=dict(type='dilation', dilation=3),
        dcn_on_last_conv=True,
        sampling=True,
        with_cls=True,
        bbox_coder=dict(
            type='TBLRBBoxCoder',
            normalizer=1.0,),
        use_tower_convs=True,
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=0.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='IoULoss', linear=True, 
                       loss_weight=10.0),
        objectness_type='Centerness',
        loss_objectness=dict(type='SmoothL1Loss', loss_weight=1.0),
        ),
    roi_head=dict(
        _delete_=True,
        type='OlnRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxScoreHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=0.0,
                ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            bbox_score_type='BoxIoU',  # 'BoxIoU' or 'Centerness'
            loss_bbox_score=dict(type='L1Loss', loss_weight=1.0),
            )),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            # Objectness assigner and sampler 
            objectness_assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.1,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            objectness_sampler=dict(
                type='RandomSampler',
                num=256,
                # Ratio 0 for negative samples.
                pos_fraction=1.,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms_thr=0.0,  # No nms
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.7),
            # max_per_img should be greater enough than k of AR@k evaluation
            # because the cross-dataset AR evaluation does not count those
            # proposals on the 'seen' classes into the budget (k), to avoid
            # evaluating recall on seen-class objects. It's recommended to use
            # max_per_img=1500 or 2000 when evaluating upto AR@1000.
            max_per_img=1500,
            )))




img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CocoSplitDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        type=dataset_type,
        pipeline=train_pipeline,
        ),
    val=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        type=dataset_type,
        pipeline=test_pipeline
        ),
    test=dict(
        is_class_agnostic=True,
        train_class='voc',
        eval_class='nonvoc',
        type=dataset_type,
        pipeline=test_pipeline
        ))

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8)

optimizer = dict(
    _delete_=True,
    type='Adan',
    lr=0.001,
    betas=(0.98, 0.92, 0.99),
    weight_decay=0.02,
)

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[6, 7])
runner = dict(type='EpochBasedRunner', max_epochs=8)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
find_unused_parameters=False
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=1, metric='bbox')

work_dir='./work_dirs/oln_box_cascade_uvo_improved_biformer_decoupled_lr002/'

# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )
