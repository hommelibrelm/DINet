_base_ = [
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
pretrained = 'trained_weights/mae_pretrain_vit_base_full.pth'
norm_cfg = dict(type='LN', requires_grad=True)
load_from = 'path_of_base_training.pth'

model = dict(
    type='DINet',
    pretrained=pretrained,
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path_rate=0.2,
        learnable_pos_embed=True,
        use_checkpoint=True,
        with_simple_fpn=True,
        last_feat=True),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='DynamicDecoupledHead',
        num_stages=2,
        stages=[
            dict(
                type='StageDynamicDecoupledHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                dcn_on_last_conv=True,
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='TBLRBBoxCoder', 
                    normalizer=1.0,),
                use_tower_convs=True,
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=1.0)),
            dict(
                type='StageDynamicDecoupledHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                dcn_on_last_conv=True,
                bridged_feature=False,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='TBLRBBoxCoder',
                    normalizer=1.0,),
                use_tower_convs=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=1.0))
        ]),
    roi_head=dict(
        type='DINetRoIHead',
        bbox_roi_extractor=[dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=768,
            featmap_strides=[4, 8, 16, 32]),
                            dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=768,
            featmap_strides=[16])],
        bbox_head=dict(
            type='InterBBoxHead',
            use_checkpoint=False,
            in_channels=768,
            img_size=224,
            patch_size=16, 
            embed_dim=512, 
            depth=8,
            num_heads=16, 
            mlp_ratio=4., 
            num_classes=20,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
        ),
    ),
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        ],
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
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
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

evaluation = dict(interval=12, metric='bbox')
checkpoint_config = dict(interval=36)
classes =  ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'chair', 'diningtable', 'dog', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor',
            'aeroplane', 'bottle', 'cow', 'horse', 'sofa')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
        train=dict(
        pipeline=train_pipeline,
        classes = classes, 
        ann_file='data/voc/voc_fewshot_split2/seed1/5shot_fewshot.json',
        img_prefix='data/voc/PascalVoc_CocoStyle/images'),
    val=dict(
        classes = classes, 
        ann_file='data/voc/PascalVoc_CocoStyle/annotations/pascal_test2007_split2_new.json',
        img_prefix='data/voc/PascalVoc_CocoStyle/images'),
    test=dict(
        classes = classes, 
        ann_file='data/voc/PascalVoc_CocoStyle/annotations/pascal_test2007_split2_new.json',
        img_prefix='data/voc/PascalVoc_CocoStyle/images')
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructorBackboneFronzen', 
    paramwise_cfg=dict(
            num_encoder_layers=12, 
            num_decoder_layers=8, 
            layer_decay_rate=0.8,
    )
)


# learning policy
lr_config = dict(policy='step', step=[81, 99])
runner = dict(type='EpochBasedRunner', max_epochs=108)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)
# find_unused_parameters=True