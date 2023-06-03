_base_ = [
    '../_base_/models/faster_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.core.optimizers.adan_t'],
    allow_failed_imports=False)
rpn_weight = 0.7
pretrained = 'trained_weights/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='FasterRCNN',
    pretrained=pretrained,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    # rpn_head=dict(
    #     _delete_=True,
    #     type='CascadeRPNHead',
    #     num_stages=2,
    #     stages=[
    #         dict(
    #             type='StageCascadeRPNHead',
    #             in_channels=256,
    #             feat_channels=256,
    #             anchor_generator=dict(
    #                 type='AnchorGenerator',
    #                 scales=[8],
    #                 ratios=[1.0],
    #                 strides=[4, 8, 16, 32, 64]),
    #             adapt_cfg=dict(type='dilation', dilation=3),
    #             bridged_feature=True,
    #             sampling=False,
    #             with_cls=False,
    #             reg_decoded_bbox=True,
    #             bbox_coder=dict(
    #                 type='DeltaXYWHBBoxCoder',
    #                 target_means=(.0, .0, .0, .0),
    #                 target_stds=(0.1, 0.1, 0.5, 0.5)),
    #             loss_bbox=dict(
    #                 type='IoULoss', linear=True,
    #                 loss_weight=10.0 * rpn_weight)),
    #         dict(
    #             type='StageCascadeRPNHead',
    #             in_channels=256,
    #             feat_channels=256,
    #             adapt_cfg=dict(type='offset'),
    #             bridged_feature=False,
    #             sampling=True,
    #             with_cls=True,
    #             reg_decoded_bbox=True,
    #             bbox_coder=dict(
    #                 type='DeltaXYWHBBoxCoder',
    #                 target_means=(.0, .0, .0, .0),
    #                 target_stds=(0.05, 0.05, 0.1, 0.1)),
    #             loss_cls=dict(
    #                 type='CrossEntropyLoss',
    #                 use_sigmoid=True,
    #                 loss_weight=1.0 * rpn_weight),
    #             loss_bbox=dict(
    #                 type='IoULoss', linear=True,
    #                 loss_weight=10.0 * rpn_weight))
    #     ]),
    rpn_head=dict(
        _delete_=True,
        type='UVORPNHead',
        num_stages=2,
        stages=[
            dict(
                type='UVOStageCascadeRPNHead',
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
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2)),
                use_tower_convs=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                ),
            dict(
                type='UVOStageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                dcn_on_last_conv=True,
                bridged_feature=False,
                sampling=False,
                with_cls=True,
                cls_head='cls_head',
                reg_decoded_bbox=False,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                use_tower_convs=True,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                )
        ]),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
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
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            ),
    train_cfg=dict(rpn=[
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25),
            aux_assigner=dict(type='RPN_SimOTAAssigner', center_radius=.25, candidate_topk=20),
            aux_sampler=None,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ]),
    # train_cfg=dict(rpn=[
    #         dict(
    #             assigner=dict(
    #                 type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
    #             aux_assigner=None,
    #             aux_sampler=None,
    #             allowed_border=-1,
    #             pos_weight=-1,
    #             debug=False),
    #         dict(
    #             assigner=dict(
    #                 type='MaxIoUAssigner',
    #                 pos_iou_thr=0.7,
    #                 neg_iou_thr=0.7,
    #                 min_pos_iou=0.3,
    #                 ignore_iof_thr=-1),
    #             aux_assigner=None,
    #             aux_sampler=None,
    #             sampler=dict(
    #                 type='RandomSampler',
    #                 num=256,
    #                 pos_fraction=0.5,
    #                 neg_pos_ub=-1,
    #                 add_gt_as_proposals=False),
    #             allowed_border=0,
    #             pos_weight=-1,
    #             debug=False)
    #     ],
    #     rpn_proposal=dict(
    #         nms_across_levels=False,
    #         nms_pre=2000,
    #         nms_post=2000,
    #         max_per_img=2000,
    #         nms_thr=0.7,
    #         min_bbox_size=0),
    #     rcnn=dict(
    #         assigner=dict(
    #             type='MaxIoUAssigner',
    #             pos_iou_thr=0.5,
    #             neg_iou_thr=0.5,
    #             min_pos_iou=0.5,
    #             match_low_quality=False,
    #             ignore_iof_thr=-1),
    #         sampler=dict(
    #             type='RandomSampler',
    #             num=512,
    #             pos_fraction=0.25,
    #             neg_pos_ub=-1,
    #             add_gt_as_proposals=True),
    #         pos_weight=-1,
    #         debug=False)),
    test_cfg=dict(
         rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0),
            min_bbox_size=0),
         rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100)
         )
    )

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


# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

optimizer = dict(
    _delete_=True,
    type='Adan',
    lr=0.002,
    betas=(0.98, 0.92, 0.99),
    weight_decay=0.02,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75,
            custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
)

lr_config = dict(warmup_iters=1000, step=[8, 11])
total_epochs = 12
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
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

evaluation = dict(interval=1, metric='proposal_fast')

work_dir='./work_dirs/swin_s/'
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
