"""This file contains code to build OLN-RPN.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import RegionAssigner
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms, DeformConv2d
from mmcv.runner import force_fp32, BaseModule, ModuleList

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.core.bbox import bbox_overlaps
from ..builder import HEADS, build_loss, build_head
from .base_dense_head import BaseDenseHead
from .rpn_head import RPNHead


class AdaptiveConv(BaseModule):
    """AdaptiveConv used to adapt the sampling location with the anchors.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the conv kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements.
            Default: 3
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If set True, adds a learnable bias to the
            output. Default: False.
        type (str, optional): Type of adaptive conv, can be either 'offset'
            (arbitrary anchors) or 'dilation' (uniform anchor).
            Default: 'dilation'.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 bias=False,
                 type='dilation',
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv'))):
        super(AdaptiveConv, self).__init__(init_cfg)
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1 and groups == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def forward(self, x, offset):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x


@HEADS.register_module()
class OlnCascadeRPNHead(RPNHead):
    """OLN-RPN head.
    
    Learning localization instead of classification at the proposal stage is
    crucial as it avoids overfitting to the foreground by classification. For
    training the localization quality estimation branch, we randomly sample
    `num` anchors having an IoU larger than `neg_iou_thr` with the matched
    ground-truth boxes. It is recommended to use 'centerness' in this stage. For
    box regression, we replace the standard box-delta targets (xyhw) with
    distances from the location to four sides of the ground-truth box (lrtb). We
    choose to use one anchor per feature location as opposed to 3 in the standad
    RPN, because we observe its better generalization as each anchor can ingest
    more data.
    """

    def __init__(self, 
                 in_channels,
                 loss_objectness=dict(type='L1Loss', loss_weight=1.0),
                 objectness_type='Centerness',
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8],
                     ratios=[1.0],
                     strides=[4, 8, 16, 32, 64]),
                 adapt_cfg=dict(type='dilation', dilation=3),
                 bridged_feature=False,
                 sampling=True,
                 with_cls=True,
                 init_cfg=None,
                 **kwargs):
        
        self.with_cls = with_cls
        self.bridged_feature = bridged_feature
        self.anchor_strides = anchor_generator['strides']
        self.anchor_scales = anchor_generator['scales']
        self.adapt_cfg = adapt_cfg
                
        super(OlnCascadeRPNHead, self).__init__(
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        # Objectness loss
        self.loss_objectness = build_loss(loss_objectness)
        self.objectness_type = objectness_type
        self.with_class_score = self.loss_cls.loss_weight > 0.0
        self.with_objectness_score = self.loss_objectness.loss_weight > 0.0

        # override sampling and sampler
        self.sampling = sampling
        # Define objectness assigner and sampler
        if self.train_cfg:
            self.objectness_assigner = build_assigner(
                self.train_cfg.objectness_assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'objectness_sampler'):
                objectness_sampler_cfg = self.train_cfg.objectness_sampler
            else:
                objectness_sampler_cfg = dict(type='PseudoSampler')
            self.objectness_sampler = build_sampler(
                objectness_sampler_cfg, context=self)
        
    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = AdaptiveConv(self.in_channels, self.feat_channels,
                                     **self.adapt_cfg)
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

        assert self.num_anchors == 1 and self.cls_out_channels == 1, (
            'objectness_rpn -- num_anchors and cls_out_channels must be 1.')
        self.rpn_obj = nn.Conv2d(self.feat_channels, self.num_anchors, 1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_conv, std=0.01)
        if self.with_cls:
            normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_obj, std=0.01)

    def forward_single(self, x, offset):
        """Forward feature map of a single scale level."""
        bridged_x = x
        x = self.relu(self.rpn_conv(x, offset))
        if self.bridged_feature:
            bridged_x = x  # update feature
            
        # We add L2 normalization for training stability
        x = F.normalize(x, p=2, dim=1) 
        
        rpn_cls_score = self.rpn_cls(x) if self.with_cls else None
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_objectness_pred = self.rpn_obj(x)
        return bridged_x, rpn_cls_score, rpn_bbox_pred, rpn_objectness_pred
    
    def forward(self, feats, offset=None):
        if offset is None:
            offset = [None for _ in range(len(feats))]
        return multi_apply(self.forward_single, feats, offset)
    

    def _region_targets_single(self,
                               anchors,
                               valid_flags,
                               gt_bboxes,
                               gt_bboxes_ignore,
                               gt_labels,
                               img_meta,
                               featmap_sizes,
                               label_channels=1):
        """Get anchor targets based on region for single level."""
        assign_result = self.assigner.assign(
            anchors,
            valid_flags,
            gt_bboxes,
            img_meta,
            featmap_sizes,
            self.anchor_scales[0],
            self.anchor_strides,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_labels=None,
            allowed_border=self.train_cfg.allowed_border)
        flat_anchors = torch.cat(anchors)
        sampling_result = self.sampler.sample(assign_result, flat_anchors,
                                              gt_bboxes)

        num_anchors = flat_anchors.shape[0]
        bbox_targets = torch.zeros_like(flat_anchors)
        bbox_weights = torch.zeros_like(flat_anchors)
        labels = flat_anchors.new_zeros(num_anchors, dtype=torch.long)
        label_weights = flat_anchors.new_zeros(num_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def region_targets(self,
                       anchor_list,
                       valid_flag_list,
                       gt_bboxes_list,
                       img_metas,
                       featmap_sizes,
                       gt_bboxes_ignore_list=None,
                       gt_labels_list=None,
                       label_channels=1,
                       unmap_outputs=True):
        """See :func:`StageCascadeRPNHead.get_targets`."""
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._region_targets_single,
             anchor_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             featmap_sizes=featmap_sizes,
             label_channels=label_channels)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)
        
        
    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes,
                    img_metas,
                    featmap_sizes,
                    gt_bboxes_ignore=None,
                    gt_labels=None,
                    label_channels=1):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            featmap_sizes (list[Tensor]): Feature mapsize each level
            gt_bboxes_ignore (list[Tensor]): Ignore bboxes of each images
            label_channels (int): Channel of label.

        Returns:
            cls_reg_targets (tuple)
        """
        if isinstance(self.assigner, RegionAssigner):
            cls_reg_targets = self.region_targets(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                featmap_sizes,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                label_channels=label_channels)
        else:
            cls_reg_targets = self.get_targets_oln(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                label_channels=label_channels)
        return cls_reg_targets
    
    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        def _shape_offset(anchors, stride, ks=3, dilation=1):
            # currently support kernel_size=3 and dilation=1
            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list


    def loss_single(self, cls_score, bbox_pred, objectness_score, anchors,
                    labels, label_weights, bbox_targets, bbox_weights, 
                    objectness_targets, objectness_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            objectness_score (Tensor): Box objectness scorees for each anchor
                point has shape (N, num_anchors, H, W) 
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            objectness_targets (Tensor): Center regresion targets of each anchor
                with shape (N, num_total_anchors)
            objectness_weights (Tensor): Objectness weights of each anchro with 
                shape (N, num_total_anchors)
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        if self.with_cls:
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                -1, self.cls_out_channels).contiguous()
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        else:
            loss_cls = None

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        # objectness loss
        objectness_targets = objectness_targets.reshape(-1)
        objectness_weights = objectness_weights.reshape(-1)
        assert self.cls_out_channels == 1, (
            'cls_out_channels must be 1 for objectness learning.')
        objectness_score = objectness_score.permute(0, 2, 3, 1).reshape(-1)

        loss_objectness = self.loss_objectness(
            objectness_score.sigmoid(), 
            objectness_targets, 
            objectness_weights, 
            avg_factor=num_total_samples)
        
        if self.with_cls:
            return loss_cls, loss_bbox, loss_objectness
        return None, loss_bbox, loss_objectness

    
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectness_scores'))
    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             objectness_scores,
             gt_bboxes,
             # gt_labels,  # gt_labels is not used since we sample the GTs.
             img_metas,
             gt_bboxes_ignore=None,
             gt_labels=None):  # gt_labels is not used since we sample the GTs.
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            objectness_scores (list[Tensor]): Box objectness scorees for each
                anchor point with shape (N, num_anchors, H, W) 
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

                
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = bbox_preds[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_objectness_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            featmap_sizes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_labels=gt_labels,
            label_channels=label_channels)
        if cls_reg_objectness_targets is None:
            return None
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        num_total_pos, num_total_neg, objectness_targets_list,
        objectness_weights_list) = cls_reg_objectness_targets

        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            # 200 is hard-coded average factor,
            # which follows guided anchoring.
            num_total_samples = sum([label.numel()
                                     for label in labels_list]) / 200.0
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
            
        losses_cls, losses_bbox, losses_objectness = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                objectness_scores,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                objectness_targets_list,
                objectness_weights_list,
                num_total_samples=num_total_samples)
        

        if self.with_cls:
            return dict(
                loss_rpn_cls=losses_cls, 
                loss_rpn_bbox=losses_bbox,
                loss_rpn_obj=losses_objectness,)
        return dict(
            loss_rpn_bbox=losses_bbox,
            loss_rpn_obj=losses_objectness,)

    def _get_targets_single_oln(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        # Assign objectness gt and sample anchors
        objectness_assign_result = self.objectness_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore, None)
        objectness_sampling_result = self.objectness_sampler.sample(
            objectness_assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            # Sanlity check: left, right, top, bottom distances must be greater
            # than 0.
            valid_targets = torch.min(pos_bbox_targets,-1)[0] > 0
            bbox_targets[pos_inds[valid_targets], :] = (
                pos_bbox_targets[valid_targets])
            bbox_weights[pos_inds[valid_targets], :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        objectness_targets = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_weights = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)
        objectness_pos_inds = objectness_sampling_result.pos_inds
        objectness_neg_inds = objectness_sampling_result.neg_inds
        objectness_pos_neg_inds = torch.cat(
            [objectness_pos_inds, objectness_neg_inds])

        if len(objectness_pos_inds) > 0:
            # Centerness as tartet -- Default
            if self.objectness_type == 'Centerness':
                pos_objectness_bbox_targets = self.bbox_coder.encode(
                    objectness_sampling_result.pos_bboxes, 
                    objectness_sampling_result.pos_gt_bboxes)
                valid_targets = torch.min(pos_objectness_bbox_targets,-1)[0] > 0
                pos_objectness_bbox_targets[valid_targets==False,:] = 0
                top_bottom = pos_objectness_bbox_targets[:,0:2]
                left_right = pos_objectness_bbox_targets[:,2:4]
                pos_objectness_targets = torch.sqrt(
                    (torch.min(top_bottom, -1)[0] / 
                        (torch.max(top_bottom, -1)[0] + 1e-12)) *
                    (torch.min(left_right, -1)[0] / 
                        (torch.max(left_right, -1)[0] + 1e-12)))
            elif self.objectness_type == 'BoxIoU':
                pos_objectness_targets = bbox_overlaps(
                    objectness_sampling_result.pos_bboxes,
                    objectness_sampling_result.pos_gt_bboxes,
                    is_aligned=True)
            else:
                raise ValueError(
                    'objectness_type must be either "Centerness" (Default) or '
                    '"BoxIoU".')

            objectness_targets[objectness_pos_inds] = pos_objectness_targets
            objectness_weights[objectness_pos_inds] = 1.0   

        if len(objectness_neg_inds) > 0: 
            objectness_targets[objectness_neg_inds] = 0.0
            objectness_weights[objectness_neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

            # objectness targets
            objectness_targets = unmap(
                objectness_targets, num_total_anchors, inside_flags)
            objectness_weights = unmap(
                objectness_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result,
                objectness_targets, objectness_weights, 
                objectness_pos_inds, objectness_neg_inds, objectness_pos_neg_inds,
                objectness_sampling_result)

    def get_targets_oln(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single_oln,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list,
         all_objectness_targets, all_objectness_weights,
         objectness_pos_inds_list, objectness_neg_inds_list,
         objectness_pos_neg_inds_list, objectness_sampling_results_list
         ) = results[:13]

        rest_results = list(results[13:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        objectness_targets_list = images_to_levels(all_objectness_targets,
                                               num_level_anchors)
        objectness_weights_list = images_to_levels(all_objectness_weights,
                                               num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg,
               objectness_targets_list, objectness_weights_list,)

        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectness_scores'))
    def get_bboxes(self,
                   anchor_list,
                   cls_scores,
                   bbox_preds,
                   objectness_scores,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            objectness_scores (list[Tensor]): Box objectness scorees for each anchor
                point with shape (N, num_anchors, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        """

        assert len(cls_scores) == len(bbox_preds) and (
            len(cls_scores) == len(objectness_scores))
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            objectness_score_list = [
                objectness_scores[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    objectness_score_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    objectness_score_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           objectness_scores,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            objectness_score_list (list[Tensor]): Box objectness scorees for
                each anchor point with shape (N, num_anchors, H, W)
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            # <
            rpn_objectness_score = objectness_scores[idx]
            # >

            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            
            assert self.use_sigmoid_cls, 'use_sigmoid_cls must be True.'
            rpn_cls_score = rpn_cls_score.reshape(-1)
            rpn_cls_scores = rpn_cls_score.sigmoid()

            rpn_objectness_score = rpn_objectness_score.permute(
                1, 2, 0).reshape(-1)
            rpn_objectness_scores = rpn_objectness_score.sigmoid()
            
            # We use the predicted objectness score (i.e., localization quality)
            # as the final RPN score output.
            scores = rpn_objectness_scores

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)

        # No NMS:
        dets = torch.cat([proposals, scores.unsqueeze(1)], 1)
        
        return dets
    
    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):
        """Refine bboxes through stages."""
        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list

@HEADS.register_module()
class OlnRPN(BaseDenseHead):
    """The CascadeRPNHead will predict more accurate region proposals, which is
    required for two-stage detectors (such as Fast/Faster R-CNN). CascadeRPN
    consists of a sequence of RPNStage to progressively improve the accuracy of
    the detected proposals.

    More details can be found in ``https://arxiv.org/abs/1909.06720``.

    Args:
        num_stages (int): number of CascadeRPN stages.
        stages (list[dict]): list of configs to build the stages.
        train_cfg (list[dict]): list of configs at training time each stage.
        test_cfg (dict): config at testing time.
    """

    def __init__(self, num_stages, stages, train_cfg, test_cfg, init_cfg=None):
        super(OlnRPN, self).__init__(init_cfg)
        assert num_stages == len(stages)
        self.num_stages = num_stages
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.stages = ModuleList()
        for i in range(len(stages)):
            train_cfg_i = train_cfg[i] if train_cfg is not None else None
            stages[i].update(train_cfg=train_cfg_i)
            stages[i].update(test_cfg=test_cfg)
            self.stages.append(build_head(stages[i]))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self):
        """loss() is implemented in StageCascadeRPNHead."""
        pass

    def get_bboxes(self):
        """get_bboxes() is implemented in StageCascadeRPNHead."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None):
        """Forward train function."""
        assert gt_labels is None, 'RPN does not require gt_labels'


        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)

        losses = dict()

        for i in range(self.num_stages):
            stage = self.stages[i]

            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            
            x, cls_score, bbox_pred, objectness_score = stage(x, offset_list)

            rpn_loss_inputs = (anchor_list, valid_flag_list, cls_score,
                               bbox_pred, objectness_score, gt_bboxes, img_metas)
  
            stage_loss = stage.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses['s{}.{}'.format(i, name)] = value

            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                       bbox_pred, objectness_score, img_metas,
                                                       self.test_cfg)
            return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)

        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
            else:
                offset_list = None
            x, cls_score, bbox_pred, objectness_score = stage(x, offset_list)
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)

        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                   bbox_pred, objectness_score, img_metas,
                                                   self.test_cfg)
        return proposal_list

    def aug_test_rpn(self, x, img_metas):
        """Augmented forward test function."""
        raise NotImplementedError(
            'CascadeRPNHead does not support test-time augmentation')

        