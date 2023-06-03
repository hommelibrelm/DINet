"""This file contains code to build OLN-Box head.

Reference:
    "Learning Open-World Object Proposals without Learning to Classify",
        Aug 2021. https://arxiv.org/abs/2108.06753
        Dahun Kim, Tsung-Yi Lin, Anelia Angelova, In So Kweon and Weicheng Kuo
"""

import torch
import torch.nn as nn

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class MyRoIHead(StandardRoIHead):
    """OLN Box head.
    
    We take the top-scoring (e.g., well-centered) proposals from OLN-RPN and
    perform RoIAlign to extract the region features from each feature pyramid
    level. Then we linearize each region features and feed it through two fc
    layers, followed by two separate fc layers, one for bbox regression and the
    other for localization quality prediction. It is recommended to use IoU as
    the localization quality target in this stage. 
    """

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        if isinstance(bbox_roi_extractor, list):
            self.ms_bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor[0])
            self.ss_bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor[1])
            self.mfm_factor = nn.Parameter(torch.zeros(bbox_head['in_channels'], requires_grad=True), requires_grad=True)
            self.mfm_fc = nn.Conv2d(in_channels=256,out_channels=bbox_head['in_channels'],kernel_size=1)
            self.with_mfm = True
        else:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
            self.with_mfm = False
        self.bbox_head = build_head(bbox_head)
        
    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            if self.with_mfm:
                self.ms_bbox_roi_extractor.init_weights()
                self.ss_bbox_roi_extractor.init_weights()
            else:
                self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights(pretrained)
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

        
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        if self.with_mfm:
            ss_bbox_feats = self.ss_bbox_roi_extractor(
                [x[-1]], rois)
            x = [self.mfm_fc(x[i]) for i in range(self.ms_bbox_roi_extractor.num_inputs)]
            ms_bbox_feats = self.ms_bbox_roi_extractor(
                x[:self.ms_bbox_roi_extractor.num_inputs], rois) # multi scale

            factor = self.mfm_factor.reshape(1, -1, 1, 1).expand_as(ms_bbox_feats)
            bbox_feats = ss_bbox_feats + ms_bbox_feats * factor
        else:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_score = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            bbox_score=bbox_score)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], 
                                        bbox_results['bbox_score'],
                                        rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        # RPN score
        rpn_score = torch.cat([p[:, -1:] for p in proposals], 0)
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        bbox_score = bbox_results['bbox_score']

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_score = bbox_score.split(num_proposals_per_img, 0)          
        rpn_score = rpn_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                bbox_score[i],
                rpn_score[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
