import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead



@HEADS.register_module()
class DINetRoIHead(StandardRoIHead):
    """RoIHead with multi-scale feature modulator on the input of bbox head."""


    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        
        self.bbox_roi_extractor_fpn = build_roi_extractor(bbox_roi_extractor[0])
        self.bbox_roi_extractor_encoder = build_roi_extractor(bbox_roi_extractor[1])
        self.bbox_factor = nn.Parameter(torch.zeros(bbox_head['in_channels'], requires_grad=True), requires_grad=True)
        self.fc = nn.Conv2d(in_channels=256,out_channels=bbox_head['in_channels'],kernel_size=1)

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
            self.bbox_roi_extractor_fpn.init_weights()
            self.bbox_roi_extractor_encoder.init_weights()
            self.bbox_head.init_weights(pretrained)
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()


    
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats_encoder = self.bbox_roi_extractor_encoder(
                [x[-1]], rois)
        x = [self.fc(x[i]) for i in range(self.bbox_roi_extractor_fpn.num_inputs)]
        bbox_feats_fpn = self.bbox_roi_extractor_fpn(
                x[:self.bbox_roi_extractor_fpn.num_inputs], rois) # multi scale

        factor = self.bbox_factor.reshape(1, -1, 1, 1).expand_as(bbox_feats_fpn)
        bbox_feats = bbox_feats_encoder + bbox_feats_fpn * factor
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results




    