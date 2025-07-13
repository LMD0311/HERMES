from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from .uvtr_ssl_base import UVTRSSLBase

count = 0


@DETECTORS.register_module()
class UVTRBEVFormer(UVTRSSLBase):
    """UVTRBEVFormer."""

    def __init__(
            self,
            num_levels=4,
            pts_voxel_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_fusion_layer=None,
            img_backbone=None,
            pts_backbone=None,
            img_neck=None,
            depth_head=None,
            pts_neck=None,
            pts_bbox_head=None,
            img_roi_head=None,
            img_rpn_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(UVTRBEVFormer, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels if img_neck is not None else self.pts_bbox_head.in_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if depth_head is not None:
                depth_dim = self.pts_bbox_head.view_trans.depth_dim
                dhead_type = depth_head.pop("type", "SimpleDepth")
                if dhead_type == "SimpleDepth":
                    self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                else:
                    self.depth_net = DepthNet(
                        out_channels, out_channels, depth_dim, **depth_head
                    )
            self.depth_head = depth_head
            self.num_levels = num_levels

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
            self, pts_feats, img_feats, points, img, img_metas, img_depth
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        batch_rays = self.pts_bbox_head.sample_rays(points, img, img_metas)
        out_dict = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )

        losses = self.pts_bbox_head.loss(out_dict, batch_rays)
        return losses

    def forward_train(self, points=None, img_metas=None, img=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats if self.num_levels is None else img_feats[:self.num_levels],
            points, img, img_metas, img_depth
        )
        losses.update(losses_pts)

        return losses

    def train_step(self, data, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if "_iter" in kwargs:
            self.cur_iter = kwargs["_iter"]
            self.pts_bbox_head.cur_iter = kwargs["_iter"]
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
