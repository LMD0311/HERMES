import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.runner.base_module import BaseModule
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
from mmcv.cnn import xavier_init
from .render_utils import models
from .render_utils.rays import RayBundle
from ..utils import Uni3DViewTrans, sparse_utils
import pickle
from .. import utils
from .render_head_bevformer import BEVUpsample, BEVFormerRenderHead

@HEADS.register_module()
class HERMESRenderHead(BEVFormerRenderHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(HERMESRenderHead, self).__init__(*args, **kwargs)

    def loss(self, preds_dict, targets):
        lidar_targets, img_targets = targets
        batch_size = len(lidar_targets)
        loss_dict = {}
        for bs_idx in range(batch_size):
            if hasattr(self, "cur_iter"):
                self.render_model.cur_iter = self.cur_iter
            i_loss_dict = self.render_model.loss(preds_dict[bs_idx], lidar_targets[bs_idx], None)
            for k, v in i_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()
        return loss_dict

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, pts_feats, img_feats, rays, img_metas, img_depth, prev_bev=None, only_bev=False,
                fusion_prev=False):
        """Forward function.
        Args:
            img_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = img_feats[0].shape
        dtype = img_feats[0].dtype
        if self.use_downsample:
            uni_feats, bev_embed_ori, bev_embed_recon = self.get_uni_feats(img_feats, img_metas, prev_bev, only_bev, bs,
                                                                           dtype)
        else:
            uni_feats = self.get_uni_feats(img_feats, img_metas, prev_bev, only_bev, bs, dtype, fusion_prev=fusion_prev)
            if only_bev:
                return uni_feats
        batch_ret = []
        lidar_rays, cam_rays = rays
        for bs_idx in range(len(img_metas)):
            i_ray_o, i_ray_d, i_ray_depth = (
                lidar_rays[bs_idx]["ray_o"],
                lidar_rays[bs_idx]["ray_d"],
                lidar_rays[bs_idx].get("depth", None),
            )
            if self.training:
                scaled_points = lidar_rays[bs_idx]["scaled_points"]
                lidar_ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                )
                preds_dict = self.render_model(
                    lidar_ray_bundle, None, uni_feats[bs_idx], points=scaled_points
                )
                if self.use_auto_encoder or self.use_downsample:
                    bev_embed_ori_ = bev_embed_ori[bs_idx]
                    bev_embed_recon_ = bev_embed_recon[bs_idx]
                    preds_dict['bev_embed_ori'] = bev_embed_ori_
                    preds_dict['bev_embed_recon'] = bev_embed_recon_
            else:
                lidar_ray_bundle = RayBundle(
                    origins=i_ray_o, directions=i_ray_d, depths=None
                )
                preds_dict = self.render_model(
                    lidar_ray_bundle, None, uni_feats[bs_idx]
                )
            batch_ret.append(preds_dict)
        return batch_ret

    def sample_rays(self, pts, imgs, img_metas):

        lidar_ret = self.sample_lidar_rays(pts, img_metas)
        return lidar_ret, None

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_lidar_rays(self, pts, img_metas, test=False):
        """Get lidar ray
        Returns:
            lidar_ret: list of dict, each dict contains:
                ray_o: (num_rays, 3)
                ray_d: (num_rays, 3)
                depth: (num_rays, 1)
                scaled_points: (num_rays, 3)
        """
        lidar_ret = []
        for i in range(len(pts)):
            lidar_pc = pts[i]
            dis = torch.norm(lidar_pc[:, :3], p=2, dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                    dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            ego_mask = self.ray_sampler_cfg.get("ego_mask", None)
            if ego_mask is not None:
                ego_mask_ = torch.logical_and(
                    torch.logical_and(ego_mask[0] <= lidar_pc[:, 0],
                                   ego_mask[2] >= lidar_pc[:, 0]),
                    torch.logical_and(ego_mask[1] <= lidar_pc[:, 1],
                                   ego_mask[3] >= lidar_pc[:, 1]),
                )
                dis_mask = dis_mask & torch.logical_not(ego_mask_)
            lidar_pc = lidar_pc[dis_mask]
            lidar_points = lidar_pc[:, :3]
            lidar_origins = torch.zeros_like(lidar_points)
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_ret.append(
                {
                    "ray_o": lidar_origins * self.render_model.scale_factor,
                    "ray_d": lidar_directions,
                    "depth": lidar_ranges * self.render_model.scale_factor,
                    "scaled_points": lidar_points * self.render_model.scale_factor,
                    "scale_factor": self.render_model.scale_factor,
                }
            )
        return lidar_ret