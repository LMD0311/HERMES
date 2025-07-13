import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.ops import MultiScaleDeformableAttention
from torch.nn.init import normal_


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, input_channel, embed_dim=256, normalize=True, pc_range=None):
        super().__init__()
        self.position_embedding = nn.Sequential(
            nn.Linear(input_channel, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.normalize = normalize
        self.pc_range = pc_range
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, xyz):
        if self.normalize:
            assert self.pc_range is not None
            pc_range = xyz.new_tensor(self.pc_range)
            xyz = (xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
        position_embedding = self.position_embedding(xyz)
        return position_embedding


class Uni3DCrossAttn(BaseModule):
    """Implements the view transformer."""

    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        num_convs=3,
        cam_sweep_feq=12,
        kernel_size=(3, 3, 3),
        sweep_fusion=dict(type="sweep_sum"),
        keep_sweep_dim=True,
        embed_dim=128,
        num_levels=4,
        num_points=2,
        normalize=True,
        norm_cfg=None,
        use_for_distill=False,
        fp16_enabled=False,
        **kwargs
    ):
        super(Uni3DCrossAttn, self).__init__()
        if fp16_enabled:
            self.fp16_enabled = True
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.voxel_shape = voxel_shape

        self.reference_voxel = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(
                        pc_range[i], pc_range[i + 3], self.voxel_size[i], device="cuda"
                    )
                    for i in range(3)
                ]
            ),
            dim=-1,
        )
        assert (
            self.reference_voxel.shape[0] == voxel_shape[0]
            and self.reference_voxel.shape[1] == voxel_shape[1]
            and self.reference_voxel.shape[2] == voxel_shape[2]
        )

        self.voxel_embedding = nn.Embedding(
            self.voxel_shape[0] * self.voxel_shape[1] * self.voxel_shape[2], embed_dim
        )
        self.position_encoding = LearnablePositionalEncoding(
            3, embed_dim=embed_dim, normalize=normalize, pc_range=pc_range
        )
        self.level_embeds = nn.Parameter(torch.Tensor(num_levels, embed_dim))
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        if norm_cfg is None:
            norm_cfg = kwargs.get("norm_cfg", dict(type="BN"))
        self.sweep_fusion = sweep_fusion.get("type", "sweep_sum")
        self.keep_sweep_dim = keep_sweep_dim
        self.use_for_distill = use_for_distill

        if "GN" in norm_cfg["type"]:
            norm_op = nn.GroupNorm
        elif "SyncBN" in norm_cfg["type"]:
            norm_op = nn.SyncBatchNorm
        else:
            norm_op = nn.BatchNorm3d

        padding = tuple([(_k - 1) // 2 for _k in kernel_size])

        self.conv_layer = nn.ModuleList()
        for k in range(num_convs):
            self.conv_layer.append(
                nn.Sequential(
                    nn.Conv3d(
                        embed_dim,
                        embed_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=True,
                    ),
                    norm_op(embed_dim)
                    if "GN" not in norm_cfg["type"]
                    else norm_op(norm_cfg["num_groups"], embed_dim),
                    nn.ReLU(inplace=True),
                )
            )

        if "sweep_cat" in self.sweep_fusion:
            self.trans_conv = nn.Sequential(
                nn.Conv3d(
                    embed_dim * self.num_sweeps,
                    embed_dim,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.BatchNorm3d(embed_dim),
                nn.ReLU(inplace=True),
            )

        if "with_time" in self.sweep_fusion:
            self.cam_sweep_time = 1.0 / cam_sweep_feq
            self.time_conv = nn.Sequential(
                nn.Conv3d(embed_dim + 1, embed_dim, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm3d(embed_dim),
                nn.ReLU(inplace=True),
            )

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for layer in self.conv_layer:
            xavier_init(layer, distribution="uniform", bias=0.0)
        normal_(self.level_embeds)

    @auto_fp16(apply_to=("mlvl_feats", "img_depth"))
    def forward(self, mlvl_feats, img_depth, img_metas):
        """Forward function for `Uni3DViewTrans`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dim, h, w].
        """
        reference_voxel_cam, mask = self.coord_preparing(img_metas)
        voxel_space = self.feat_sampling(
            mlvl_feats, img_depth, reference_voxel_cam, mask
        )
        voxel_space = self.feat_encoding(voxel_space, img_metas)

        return voxel_space

    @force_fp32()
    def coord_preparing(self, img_metas):
        B = len(img_metas)
        # (B, X, Y, Z, 3)
        reference_voxel = self.reference_voxel.unsqueeze(0).repeat(B, 1, 1, 1, 1)

        lidar2img, img_shape, uni_rot_aug, img_rot_aug = [], [], [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            img_shape.append(img_meta["img_shape"])
            if "uni_rot_aug" in img_meta:
                uni_rot_aug.append(img_meta["uni_rot_aug"])
            if "img_rot_aug" in img_meta:
                img_rot_aug.append(img_meta["img_rot_aug"])

        lidar2img = reference_voxel.new_tensor(np.asarray(lidar2img))
        _, num_cam, num_sweep = lidar2img.shape[:3]
        lidar2img = lidar2img.flatten(1, 2)

        reference_voxel = torch.cat(
            (reference_voxel, torch.ones_like(reference_voxel[..., :1])), -1
        ).flatten(1, 3)
        # Conduct inverse voxel augmentation first
        if len(uni_rot_aug) > 0:
            uni_rot_aug = torch.stack(uni_rot_aug, dim=0).to(reference_voxel)
            reference_voxel = reference_voxel @ torch.inverse(uni_rot_aug)

        reference_voxel_cam = torch.matmul(
            lidar2img.unsqueeze(2), reference_voxel.unsqueeze(1).unsqueeze(-1)
        ).squeeze(-1)
        eps = 1e-5
        referenece_depth = reference_voxel_cam[..., 2:3].clone()
        mask = referenece_depth > eps

        reference_voxel_cam = reference_voxel_cam[..., 0:2] / torch.maximum(
            reference_voxel_cam[..., 2:3],
            torch.ones_like(reference_voxel_cam[..., 2:3]) * eps,
        )

        # transfer if have image-level augmentation
        if len(img_rot_aug) > 0:
            img_rot_aug = reference_voxel_cam.new_tensor(np.asarray(img_rot_aug))
            reference_voxel_cam = (
                torch.cat(
                    [reference_voxel_cam, torch.ones_like(reference_voxel_cam)], dim=-1
                )
                @ img_rot_aug
            )
            reference_voxel_cam = reference_voxel_cam[..., :2]

        img_shape = reference_voxel_cam.new_tensor(np.asarray(img_shape))
        Hs, Ws = img_shape[..., 0:1], img_shape[..., 1:2]
        reference_voxel_cam[..., 0] /= Ws
        reference_voxel_cam[..., 1] /= Hs

        # (B, N*C, X*Y*Z, 1)
        mask = (
            mask
            & (reference_voxel_cam[..., 0:1] > 0.0)
            & (reference_voxel_cam[..., 0:1] < 1.0)
            & (reference_voxel_cam[..., 1:2] > 0.0)
            & (reference_voxel_cam[..., 1:2] < 1.0)
        )

        reference_voxel_cam = reference_voxel_cam.view(B, num_cam, num_sweep, -1, 2)
        mask = mask.view(B, num_cam, num_sweep, -1)
        return reference_voxel_cam, mask

    @force_fp32(apply_to=("mlvl_feats"))
    def feat_sampling(self, mlvl_feats, img_depth, reference_voxel_cam, mask):
        B, num_cam, num_sweep = reference_voxel_cam.shape[:3]
        reference_voxel_cam, mask = reference_voxel_cam.flatten(0, 2), mask.flatten(
            0, 2
        )
        reference_voxel_embed = self.voxel_embedding.weight + self.position_encoding(
            self.reference_voxel.view(-1, 3)
        )

        (
            batch_reference_voxel_cam,
            batch_reference_voxel_embed,
        ), batch_mask_indices = batch_mask_sequence(
            [reference_voxel_cam, reference_voxel_embed], mask
        )

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            B, M, C, H, W = feat.size()
            spatial_shape = (H, W)
            feat = feat.flatten(-2, -1).permute(0, 1, 3, 2)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, dim=2).flatten(0, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
            ],
            dim=0,
        )

        batch_reference_voxel_embed = self.cross_attn(
            query=batch_reference_voxel_embed,
            key=feat_flatten,
            value=feat_flatten,
            reference_points=batch_reference_voxel_cam.unsqueeze(-2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        feats_volume = rebatch_mask_sequence(
            reference_voxel_cam, batch_reference_voxel_embed, batch_mask_indices
        )
        feats_volume = feats_volume.reshape(
            B, num_cam, num_sweep, *self.voxel_shape, -1
        ).sum(dim=1)
        count = torch.clamp(
            mask.reshape(B, num_cam, num_sweep, *self.voxel_shape, 1).sum(dim=1),
            min=1.0,
        )
        # average pooling
        feats_volume = (feats_volume / count).permute(0, 1, 5, 4, 3, 2)
        return feats_volume

    @auto_fp16(apply_to=("voxel_space"))
    def feat_encoding(self, voxel_space, img_metas):
        B, num_sweep = voxel_space.shape[:2]
        voxel_space = voxel_space.flatten(0, 1)

        if "with_time" in self.sweep_fusion:
            sweep_time = torch.stack(
                [torch.from_numpy(_meta["sweeps_ids"]) for _meta in img_metas], dim=0
            )
            sweep_time = self.cam_sweep_time * sweep_time[..., 0].to(
                device=voxel_space.device
            )
            sweep_time = sweep_time.view(-1, 1, 1, 1, 1).repeat(
                1, 1, *voxel_space.shape[-3:]
            )
            voxel_space = torch.cat([voxel_space, sweep_time], dim=1)
            voxel_space = self.time_conv(voxel_space)

        voxel_space = voxel_space.view(B, num_sweep, *voxel_space.shape[1:])
        if "sweep_sum" in self.sweep_fusion:
            voxel_space = voxel_space.sum(1)
        elif "sweep_cat" in self.sweep_fusion:
            voxel_space = voxel_space.flatten(1, 2)
            voxel_space = self.trans_conv(voxel_space)

        # used for distill
        out_before_relu = []
        for _idx, layer in enumerate(self.conv_layer):
            if self.use_for_distill:
                out_mid = layer[:-1](voxel_space)
                out_before_relu.append(out_mid.clone())
                voxel_space = layer[-1](out_mid)
            else:
                voxel_space = layer(voxel_space)

        if self.keep_sweep_dim:
            voxel_space = voxel_space.unsqueeze(1)

        if self.use_for_distill:
            voxel_space = {"final": voxel_space, "before_relu": out_before_relu}

        return voxel_space


def batch_mask_sequence(feats_list, mask):
    """
    Args:
        feats_list: [(B, N, C) or (N, C), ...]
        mask: (B, N)
    Returns:
        rebatch_feats_list: [(B, M, C), ...]
        mask_indices: [(M1,), (M2,), ...]
    """
    batch_size = mask.shape[0]
    mask_indices = []
    for bs_idx in range(batch_size):
        mask_indices.append(mask[bs_idx].nonzero(as_tuple=True)[0])
    max_len = max([len(each) for each in mask_indices])
    rebatch_feats_list = []
    for feats in feats_list:
        rebatch_feats = feats.new_zeros(
            [batch_size, max_len, feats.shape[-1]], dtype=feats.dtype
        )
        for bs_idx in range(batch_size):
            i_index = mask_indices[bs_idx]
            rebatch_feats[bs_idx, : len(i_index)] = (
                feats[bs_idx, i_index] if len(feats.shape) == 3 else feats[i_index]
            )
        rebatch_feats_list.append(rebatch_feats)
    return rebatch_feats_list, mask_indices


def rebatch_mask_sequence(feats, rebatch_feats, mask_indices):
    """
    Args:
        feats: (B, N, C)
        rebatch_feats: (B, M, C)
        mask_indices: [(M1,), (M2,), ...]
    Returns:
        new_feats: (B, N, C)
    """
    batch_size = feats.shape[0]
    new_feats = rebatch_feats.new_zeros(
        [batch_size, feats.shape[1], rebatch_feats.shape[-1]], dtype=rebatch_feats.dtype
    )
    for bs_idx in range(batch_size):
        i_index = mask_indices[bs_idx]
        new_feats[bs_idx, i_index] = rebatch_feats[bs_idx, : len(i_index)]
    return new_feats
