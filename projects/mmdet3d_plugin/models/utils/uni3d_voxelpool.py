import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import Conv2d
from projects.mmdet3d_plugin.ops import voxel_pool


class Uni3DVoxelPool(BaseModule):
    """Implements the view transformer."""

    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        frustum_range,
        frustum_size,
        num_convs=3,
        cam_sweep_feq=12,
        kernel_size=(3, 3, 3),
        sweep_fusion=dict(type="sweep_sum"),
        keep_sweep_dim=True,
        embed_dim=128,
        norm_cfg=None,
        use_for_distill=False,
        fp16_enabled=False,
        **kwargs
    ):
        super(Uni3DVoxelPool, self).__init__()
        if fp16_enabled:
            self.fp16_enabled = True
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.voxel_shape = voxel_shape
        self.frustum = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(
                        frustum_range[i],
                        frustum_range[i + 3],
                        frustum_size[i],
                        device="cuda",
                    )
                    for i in range(3)
                ]
            ),
            dim=-1,
        )  # (W, H, D, 3)

        self.depth_dim = int((frustum_range[5] - frustum_range[2]) / frustum_size[2])

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

    def forward(self, mlvl_feats, img_depth, img_metas):
        """Forward function for `Uni3DViewTrans`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dim, h, w].
        """
        voxel_coords, mask = self.coord_preparing(img_metas)
        voxel_space = self.feat_sampling(mlvl_feats, img_depth, voxel_coords, mask)
        voxel_space = self.feat_encoding(voxel_space, img_metas)

        return voxel_space

    @force_fp32()
    def coord_preparing(self, img_metas):
        B = len(img_metas)

        frustum = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        W, H, D = frustum.shape[1:-1]

        lidar2img, uni_rot_aug, img_rot_aug = [], [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            if "uni_rot_aug" in img_meta:
                uni_rot_aug.append(img_meta["uni_rot_aug"])
            if "img_rot_aug" in img_meta:
                img_rot_aug.append(img_meta["img_rot_aug"])

        lidar2img = frustum.new_tensor(np.asarray(lidar2img))
        _, N, C = lidar2img.shape[:3]
        lidar2img = lidar2img.flatten(1, 2)

        frustum = (
            torch.cat([frustum, torch.ones_like(frustum[..., :1])], -1)
            .flatten(1, 3)
            .unsqueeze(1)
        )
        if len(img_rot_aug) > 0:
            img_rot_aug = frustum.new_tensor(np.asarray(img_rot_aug))
            frustum = frustum @ torch.inverse(img_rot_aug)

        frustum[..., :2] *= frustum[..., 2:3]
        frustum = torch.matmul(
            torch.inverse(lidar2img).unsqueeze(2), frustum.unsqueeze(-1)
        ).squeeze(-1)

        # Conduct inverse voxel augmentation first
        if len(uni_rot_aug) > 0:
            uni_rot_aug = torch.stack(uni_rot_aug, dim=0).to(frustum)
            frustum = frustum @ uni_rot_aug.unsqueeze(1)

        pc_range = frustum.new_tensor(self.pc_range)
        voxel_size = frustum.new_tensor(self.voxel_size)
        voxel_coords = ((frustum[..., :3] - pc_range[:3]) / voxel_size).int()
        batch_ix = torch.cat(
            [
                torch.full_like(voxel_coords[ix : ix + 1, ..., 0:1], ix)
                for ix in range(B)
            ],
            dim=0,
        )
        voxel_coords = torch.cat([batch_ix, voxel_coords], dim=-1)
        voxel_coords = (
            voxel_coords.view(B, N, C, W, H, D, 4)
            .permute(0, 1, 2, 5, 4, 3, 6)
            .contiguous()
        )

        mask = (
            (voxel_coords[..., 1] >= 0)
            & (voxel_coords[..., 1] < self.voxel_shape[0])
            & (voxel_coords[..., 2] >= 0)
            & (voxel_coords[..., 2] < self.voxel_shape[1])
            & (voxel_coords[..., 3] >= 0)
            & (voxel_coords[..., 3] < self.voxel_shape[2])
        )
        return voxel_coords, mask

    @force_fp32(apply_to=("mlvl_feats", "img_depth"))
    def feat_sampling(self, mlvl_feats, img_depth, voxel_coords, mask):
        assert (
            len(mlvl_feats) == len(img_depth) == 1
        ), "Only support single level feature"
        img_feats = mlvl_feats[0]
        img_depth = img_depth[0]
        feats = img_feats.flatten(0, 1).unsqueeze(-3) * img_depth.unsqueeze(1)

        B, N, C = voxel_coords.shape[:3]
        feats = feats.view(B, N, C, feats.shape[1], -1).permute(2, 0, 1, 4, 3)
        voxel_coords = voxel_coords.view(B, N, C, -1, 4).permute(2, 0, 1, 3, 4)
        mask = mask.view(B, N, C, -1).permute(2, 0, 1, 3)

        feats_volume = []
        for cur_vox, cur_mask, cur_feat in zip(voxel_coords, mask, feats):
            assert cur_mask.shape == cur_vox.shape[:-1] == cur_feat.shape[:-1]
            feats_volume.append(
                voxel_pool(cur_feat[cur_mask], cur_vox[cur_mask], B, *self.voxel_shape)
            )
        feats_volume = torch.stack(feats_volume, dim=1)
        feats_volume = feats_volume.permute(0, 1, 5, 4, 3, 2)
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
