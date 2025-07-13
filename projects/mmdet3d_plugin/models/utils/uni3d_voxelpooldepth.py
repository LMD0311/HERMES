import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from projects.mmdet3d_plugin.ops import voxel_pool


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        depth_channels,
        use_dcn=True,
        use_aspp=True,
        aspp_mid_channels=-1,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        depth_conv_input_channels = mid_channels
        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type="DCN",
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )
                )
            )
        depth_conv_list.append(
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        )
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.depth_channels = depth_channels

    def forward(self, x):
        x = self.reduce_conv(x)
        depth = self.depth_conv(x)
        return depth


class Uni3DVoxelPoolDepth(BaseModule):
    """Implements the view transformer."""

    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        frustum_range,
        frustum_size,
        loss_cfg,
        num_convs=3,
        cam_sweep_feq=12,
        kernel_size=(3, 3, 3),
        sweep_fusion=dict(type="sweep_sum"),
        num_sweeps=1,
        keep_sweep_dim=True,
        embed_dim=128,
        norm_cfg=None,
        use_for_distill=False,
        fp16_enabled=False,
        **kwargs,
    ):
        super(Uni3DVoxelPoolDepth, self).__init__()
        if fp16_enabled:
            self.fp16_enabled = True
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.voxel_shape = voxel_shape
        self.loss_cfg = loss_cfg
        self.frustum_size = frustum_size
        self.frustum_range = frustum_range
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
        self.num_sweeps = num_sweeps

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

    def assign_depth_target(self, pts, imgs, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])

        lidar2img = pts[0].new_tensor(np.asarray(lidar2img))
        lidar2img = lidar2img.flatten(1, 2)

        depth_gt = []
        for bs_idx in range(len(pts)):
            i_pts = pts[bs_idx]
            i_pts = i_pts[i_pts[:, :2].norm(dim=-1) > self.loss_cfg.close_radius]
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_imgs = imgs[bs_idx]

            i_lidar2img = lidar2img[bs_idx]
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_depth = i_pts_cam[..., 2].clone()
            i_pts_mask = i_pts_depth > eps
            i_pts_cam = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
                & (i_pts_depth > self.frustum_range[2])
                & (i_pts_depth < self.frustum_range[5])
            )

            depth_map = i_imgs.new_zeros((i_imgs.shape[0], *i_imgs.shape[2:]))
            for c_idx in range(len(i_pts_mask)):
                # (M,) -> (Q,)
                j_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                coor, depth = (
                    torch.round(i_pts_cam[c_idx][j_pts_idx]),
                    i_pts_depth[c_idx][j_pts_idx],
                )
                ranks = coor[:, 0] + coor[:, 1] * i_imgs.shape[-1]
                sort = (ranks + depth / 100.0).argsort()
                coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
                kept = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
                kept[1:] = ranks[1:] != ranks[:-1]
                coor, depth = coor[kept], depth[kept]
                coor = coor.to(torch.long)
                depth_map[c_idx, coor[:, 1], coor[:, 0]] = depth
            depth_gt.append(depth_map)
        depth_gt = torch.stack(depth_gt, dim=0)

        return depth_gt

    def get_downsampled_gt_depth(self, depth_gt, downsample):
        B, NC, H, W = depth_gt.shape
        depth_gt = depth_gt.view(
            B * NC, H // downsample, downsample, W // downsample, downsample
        )
        depth_gt = depth_gt.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        depth_gt_tmp = torch.where(
            depth_gt == 0.0, 1e5 * torch.ones_like(depth_gt), depth_gt
        )
        depth_gt = torch.min(depth_gt_tmp, dim=-1).values
        depth_gt = ((depth_gt - self.frustum_range[2]) / self.frustum_size[2]).long()
        depth_gt = torch.where(
            (depth_gt < self.depth_dim) & (depth_gt >= 0),
            depth_gt + 1,
            torch.zeros_like(depth_gt),
        )
        depth_gt = (
            F.one_hot(depth_gt, num_classes=self.depth_dim + 1)
            .view(-1, self.depth_dim + 1)[..., 1:]
            .float()
        )
        return depth_gt

    @force_fp32(apply_to=("depth_preds", "pts", "imgs"))
    def loss(self, depth_preds, pts, imgs, img_metas):
        depth_gt = self.assign_depth_target(pts, imgs, img_metas)
        loss_dict = {}
        for i, d_pred in enumerate(depth_preds):
            if not self.loss_cfg.depth_loss_weights[i] > 0.0:
                continue
            downsample = depth_gt.shape[-1] // d_pred.shape[-1]
            d_gt = self.get_downsampled_gt_depth(depth_gt, downsample)
            d_pred = d_pred.permute(0, 2, 3, 1).contiguous().view(-1, d_gt.shape[-1])
            assert d_gt.shape[0] == d_pred.shape[0]
            fg_mask = torch.max(d_gt, dim=1).values > 0.0
            depth_loss = F.binary_cross_entropy(
                d_pred[fg_mask],
                d_gt[fg_mask],
                reduction="none",
            ).sum() / max(1.0, fg_mask.sum())
            loss_dict[f"loss_depth_{i}"] = (
                depth_loss * self.loss_cfg.depth_loss_weights[i]
            )
        return loss_dict

    @force_fp32()
    def coord_preparing(self, img_metas):
        B = len(img_metas)
        # (B, W, H, D, 3)
        frustum = self.frustum.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        W, H, D = frustum.shape[1:-1]

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])

        lidar2img = frustum.new_tensor(np.asarray(lidar2img))
        _, N, C = lidar2img.shape[:3]
        lidar2img = lidar2img.flatten(1, 2)

        frustum = (
            torch.cat([frustum, torch.ones_like(frustum[..., :1])], -1)
            .flatten(1, 3)
            .unsqueeze(1)
        )
        frustum[..., :2] *= frustum[..., 2:3]
        frustum = torch.matmul(
            torch.inverse(lidar2img).unsqueeze(2), frustum.unsqueeze(-1)
        ).squeeze(-1)

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
