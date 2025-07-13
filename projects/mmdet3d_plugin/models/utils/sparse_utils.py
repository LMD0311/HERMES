import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2dPack
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp

_cur_voxel_coords: torch.Tensor = None  # [N1+N2+..., 4], [bs_idx, z, y, x]
_cur_active_voxel: torch.Tensor = None  # [N1+N2+...,]
_cur_voxel_scale: int = None
_cur_active: torch.Tensor = None  # B1hw


# todo: try to use `gather` for speed?
def _get_active_ex_or_ii(H, returning_active_ex=True):
    downsample_raito = H // _cur_active.shape[2]
    active_ex = _cur_active.repeat_interleave(downsample_raito, 2).repeat_interleave(
        downsample_raito, 3
    )
    return (
        active_ex
        if returning_active_ex
        else active_ex.squeeze(1).nonzero(as_tuple=True)
    )  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(
        H=x.shape[2], returning_active_ex=True
    )  # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[
        ii
    ]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(
        nc
    )  # use BN1d to normalize this flatten feature `nc`

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseDCNV2(ModulatedDeformConv2dPack):
    forward = sp_conv_forward


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward


class SparseLayerNorm(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, eps=1e-6, data_format="channel_last", sparse=True
    ):
        if data_format not in ["channel_last", "channel_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4:  # BHWC or BCHW
            if self.data_format == "channel_last":  # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseLayerNorm, self).forward(nc)

                    x = torch.zeros_like(x)
                    x[ii] = nc.to(x.dtype)
                    return x
                else:
                    return super(SparseLayerNorm, self).forward(x)
            else:  # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], returning_active_ex=False)
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseLayerNorm, self).forward(nc)

                    x = torch.zeros_like(bhwc)
                    x[ii] = nc.to(x.dtype)
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseLayerNorm, self).forward(x)


class SparseConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(
        self,
        in_channels,
        dw_conv_cfg=dict(kernel_size=7, padding=3),
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        with_cp=False,
        sparse=True,
    ):
        super().__init__()
        self.with_cp = with_cp

        self.dwconv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg
        )
        self.norm = SparseLayerNorm(in_channels, eps=1e-6, sparse=sparse)

        mid_channels = int(mlp_ratio * in_channels)
        self.pwconv1 = nn.Linear(in_channels, mid_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mid_channels, in_channels)

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((in_channels)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.sparse = sparse

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.dwconv(x)

            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            x = x.permute(0, 3, 1, 2)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            if self.sparse:
                x *= _get_active_ex_or_ii(H=x.shape[2], returning_active_ex=True)

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


def dense_model_to_sparse(m, name="", verbose=False, sbn=False):
    oup = m
    if isinstance(m, nn.Conv2d):
        bias = m.bias is not None
        oup = SparseConv2d(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=bias,
            padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, ModulatedDeformConv2dPack):
        bias = m.bias is not None
        oup = SparseDCNV2(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            deform_groups=m.deform_groups,
            bias=bias,
        )
        oup.conv_offset.weight.data.copy_(m.conv_offset.weight.data)
        oup.conv_offset.bias.data.copy_(m.conv_offset.bias.data)
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool2d):
        oup = SparseMaxPooling(
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            return_indices=m.return_indices,
            ceil_mode=m.ceil_mode,
        )
    elif isinstance(m, nn.AvgPool2d):
        oup = SparseAvgPooling(
            m.kernel_size,
            m.stride,
            m.padding,
            ceil_mode=m.ceil_mode,
            count_include_pad=m.count_include_pad,
            divisor_override=m.divisor_override,
        )
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(
            m.weight.shape[0],
            eps=m.eps,
            momentum=m.momentum,
            affine=m.affine,
            track_running_stats=m.track_running_stats,
        )
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        oup.running_mean.data.copy_(m.running_mean.data)
        oup.running_var.data.copy_(m.running_var.data)
        oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseLayerNorm):
        oup = SparseLayerNorm(m.weight.shape[0], eps=m.eps)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError
    else:
        for name, child in m.named_children():
            oup.add_module(
                name, dense_model_to_sparse(child, name, verbose=verbose, sbn=sbn)
            )
    return oup


def random_masking(B, H, W, ratio, device):
    len_keep = round(H * W * (1 - ratio))
    idx = torch.rand(B, H * W).argsort(dim=1)
    idx = idx[:, :len_keep].to(device)  # (B, len_keep)
    # (B, 1, H, W)
    mask = (
        torch.zeros(B, H * W, dtype=torch.bool, device=device)
        .scatter_(dim=1, index=idx, value=True)
        .view(B, 1, H, W)
    )
    return mask
