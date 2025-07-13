import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, build_conv_layer
import torch.utils.checkpoint as checkpoint
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import os
from .flash_attention import FlashAttention

class UPBlock(nn.Module):
    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super(UPBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels // 2)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
        self.inner_attn = FlashAttention(attention_dropout=0.0, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3,
                                                                                  4)  # 3, B, N, num_heads, head_dim
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        qkv = torch.stack([q, k, v], dim=2).to(torch.float16)
        x, _ = self.inner_attn(qkv, key_padding_mask=None, causal=False)
        x = x.reshape(B, N, C).to(x_dtype)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 qkv_bias: bool = True,
                 qk_norm: bool = True,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, N, head_dim
        kv = self.kv(y.to(x_dtype)).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim: int,
                 mlp_ratio: int = 4,
                 drop: float = 0.) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 init_values: float = 1e-5,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionBlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class TransformerCrossLayer(nn.Module):
    def __init__(self, dim: int,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.cross_attn = CrossAttentionBlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm4 = norm_layer(dim)
        self.mlp = FFN(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.self_attn = AttentionBlock(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x: torch.Tensor, learnable_query: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)
        x = x + self.cross_attn(self.norm1(x), self.norm2(learnable_query))
        x = x + self.self_attn(self.norm3(x))
        x = x + self.mlp(self.norm4(x))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class EgoAttention(nn.Module):
    def __init__(self, dim: int,
                 num_layers: int,
                 num_heads: int = 16,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 qk_norm: bool = False,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.EgoAttention = nn.ModuleList()
        for i in range(num_layers):
            self.EgoAttention.append(
                TransformerCrossLayer(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                                      drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer))

    def forward(self, x: torch.Tensor, learnable_query: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x.permute(0, 2, 3, 1)
        else:
            for i, layer in enumerate(self.EgoAttention):
                x = layer(x, learnable_query)
            return x.permute(0, 2, 3, 1)


class DownSampleCross(nn.Module):
    def __init__(self, in_channels, num_layers=2, norm_cfg=dict(type='SyncBN', requires_grad=True),
                 use_ego_attention=False, num_ego_layers=3):
        super(DownSampleCross, self).__init__()
        # check CUDA_VISIBLE_DEVICES, if >= 2, then use SyncBN, otherwise use BN
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            if norm_cfg['type'] == 'SyncBN':
                if len(os.environ.get('CUDA_VISIBLE_DEVICES').split(',')) == 1:
                    norm_cfg = dict(type='BN', requires_grad=True)


        down_sample_layers = []
        inter_channels = in_channels
        for i in range(num_layers):
            down_sample_layers.append(nn.Sequential(
                build_conv_layer(None, inter_channels, inter_channels, 3, stride=1, padding=1, bias=False),
                build_norm_layer(norm_cfg, inter_channels)[1],
                nn.LeakyReLU(inplace=True)
            ))
            inter_channels = inter_channels * 2
            down_sample = nn.Sequential(
                build_conv_layer(None, inter_channels // 2, inter_channels, 1, stride=2, padding=0, bias=False),
                build_norm_layer(norm_cfg, inter_channels)[1])
            down_sample_layers.append(
                BasicBlock(inter_channels // 2, inter_channels, stride=2, norm_cfg=norm_cfg, downsample=down_sample))
            down_sample_layers.append(nn.Sequential(
                build_conv_layer(None, inter_channels, inter_channels, 3, stride=1, padding=1, bias=False),
                build_norm_layer(norm_cfg, inter_channels)[1],
                nn.LeakyReLU(inplace=True)
            ))
        down_sample_layers.append(nn.Sequential(
            TransformerEncoderLayer(inter_channels, num_heads=16, mlp_ratio=4, drop=0.1),
        ))

        self.down_sample = nn.Sequential(*down_sample_layers)

        up_sample_layers = []
        up_sample_layers.append(nn.Sequential(
            TransformerEncoderLayer(inter_channels, num_heads=16, mlp_ratio=4, drop=0.1),
        ))
        if use_ego_attention:
            self.ego_attention = EgoAttention(inter_channels, num_layers=num_ego_layers, num_heads=16, mlp_ratio=4,
                                              drop=0.1)

        for i in range(num_layers):
            up_sample_layers.append(nn.Sequential(
                build_conv_layer(None, inter_channels, inter_channels, 3, stride=1, padding=1, bias=False),
                build_norm_layer(norm_cfg, inter_channels)[1],
                nn.LeakyReLU(inplace=True)
            ))
            up_sample_layers.append(UPBlock(inter_channels, norm_cfg=norm_cfg))
            inter_channels = inter_channels // 2
            up_sample_layers.append(nn.Sequential(
                build_conv_layer(None, inter_channels, inter_channels, 3, stride=1, padding=1, bias=False),
                build_norm_layer(norm_cfg, inter_channels)[1],
                nn.LeakyReLU(inplace=True)
            ))

        self.up_sample = nn.Sequential(*up_sample_layers)

        assert inter_channels == in_channels, f"inter_channels {inter_channels} != in_channels {in_channels}"

        # change the padding mode to 'reflect' for conv in self.down_sample and self.up_sample
        for m in self.down_sample.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'replicate'
        for m in self.up_sample.modules():
            if isinstance(m, nn.Conv2d):
                m.padding_mode = 'replicate'

    def forward(self, x, out_query=None):
        down_x = self.down_sample(x)
        recon_x = self.up_sample(down_x)

        assert x.shape == recon_x.shape, f"x shape {x.shape} != recon_x shape {recon_x.shape}"
        return x, recon_x, down_x
