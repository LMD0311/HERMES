import re
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    build_transformer_layer_sequence,
)
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER.register_module()
class Uni3DTransformer(BaseModule):
    """
    Implements the UVTR transformer.
    """

    def __init__(self, decoder=None, fp16_enabled=False, **kwargs):
        super(Uni3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(
                m, UniCrossAttenV2
            ):
                m.init_weight()

    @auto_fp16(apply_to=("query", "value", "reference_points", "query_pos"))
    def forward(
        self,
        query,
        value,
        reference_points,
        query_pos,
        key_pos=None,
        attn_masks=None,
        reg_branches=None,
    ):
        inter_states, inter_references = self.decoder(
            query=query,
            key=value,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
        )
        return inter_states, inter_references


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniTransformerDecoderV2(TransformerLayerSequence):
    """
    Implements the decoder in UVTR transformer.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(UniTransformerDecoderV2, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self, query, *args, reference_points=None, reg_branches=None, **kwargs):
        """
        Forward function for `UniTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, *args, reference_points=reference_points, **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 3
                # tmp: (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., 0:2] = (
                    tmp[..., 0:2] + reference_points[..., 0:2]
                )
                new_reference_points[..., 2:3] = (
                    tmp[..., 4:5] + reference_points[..., 2:3]
                )
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@ATTENTION.register_module()
class UniCrossAttenV2(BaseModule):
    """
    Cross attention module in UVTR.
    """

    def __init__(
        self,
        embed_dims=256,
        dropout=0.1,
        norm_cfg=None,
        init_cfg=None,
        batch_first=False,
        fp16_enabled=False,
    ):
        super(UniCrossAttenV2, self).__init__(init_cfg)
        assert not batch_first
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)

        self.embed_dims = embed_dims
        self.attention_weights = nn.Linear(embed_dims, 1)

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.init_weight()
        if fp16_enabled:
            self.fp16_enabled = fp16_enabled

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    @auto_fp16(apply_to=("query", "key"))
    def forward(
        self,
        query,
        key,
        value,
        identity=None,
        query_pos=None,
        key_pos=None,
        reference_points=None,
        **kwargs
    ):
        # ATTENTION: reference_points is decoupled from sigmoid function!
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # change to (bs, num_query, embed_dims)
        query = query.permute(1, 0, 2)

        # change to (bs, num_query, num_points)
        attention_weights = self.attention_weights(query).sigmoid()
        # normalize X, Y, Z to [-1,1]
        reference_points_voxel = (reference_points.sigmoid() - 0.5) * 2

        # value: (B, C, Z, Y, X) or (B, C, Y, X)
        # without height
        if len(value.shape) == 4:
            # sample image feature in bev space
            sample_embed = F.grid_sample(
                value, reference_points_voxel[:, None, :, :2]
            )  # (bs, 1, num_query, 2)
        else:
            # sample image feature in voxel space
            sample_embed = F.grid_sample(
                value, reference_points_voxel[:, None, None, :, :]
            )  # (bs, 1, 1, num_query, 3)
        sample_embed = sample_embed.view(
            len(query), sample_embed.shape[1], sample_embed.shape[-1]
        )  # (bs, C, num_query)
        sample_embed = sample_embed.permute(0, 2, 1)  # (bs, num_query, C)

        output = sample_embed * attention_weights

        # output = torch.nan_to_num(output)
        # avoid nan output
        # output = torch.where(torch.isnan(output), torch.zeros_like(output), output)

        output = output.permute(1, 0, 2)  # (num_query, bs, C)
        output = self.output_proj(output)
        # (num_query, bs, embed_dims)
        pos_feat = self.position_encoder(reference_points).permute(1, 0, 2)

        return self.dropout(output) + identity + pos_feat
