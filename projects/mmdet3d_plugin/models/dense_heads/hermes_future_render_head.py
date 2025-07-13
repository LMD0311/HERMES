import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .down_util.down_sample_cross import DownSampleCross
from ..internvl_chat.internvl.train.internlm2_chat import HERMESLLM
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from .render_utils import models
from .render_utils.rays import RayBundle
from ..utils import Uni3DViewTrans, sparse_utils
from .render_head_bevformer import BEVFormerRenderHead
import pickle
from .. import utils
import numpy as np

COUNT = 0


@HEADS.register_module()
class HERMESFutureRenderHead(BEVFormerRenderHead):
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
                 loss_weight=None,
                 use_can_bus=False,
                 llm_cfg=None,
                 use_atten_pool=True,
                 use_avg_pool=False,
                 use_max_pool=False,
                 downsample_scale=2,
                 num_ego_layers=3,
                 future_generate=True,
                 pred_can_bus=False,
                 **kwargs):
        super(HERMESFutureRenderHead, self).__init__(*args, **kwargs)

        self.loss_weight = loss_weight
        self.use_atten_pool = use_atten_pool
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool
        self.future_generate = future_generate
        self.bev_channels = self.embed_dims * (2 ** downsample_scale)
        self.llm_cfg = llm_cfg
        self.chat_cfg = llm_cfg.chat_config
        self.init_llm()
        if hasattr(self.llm_cfg, "num_learnable_query") and self.llm_cfg.num_learnable_query > 0:
            self.num_learnable_query = self.llm_cfg.num_learnable_query
            self.pool = nn.AdaptiveMaxPool1d(self.num_learnable_query)
            # assert use_can_bus is True, "default use_can_bus when num_learnable_query > 0 for future prediction"
            # query_pos is a positional encoding for the learnable queries
            self.query_pos = nn.Parameter(
                torch.randn(self.num_learnable_query * (len(self.frames) - 1), self.bev_channels) / (
                    self.bev_channels) ** 0.5) if len(self.frames) > 1 else nn.Parameter(
                torch.randn(self.num_learnable_query * 3, self.bev_channels) / (self.bev_channels) ** 0.5)
        else:
            self.num_learnable_query = 0
            self.query_embedding = None
        self.frame_embedding = nn.Parameter(
            torch.randn([len(self.frames), self.bev_channels]) / (self.bev_channels) ** 0.5) if len(
            self.frames) > 1 else None

        self.use_can_bus = use_can_bus
        self.pred_can_bus = pred_can_bus
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims * 2),
                nn.LayerNorm(self.embed_dims * 2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.embed_dims * 2, self.bev_channels),
                nn.LayerNorm(self.bev_channels),
                nn.LeakyReLU(inplace=True),
            )
            xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

        if self.use_downsample:
            use_ego_attention = False if num_ego_layers == 0 or self.num_learnable_query == 0 else True
            self.down_sample = DownSampleCross(self.in_channels, num_layers=downsample_scale,
                                               use_ego_attention=use_ego_attention, num_ego_layers=num_ego_layers)

    def llm_loss_mse(self, pred, gt):
        loss_fn = nn.MSELoss(reduction="sum")
        return loss_fn(pred, gt) * self.loss_weight.mse_loss

    def loss(self, preds_dict, targets):
        lidar_targets, _ = targets
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

    def init_llm(self):
        self.llm = HERMESLLM(model_path=self.llm_cfg.llm_path, load_weight=self.llm_cfg.load_internvl_weight,
                             set_lora=self.llm_cfg.set_lora, is_pretraining=self.llm_cfg.is_pretraining,
                             chat_cfg=self.chat_cfg, attention_type=self.llm_cfg.attention_type,
                             img_length=self.llm_cfg.img_length,
                             num_learnable_query=self.llm_cfg.num_learnable_query \
                                 if hasattr(self.llm_cfg, "num_learnable_query") else 0,
                             input_dim=self.bev_channels
                             )
        self.llm.create_llm()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def is_weight_equal(self, w1, w2):
        return (w1 != w2).sum() == 0

    def change_para(self, w1, w2):
        for name, param in w1['state_dict'].items():
            if "pts_bbox_head.down_sample" in name:
                print(name, 'CHANGED')
                w2['state_dict'][name] = param
        return w2

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, rays, img_metas, img_depth, input_bev_embed=None, prev_bev_down=None,
                prev_bev=None, gt_bev=None,
                only_bev=False,
                fusion_prev=False, prev_img_metas=None,
                conv=None, llm_pred_bev=None):
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
        # we only support single batch size for now
        bs = 1
        dtype = torch.float16
        B, C, H, W = input_bev_embed.shape
        assert len(self.frames) == len(rays), "frame number should be equal to rays number"
        len_future = len(self.frames) - 1 if len(self.frames) > 1 else 3

        if self.use_downsample:
            input_bev_embed = input_bev_embed.permute(0, 2, 3, 1).to(self.llm.torch_dtype)
            query_embedding = None
            if self.num_learnable_query > 0:
                with torch.cuda.amp.autocast():
                    query_embedding = self.pool(input_bev_embed.view(B, -1, C).permute(0, 2, 1)
                                                ).permute(0, 2, 1).repeat(1, len_future, 1).contiguous()
                    if self.use_can_bus:
                        query_embedding = torch.cat(
                            [
                                query_embedding[:, self.num_learnable_query * i: self.num_learnable_query * (i + 1)]
                                + self.frame_embedding[i + 1] + self.can_bus_mlp(
                                    torch.from_numpy(img_metas[self.frames[i + 1]]['can_bus']).to(
                                        input_bev_embed.device).to(torch.float32))
                                for i in range(len_future)
                            ],
                            dim=1).to(self.llm.torch_dtype)
                assert query_embedding.shape[1] == self.num_learnable_query * len_future
                query_embedding = (query_embedding + self.query_pos).to(self.llm.torch_dtype)
            llm_out = checkpoint(self.llm, input_bev_embed, conv, query_embedding, use_reentrant=False)
            llm_bev_embed = llm_out['out_bev']
            if self.training:
                self.loss_chat = llm_out['chat_loss']
            out_query = llm_out['out_query']
            if out_query is not None:
                with (torch.cuda.amp.autocast()):
                    out_query = out_query.view(-1, self.num_learnable_query, self.bev_channels)
                    if self.future_generate:
                        assert len(self.frames) > 1
                        bev_embed_down = llm_bev_embed.expand(len(self.frames), -1, -1, -1)
                        if self.use_can_bus:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :] + self.can_bus_mlp(
                                torch.cat([torch.from_numpy(img_metas[self.frames[i]]['can_bus']).to(
                                    bev_embed_down.device).to(torch.float32).unsqueeze(0)
                                           for i in range(len(self.frames))],
                                          dim=0))[:, None, None, :]
                        bev_embed_down_ = self.down_sample.ego_attention(
                            bev_embed_down[-out_query.shape[0]:, ...].permute(0, 3, 1, 2), out_query)
                        bev_embed_down_current = bev_embed_down[:1]
                        bev_embed_down = torch.cat([bev_embed_down_current, bev_embed_down_], dim=0)
                    else:
                        # when not future_generate, the pred_can_bus should be True to avoid unused parameters
                        bev_embed_down = llm_bev_embed
            else:
                if len(self.frames) > 1:
                    with torch.cuda.amp.autocast():
                        bev_embed_down = llm_bev_embed.expand(len(self.frames), -1, -1, -1)
                        if self.use_can_bus:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :] + self.can_bus_mlp(
                                torch.cat([torch.from_numpy(img_metas[self.frames[i]]['can_bus']).to(
                                    bev_embed_down.device).to(torch.float32).unsqueeze(0)
                                           for i in range(len(self.frames))],
                                          dim=0))[:, None, None, :]
                        else:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :]
                else:
                    bev_embed_down = llm_bev_embed

            uni_feats = checkpoint(self.down_sample.up_sample, bev_embed_down.permute(0, 3, 1, 2))
            # uni_feats = self.down_sample.up_sample(bev_embed_down.permute(0, 3, 1, 2))
            uni_feats = checkpoint(self.bev_upsample, uni_feats)
            uni_feats = checkpoint(self.render_conv, uni_feats)
            # uni_feats = self.render_conv(self.bev_upsample(uni_feats))
            uni_feats_dict = {}
            for i in range(len(self.frames)):
                uni_feats_dict[self.frames[i]] = uni_feats[i]

        else:
            uni_feats = self.get_uni_feats(img_metas, input_bev_embed, only_bev, bs, dtype, fusion_prev=fusion_prev)
            if only_bev:
                return uni_feats

        batch_ret = []
        for bs_idx in range(bs):
            for frame_idx in self.frames:
                lidar_rays, cam_rays = rays[frame_idx]
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
                        lidar_ray_bundle, None, uni_feats_dict[frame_idx], points=scaled_points
                    )
                else:
                    lidar_ray_bundle = RayBundle(
                        origins=i_ray_o, directions=i_ray_d, depths=None
                    )
                    preds_dict = self.render_model(
                        lidar_ray_bundle, None, uni_feats_dict[frame_idx]
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

    def get_uni_feats(self, img_feats, img_metas, prev_bev, only_bev, bs, dtype, fusion_prev=False):
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            img_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )  # (B, bev_h*bev_w, embed_dims)

        if only_bev is not None and only_bev:
            return bev_embed

        bev_embed = bev_embed.permute(0, 2, 1).contiguous().view(bs, -1, self.bev_h,
                                                                 self.bev_w)  # (B, embed_dims, bev_h, bev_w)

        if self.use_downsample:
            bev_embed_ori, bev_embed_recon = self.down_sample(bev_embed)
            bev_embed = bev_embed_recon.clone()

        bev_embed = self.bev_upsample(bev_embed)

        uni_feats = self.render_conv(bev_embed)
        if self.use_downsample:
            return uni_feats, bev_embed_ori, bev_embed_recon
        else:
            return uni_feats


if __name__ == '__main__':
    from ipdb import set_trace

    set_trace()
