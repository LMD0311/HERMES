import copy
import os
from codecs import ignore_errors
from re import I
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from .uvtr_ssl_base import UVTRSSLBase
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple
import mmcv
from matplotlib import pyplot as plt
from chamferdist import ChamferDistance
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
import textwrap

chamfer_distance = ChamferDistance()

count = 0
chamfer_dis_list = []
chamfer_dis_list_gt = []
chamfer_dis_list_input_pred = []
depth_l1_list = []


def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


@DETECTORS.register_module()
class HERMES(UVTRSSLBase):
    """HERMES Detector."""

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
            frames=(0,),
            pre_bev_checkpoint=None,
            frame_loss_weight=None,
            text_loss_weight=None,
    ):
        super(HERMES, self).__init__(
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
        self.frames = frames
        self.pre_bev_checkpoint = pre_bev_checkpoint
        self.frame_loss_weight = frame_loss_weight
        self.text_loss_weight = text_loss_weight
        if frame_loss_weight is not None:
            assert len(frame_loss_weight) == len(frames)

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
            self, points, img, img_metas, img_depth, input_bev_embed, prev_bev_down=None, prev_bev=None,
            gt_bev_emb=None,
            prev_img_metas=None,
            conv=None
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
            batch_rays, img_metas, img_depth, input_bev_embed, prev_bev_down, prev_bev, gt_bev_emb,
            fusion_prev=False,
            prev_img_metas=prev_img_metas,
            conv=conv
        )

        losses = self.pts_bbox_head.loss(out_dict, batch_rays)
        if hasattr(self.pts_bbox_head, "loss_chat") and self.pts_bbox_head.loss_chat:
            losses.update({'pretrain_chat_loss': self.pts_bbox_head.loss_chat})
        return losses

    def forward_train(self, points=None, img_metas=None, img=None, **mono_input_dict):
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
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        target_points = {}
        for i in range(len(self.frames)):
            # 1. if the future frame is not in the img_dict, copy the current frame to the future frame
            if self.frames[i] not in img_dict.keys() and self.frames[i] > 0:
                assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])

            # 2. if the future scene is not the same as the current scene, copy the current scene to the future scene
            if "prev_bev_exists" in img_metas[self.frames[i]]:
                if self.frames[i] != 0 and img_metas[self.frames[i]]['prev_bev_exists'] == False:
                    assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                    img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                    img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])
            # 3. replace the target point with future point
            target_points[self.frames[i]] = img_metas[self.frames[i]]['points'].cuda()

        img = img_dict[0]
        img_dict.pop(self.frames[-1])

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(self.frames[-1])

        curr_img_meta = [copy.deepcopy(img_metas)[0], ]

        img = img.squeeze(1)
        _, curr_img_feats, _ = self.extract_feat(
            points=points, img=img.cuda(), img_metas=curr_img_meta
        )
        # input_bev_emb = self.pts_bbox_head.get_uni_feats(img_feats=curr_img_feats, img_metas=curr_img_meta,
        #                                                  prev_bev=None, only_bev=True, bs=1, dtype=torch.float16)
        input_bev_emb = checkpoint(self.pts_bbox_head.get_uni_feats, curr_img_feats, curr_img_meta, None, True, 1,
                                   torch.float16, use_reentrant=False)
        input_bev_emb = input_bev_emb.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h,
                                                               self.pts_bbox_head.bev_w)
        input_bev_emb = self.pts_bbox_head.down_sample.down_sample(input_bev_emb)

        losses = dict()
        img_depth = None
        batch_rays = {}
        for index in range(len(self.frames)):
            batch_rays[self.frames[index]] = self.pts_bbox_head.sample_rays([target_points[self.frames[index]], ],
                                                                            img, curr_img_meta)

        out_dict = self.pts_bbox_head(
            batch_rays, img_metas, img_depth, input_bev_emb, None, None, None,
            fusion_prev=False, prev_img_metas=None, conv=curr_img_meta[0]["conv"]
        )
        for index, frame in enumerate(self.frames):
            losses_pts = self.pts_bbox_head.loss([out_dict[index], ], batch_rays[frame])
            for key in losses_pts.keys():
                loss_weight = self.frame_loss_weight[index] if self.frame_loss_weight is not None else 1.0
                if loss_weight > 0:
                    losses[key + "_frame{}".format(frame)] = losses_pts[key] * loss_weight
        if hasattr(self.pts_bbox_head, "loss_chat") and self.pts_bbox_head.loss_chat:
            loss_weight = self.text_loss_weight if self.text_loss_weight is not None else 1.0
            if loss_weight > 0:
                losses.update({'pretrain_chat_loss': self.pts_bbox_head.loss_chat * loss_weight})
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

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        if isinstance(img[0], torch.Tensor):
            results = self.simple_test(img_metas[0], points[0], img[0])
        else:
            results = self.simple_test(img_metas[0].data[0], points[0].data[0], img[0].data[0])
        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        ignore_flag = False
        target_points = {}

        # if i > 0 not in img_dict.keys(), copy 0 to i
        for i in range(len(self.frames)):
            # 1. if the future frame is not in the img_dict, copy the current frame to the future frame
            if self.frames[i] not in img_dict.keys() and self.frames[i] > 0:
                assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])
                ignore_flag = True
            # 2. if the future scene is not the same as the current scene, copy the current scene to the future scene
            if self.frames[i] > 0 and img_metas[self.frames[i]]['prev_bev_exists'] == False:
                assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])
                ignore_flag = True
            # 3. replace the target point with future point
            target_points[self.frames[i]] = img_metas[self.frames[i]]['points'].cuda()

        img = img_dict[0]
        img_dict.pop(self.frames[-1])

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(self.frames[-1])

        curr_img_meta = [copy.deepcopy(img_metas)[self.frames[0]], ]

        img = img.squeeze(1)
        _, curr_img_feats, _ = self.extract_feat(
            points=points, img=img.cuda(), img_metas=curr_img_meta
        )
        with torch.cuda.amp.autocast():
            input_bev_emb = self.pts_bbox_head.get_uni_feats(img_feats=curr_img_feats, img_metas=curr_img_meta,
                                                             prev_bev=None, only_bev=True, bs=1, dtype=torch.float16)
            input_bev_emb = input_bev_emb.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h,
                                                                   self.pts_bbox_head.bev_w)
            input_bev_emb = self.pts_bbox_head.down_sample.down_sample(input_bev_emb)
        conv_dict = {}
        batch_rays = {}
        for index in range(len(self.frames)):
            batch_rays[self.frames[index]] = self.pts_bbox_head.sample_rays([target_points[self.frames[index]], ],
                                                                            img, curr_img_meta)

        curr_img_meta = [copy.deepcopy(img_metas)[self.frames[0]], ]

        # set the target point
        token = curr_img_meta[0]['sample_idx']
        conv_save = []
        question_list = []
        gt_answer_list = []
        conv_path = "./data/data_nusc/conv/val/"
        conv_path = os.path.join(conv_path, token + ".json")
        desc_path = "./data/data_nusc/desc/val/"
        desc_path = os.path.join(desc_path, token + ".json")
        with open(conv_path, 'r') as f:
            conv_gt = json.load(f)
            # add the questions to the question_list and the gt_answer to the gt_answer_list
            for i in range(len(conv_gt)):
                question_list.append(conv_gt[i]['question'])
                gt_answer_list.append(conv_gt[i]['answer'])
        with open(desc_path, 'r') as f:
            desc_gt = json.load(f)
            desc = desc_gt['description']
            action = desc_gt['action']
            question_list = question_list + ["What action should be taken in the current driving scenario?"]
            gt_answer_list = gt_answer_list + [action]
        question_list = ["Can you provide a summary of the current driving scenario based on the input images?"] \
                        + question_list
        gt_answer_list = [desc] + gt_answer_list

        question_list = question_list + ["Please try to imagine the future scene."]
        gt_answer_list = gt_answer_list + [
            "Sure, I can try to envision the future scene based on the understanding of the driving scenario."]

        assert len(question_list) == len(gt_answer_list)
        question_list[0] = "<image>\n" + question_list[0]

        history_list = []
        conv = []
        for i, question in enumerate(question_list):
            response, output = self.pts_bbox_head.llm.chat(
                tokenizer=None,
                pixel_values=input_bev_emb.permute(0, 2, 3, 1).to(self.pts_bbox_head.llm.torch_dtype),
                question=question,
                conv=None,
                generation_config=self.pts_bbox_head.chat_cfg,
                verbose=True,
                history=history_list
            )
            conv.append({'from': 'human', 'value': question})
            conv.append({'from': 'gpt', 'value': response})
            conv_save.append({
                "id": "{}".format(token),
                "question": question,
                "answer": response,
                "gt_answer": gt_answer_list[i]
            })

        results = self.pts_bbox_head(
            batch_rays, img_metas, None, input_bev_emb, fusion_prev=False,
            llm_pred_bev=None, conv=conv
        )

        output_dict = {}
        for index, frame in enumerate(self.frames):
            lidar_rays, _ = batch_rays[frame]
            scale_factor = lidar_rays[0]["scale_factor"].astype(np.float32)
            llm_render_depth = results[index]["depth"] / scale_factor
            lidar_ray_o = lidar_rays[0]["ray_o"].detach().cpu().numpy()
            lidar_ray_d = lidar_rays[0]["ray_d"].detach().cpu().numpy()
            lidar_pred = lidar_ray_d * llm_render_depth.detach().cpu().numpy() + lidar_ray_o
            lidar_gt = lidar_ray_d * lidar_rays[0][
                "depth"].detach().cpu().numpy() / scale_factor + lidar_ray_o
            depth_l1 = torch.abs(lidar_rays[0]["depth"] - results[index]["depth"]).mean()
            lidar_error = np.abs((lidar_pred - lidar_gt) / lidar_gt).mean()
            chamfer_dis = self.compute_chamfer_distance(torch.from_numpy(lidar_pred).float(),
                                                        torch.from_numpy(lidar_gt).float(), 'cuda')
            output_dict[frame] = {
                frame: {
                    "lidar_pred": lidar_pred,
                    "lidar_gt": lidar_gt,
                    "depth_l1": depth_l1,
                    "lidar_error": lidar_error,
                    "chamfer_dis": chamfer_dis,
                }
            }

        config_path = os.environ.get("IMG_SAVE_CONFIG_PATH", "stage3.py")
        formatted_time = os.environ.get("TIME_STR", "hermes_eval")
        if config_path is None or formatted_time is None:
            raise ValueError("IMG_SAVE_CONFIG_PATH or TIME_STR is not set")
        dir_name = config_path.split('/')[-1].split('.py')[0]
        dir_name = os.path.join(dir_name, formatted_time)
        save_dir = os.path.join("./outputs/", dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # save results to a json file
        results = {}
        for index, frame in enumerate(self.frames):
            results["chamfer_dis_frame_{}".format(frame)] = output_dict[frame][frame]["chamfer_dis"].item()
        results.update({
            "ignore_flag": ignore_flag,
            "desc": conv_save,
        })
        os.makedirs(f"{save_dir}/results", exist_ok=True)
        with open(f"{save_dir}/results/{img_metas[0]['sample_idx']}.json", 'w') as f:
            json.dump(results, f, indent=1)
        print(f"save results to {save_dir}/results/{img_metas[0]['sample_idx']}.json")

        if ignore_flag:
            print("ignore this frame")
        for index, frame in enumerate(self.frames):
            print("chamfer_dis_frame_{}:".format(frame), output_dict[frame][frame]["chamfer_dis"])

        print("#" * 10)

        results = [None, ]
        return results

    def compute_chamfer_distance(self, pred_pcd, gt_pcd, device):
        cd_forward, cd_backward, CD_info = chamfer_distance(
            pred_pcd[None, ...].to(device),
            gt_pcd[None, ...].to(device),
            bidirectional=True,
            reduction='sum')

        chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
        return chamfer_dist_value / 2.0
