from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from re import I
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import cv2

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import pickle
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from datetime import datetime
from typing import List, Optional, Tuple
import mmcv
from matplotlib import pyplot as plt
from chamferdist import ChamferDistance

chamfer_distance = ChamferDistance()

count = 0
chamfer_dis_list = []
psnr_list = []
depth_l1_list = []


@DETECTORS.register_module()
class UVTRSSLBase(MVXTwoStageDetector):
    """UVTRBEVFormer."""

    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, "depth_head") and self.depth_head is not None

    @force_fp32()
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_voxel_encoder or pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        if not self.pts_fp16:
            voxel_features = voxel_features.float()

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"))
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if hasattr(self, "img_backbone"):
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(
                img=img, img_metas=img_metas, img_feats=img_feats
            )
        else:
            img_feats, img_depth = None, None

        if hasattr(self, "pts_voxel_encoder"):
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None

        return pts_feats, img_feats, img_depth

    @auto_fp16(apply_to=("img"))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None or not self.with_depth_head:
            return None
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        depth = []
        for _feat in img_feats:
            _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
            _depth = _depth.softmax(dim=1)
            depth.append(_depth)
        return depth

    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    "num of augmentations ({}) != num of image meta ({})".format(
                        len(points), len(img_metas)
                    )
                )

        assert num_augs == 1
        if not isinstance(img_metas[0], list):
            img_metas = [img_metas]
        if not isinstance(img, list):
            img = [img]
        results = self.simple_test(img_metas[0], points, img[0])

        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        # self.train()
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        batch_rays = self.pts_bbox_head.sample_rays_test(points, img.unsqueeze(0), img_metas)
        results = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth
        )
        H, W = img_metas[0]["ori_shape"][0][0], img_metas[0]["ori_shape"][0][1]
        num_cam = len(img_metas[0]["img_shape"])
        # mean = [123.675, 116.28, 103.53]
        # std = [1, 1, 1]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        global chamfer_dis_list
        global depth_l1_list

        cmap = plt.get_cmap('Spectral_r')

        img_rgb = (img[:, :, :H, :W].cpu().permute(0, 2, 3, 1).numpy() * std + mean).astype(np.uint8)

        # visualize the lidar
        lidar_rays, cam_rays = batch_rays
        scale_factor = lidar_rays[0]["scale_factor"].astype(np.float32)
        llm_render_depth = results[0]["depth"] / scale_factor
        lidar_ray_o = lidar_rays[0]["ray_o"].detach().cpu().numpy()
        lidar_ray_d = lidar_rays[0]["ray_d"].detach().cpu().numpy()
        lidar_pred = lidar_ray_d * llm_render_depth.detach().cpu().numpy() + lidar_ray_o
        lidar_gt = lidar_ray_d * lidar_rays[0]["depth"].detach().cpu().numpy() / scale_factor + lidar_ray_o
        depth_l1 = torch.abs(lidar_rays[0]["depth"] - results[0]["depth"]).mean()
        lidar_error = np.abs((lidar_pred - lidar_gt) / lidar_gt).mean()
        chamfer_dis = self.compute_chamfer_distance(torch.from_numpy(lidar_pred).float(),
                                                    torch.from_numpy(lidar_gt).float(), 'cuda')
        chamfer_dis_list.append(chamfer_dis.item())
        depth_l1_list.append(depth_l1.item())
        lidar_gt_vis = self.visualize_lidar(lidar_gt)
        lidar_pred_vis = self.visualize_lidar(lidar_pred)

        config_path = os.environ.get("IMG_SAVE_CONFIG_PATH", "debug")
        formatted_time = os.environ.get("TIME_STR", "2024")
        BGR_FORMAT = int(os.environ.get("BGR_FORMAT", "1"))
        if config_path is None or formatted_time is None:
            raise ValueError("IMG_SAVE_CONFIG_PATH or TIME_STR is not set")
        dir_name = config_path.split('/')[-1].split('.py')[0]
        dir_name = os.path.join(dir_name, formatted_time)
        save_dir = os.path.join("./outputs/", dir_name)
        os.makedirs(save_dir, exist_ok=True)
        gt_image = []
        for i in range(num_cam):
            rgb_i = img_rgb[i]
            if BGR_FORMAT:
                rgb_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2RGB)
            # resize the image to (H//2, W//2)
            rgb_i = cv2.resize(rgb_i, (W // 2, H // 2))
            gt_image.append(rgb_i)
        gt_image = [gt_image[i] for i in [2, 0, 1, 4, 3, 5]]
        gt_image_1 = cv2.hconcat(gt_image[:len(gt_image) // 2])
        gt_image_2 = cv2.hconcat(gt_image[len(gt_image) // 2:])

        save_image = cv2.vconcat([gt_image_1, gt_image_2])

        # copy a blank image with the same size as save_image and put gpt_reply on it, auto change line
        lidar_gt_vis = cv2.resize(lidar_gt_vis, (save_image.shape[0], save_image.shape[0]))
        lidar_pred_vis = cv2.resize(lidar_pred_vis, (save_image.shape[0], save_image.shape[0]))
        lidar_gt_vis = cv2.cvtColor(lidar_gt_vis, cv2.COLOR_BGR2RGB)
        lidar_gt_vis[-1, :] = 0
        lidar_pred_vis = cv2.cvtColor(lidar_pred_vis, cv2.COLOR_BGR2RGB)
        # add the lidar_error at the bottom center of the lidar_pred_vis
        cv2.putText(lidar_pred_vis, f"avg_error: {lidar_error * 100:.2f}%, chamfer: {chamfer_dis:.2f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(lidar_gt_vis, f"depth_l1: {depth_l1:.4f}",
                    (15, lidar_pred_vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        save_image = cv2.hconcat(
            [save_image, cv2.hconcat([lidar_gt_vis, lidar_pred_vis])])

        cv2.imwrite(f"{save_dir}/{img_metas[0]['sample_idx']}.png", save_image)

        print(f"save to {save_dir}/{img_metas[0]['sample_idx']}.png")
        global count
        count += 1
        print("#" * 10)
        print("file count: ", count)
        print("chamfer_dis: ", chamfer_dis)
        print("depth_l1: ", depth_l1)
        print("#" * 10)

        if count % 20 == 0:
            print("-" * 10)
            print("avg chamfer_dis: ", sum(chamfer_dis_list) / len(chamfer_dis_list))
            print("avg_depth_l1: ", sum(depth_l1_list) / len(depth_l1_list))
            print("-" * 10)
            with open(f"{save_dir}/results.txt", 'a') as f:
                f.write(
                    f"for first {count} files, "
                    f"avg chamfer_dis: {(sum(chamfer_dis_list) / len(chamfer_dis_list)):.3f}, "
                    f"avg_depth_l1: {(sum(depth_l1_list) / len(depth_l1_list)):.4f}\n")

        if count == 150:
            if len(chamfer_dis_list) != 0:
                avg_chamfer_dis = sum(chamfer_dis_list) / len(chamfer_dis_list)
                print("avg_chamfer_dis: ", avg_chamfer_dis)
                with open(f"{save_dir}/results.txt", 'a') as f:
                    f.write(f"\navg_chamfer_dis: {avg_chamfer_dis:.3f} for all {count} files")
                    f.write(f"\navg_depth_l1: {sum(depth_l1_list) / len(depth_l1_list):.4f} for all {count} files")

            # print the context in the results.txt to a png file
            with open(f"{save_dir}/results.txt", 'r') as f:
                txt = f.read()
            img = self.text_to_image_cv2(txt, save_image)
            cv2.imwrite(f"{save_dir}/fid_results_1.png", img)
            cv2.imwrite(f"{save_dir}/fid_results_2.png", img)

            exit()

        results = []
        return results

    def visualize_lidar(
            self,
            lidar: Optional[np.ndarray] = None,
            *,
            xlim: Tuple[float, float] = (-54, 54),
            ylim: Tuple[float, float] = (-54, 54),
            color: Optional[Tuple[int, int, int]] = None,
            radius: float = 0.5,
            thickness: float = 25,
    ):
        fig = plt.figure(figsize=(7.68, 7.68))

        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()
        # make the scatter plot's color align with the lidar's height
        if color is None:
            color = lidar[:, 2].clip(-3, 5)
            color = (color - color.min()) / (color.max() - color.min())

        if lidar is not None:
            plt.scatter(
                lidar[:, 0],
                lidar[:, 1],
                s=3.14 * radius ** 2,
                c=color,
                cmap="viridis",
            )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data

    def text_to_image_cv2(self, text, image_size, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, font_color=(0, 0, 0),
                          line_type=3, ):
        # Create a blank image with white background
        img = np.ones_like(image_size, np.uint8) * 255
        # Initialize starting position
        y0, dy = 100, 50  # Start at y=y0, and move down by dy pixels for each new line

        # Split text into lines to fit within the image width
        lines = text.split('\n')

        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(img, line, (400, y), font, font_scale, font_color, line_type)
        return img

    def compute_chamfer_distance(self, pred_pcd, gt_pcd, device):
        cd_forward, cd_backward, CD_info = chamfer_distance(
            pred_pcd[None, ...].to(device),
            gt_pcd[None, ...].to(device),
            bidirectional=True,
            reduction='sum')

        chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
        return chamfer_dist_value / 2.0

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        pts_feats, img_feats, img_depths = multi_apply(
            self.extract_feat, points, imgs, img_metas
        )
        return pts_feats, img_feats, img_depths
