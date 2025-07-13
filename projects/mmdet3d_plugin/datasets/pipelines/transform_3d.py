import numpy as np
import torch
import mmcv
import cv2
import mmdet3d
from mmdet.datasets.builder import PIPELINES
from mmcv.utils import build_from_cfg
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    box_np_ops,
)
from mmcv.parallel import DataContainer as DC
import json
import math
from transformers import AutoTokenizer, AutoModel
from copy import deepcopy
from transformers import AutoTokenizer
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm, preprocess_mpt,
                                    preprocess_phi3)
import os
import random


def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        results["pad_before_shape"] = [img.shape for img in results["img"]]
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class CropResizeMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results["resize_before_shape"] = [img.shape for img in results["img"]]
        resize_img_list = []
        for img in results["img"]:
            resize_img = np.copy(img)
            # unnormalize to [0-1]
            # img rgb
            # if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
            #     imgs[..., [0, 1, 2]] = imgs[..., [2, 1, 0]]  # bgr->rgb
            resize_img = resize_img[160 - 28: -28, :]
            # INTER_AREA is better for image decimation

            img_norm_cfg = results["img_norm_cfg"]
            mean, std = img_norm_cfg["mean"], img_norm_cfg["std"]
            # to_rgb = img_norm_cfg["to_rgb"]

            # rgb
            resize_img = (resize_img * std[None, None]) + mean[None, None]
            resize_img = cv2.resize(resize_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            resize_img /= 255.

            assert resize_img.shape[0] == 384 and resize_img.shape[1] == 800, f"resize_img.shape: {resize_img.shape}"
            resize_img_list.append(resize_img)

        results["resize_img"] = np.stack(resize_img_list, axis=0)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if np.random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class RandomResizeCropFlipMultiViewImage(object):
    def __init__(
            self,
            image_size,
            resize_scales=None,
            crop_sizes=None,
            flip_ratio=None,
            rot_angles=None,
            training=True,
    ):
        self.image_size = image_size
        self.flip_ratio = flip_ratio
        self.resize_scales = resize_scales
        self.crop_sizes = crop_sizes
        self.rot_angles = rot_angles
        self.training = training

    def _resize_img(self, results):
        img_scale_mat = []
        new_img = []
        for img in results["img"]:
            resize = float(self.image_size[1]) / float(
                img.shape[1]
            ) + np.random.uniform(*self.resize_scales)
            new_img.append(
                mmcv.imresize(
                    img,
                    (int(img.shape[1] * resize), int(img.shape[0] * resize)),
                    return_scale=False,
                )
            )
            img_scale_mat.append(
                np.array(
                    [[resize, 0, 0, 0], [0, resize, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_scale_mat"] = img_scale_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _crop_img(self, results):
        img_crop_mat = []
        new_img = []
        for img in results["img"]:
            crop_pos = np.random.uniform(0.0, 1.0)
            # crop from image bottom
            start_h = img.shape[0] - self.crop_sizes[0]
            start_w = (
                int(crop_pos * max(0, img.shape[1] - self.crop_sizes[1]))
                if self.training
                else max(0, img.shape[1] - self.crop_sizes[1]) // 2
            )
            new_img.append(
                img[
                start_h: start_h + self.crop_sizes[0],
                start_w: start_w + self.crop_sizes[1],
                ...,
                ]
            )
            img_crop_mat.append(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [-start_w, -start_h, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_crop_mat"] = img_crop_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _flip_img(self, results):
        img_flip_mat = []
        new_img = []
        for img in results["img"]:
            if np.random.rand() >= self.flip_ratio or (not self.training):
                new_img.append(img)
                img_flip_mat.append(
                    np.array(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        dtype=np.float32,
                    )
                )
            else:
                new_img.append(mmcv.imflip(img, "horizontal"))
                img_flip_mat.append(
                    np.array(
                        [
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [img.shape[1] - 1, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    )
                )

        results["img"] = new_img
        results["img_flip_mat"] = img_flip_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def _rotate_img(self, results):
        new_img = []
        img_rot_mat = []
        for img in results["img"]:
            # Rotation angle in degrees
            angle = np.random.uniform(*self.rot_angles)
            new_img.append(mmcv.imrotate(img, angle))
            h, w = img.shape[:2]
            c_x, c_y = (w - 1) * 0.5, (h - 1) * 0.5
            rot_sin, rot_cos = np.sin(angle / 180 * np.pi), np.cos(angle / 180 * np.pi)
            img_rot_mat.append(
                np.array(
                    [
                        [rot_cos, rot_sin, 0, 0],
                        [-rot_sin, rot_cos, 0, 0],
                        [
                            (1 - rot_cos) * c_x + rot_sin * c_y,
                            (1 - rot_cos) * c_y - rot_sin * c_x,
                            1,
                            0,
                        ],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                )
            )

        results["img"] = new_img
        results["img_rot_mat"] = img_rot_mat
        results["img_shape"] = [img.shape for img in results["img"]]
        return results

    def __call__(self, results):
        # TODO: aug sweep-wise or camera-wise?
        # resize image
        if self.resize_scales is not None:
            results = self._resize_img(results)
        # crop image
        if self.crop_sizes is not None:
            results = self._crop_img(results)
        # flip image
        if self.flip_ratio is not None:
            results = self._flip_img(results)
        # rotate image
        if self.rot_angles is not None:
            results = self._rotate_img(results)

        img_rot_aug = []
        for i in range(len(results["img"])):
            rot_mat = np.eye(4, dtype=np.float32)
            if "img_scale_mat" in results:
                rot_mat = results["img_scale_mat"][i].T @ rot_mat
            if "img_crop_mat" in results:
                rot_mat = results["img_crop_mat"][i].T @ rot_mat
            if "img_flip_mat" in results:
                rot_mat = results["img_flip_mat"][i].T @ rot_mat
            if "img_rot_mat" in results:
                rot_mat = results["img_rot_mat"][i].T @ rot_mat
            img_rot_aug.append(rot_mat)
        results["img_rot_aug"] = img_rot_aug

        num_cam, num_sweep = len(results["lidar2img"]), len(results["lidar2img"][0])
        img_rot_aug = np.concatenate(img_rot_aug, axis=0).reshape(
            num_cam, num_sweep, 4, 4
        )
        results["lidar2img"] = [
            img_rot_aug[_idx] @ results["lidar2img"][_idx]
            for _idx in range(len(results["lidar2img"]))
        ]
        return results


@PIPELINES.register_module()
class UnifiedRotScaleTransFlip(object):
    """
    Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(
            self,
            rot_range=[-0.78539816, 0.78539816],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            flip_ratio_bev_horizontal=0.0,
            flip_ratio_bev_vertical=0.0,
            shift_height=False,
    ):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.shift_height = shift_height

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if "rot_degree" in input_dict:
            noise_rotation = input_dict["rot_degree"]
        else:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
            input_dict["rot_degree"] = noise_rotation

        # calculate rotation matrix
        rot_sin = torch.sin(torch.tensor(noise_rotation))
        rot_cos = torch.cos(torch.tensor(noise_rotation))
        # align coord system with previous version
        rot_mat_T = torch.Tensor(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        input_dict["uni_rot_mat"] = rot_mat_T

        if len(input_dict["bbox3d_fields"]) == 0:  # test mode
            input_dict["bbox3d_fields"].append("empty_box3d")
            input_dict["empty_box3d"] = input_dict["box_type_3d"](
                np.array([], dtype=np.float32)
            )

        # rotate points with bboxes
        for key in input_dict["bbox3d_fields"]:
            if "points" in input_dict:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict["points"]
                )
                input_dict["points"] = points
                input_dict["pcd_rotation"] = rot_mat_T
            else:
                input_dict[key].rotate(noise_rotation)

        input_dict["transformation_3d_flow"].append("R")

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if "pcd_scale_factor" not in input_dict:
            scale_factor = np.random.uniform(
                self.scale_ratio_range[0], self.scale_ratio_range[1]
            )
            input_dict["pcd_scale_factor"] = scale_factor

        scale = input_dict["pcd_scale_factor"]
        if "points" in input_dict:
            points = input_dict["points"]
            points.scale(scale)
            if self.shift_height:
                assert (
                        "height" in points.attribute_dims.keys()
                ), "setting shift_height=True but points have no height attribute"
                points.tensor[:, points.attribute_dims["height"]] *= scale
            input_dict["points"] = points

        input_dict["uni_scale_mat"] = torch.Tensor(
            [[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]
        )

        for key in input_dict["bbox3d_fields"]:
            input_dict[key].scale(scale)

        input_dict["transformation_3d_flow"].append("S")

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if "pcd_trans" not in input_dict:
            translation_std = np.array(self.translation_std, dtype=np.float32)
            trans_factor = np.random.normal(scale=translation_std, size=3).T
            input_dict["pcd_trans"] = trans_factor
        else:
            trans_factor = input_dict["pcd_trans"]

        input_dict["uni_trans_mat"] = torch.Tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [trans_factor[0], trans_factor[1], trans_factor[2], 1],
            ]
        )

        if "points" in input_dict:
            input_dict["points"].translate(trans_factor)

        for key in input_dict["bbox3d_fields"]:
            input_dict[key].translate(trans_factor)

        input_dict["transformation_3d_flow"].append("T")

    def _flip_bbox_points(self, input_dict):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """

        def _flip_single(input_dict, direction="horizontal"):
            assert direction in ["horizontal", "vertical"]
            if len(input_dict["bbox3d_fields"]) == 0:  # test mode
                input_dict["bbox3d_fields"].append("empty_box3d")
                input_dict["empty_box3d"] = input_dict["box_type_3d"](
                    np.array([], dtype=np.float32)
                )
            assert len(input_dict["bbox3d_fields"]) == 1
            for key in input_dict["bbox3d_fields"]:
                if "points" in input_dict:
                    input_dict["points"] = input_dict[key].flip(
                        direction, points=input_dict["points"]
                    )
                else:
                    input_dict[key].flip(direction)

        if "pcd_horizontal_flip" not in input_dict:
            flip_horizontal = (
                True if np.random.rand() < self.flip_ratio_bev_horizontal else False
            )
            input_dict["pcd_horizontal_flip"] = flip_horizontal
        if "pcd_vertical_flip" not in input_dict:
            flip_vertical = (
                True if np.random.rand() < self.flip_ratio_bev_vertical else False
            )
            input_dict["pcd_vertical_flip"] = flip_vertical

        # flips the y (horizontal) or x (vertical) axis
        flip_mat = torch.Tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        if input_dict["pcd_horizontal_flip"]:
            _flip_single(input_dict, "horizontal")
            input_dict["transformation_3d_flow"].append("HF")
            flip_mat[1, 1] *= -1
        if input_dict["pcd_vertical_flip"]:
            _flip_single(input_dict, "vertical")
            input_dict["transformation_3d_flow"].append("VF")
            flip_mat[0, 0] *= -1

        input_dict["uni_flip_mat"] = flip_mat

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        self._rot_bbox_points(input_dict)

        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        self._flip_bbox_points(input_dict)

        # unified augmentation for point and voxel
        uni_rot_aug = (
                input_dict["uni_flip_mat"].T
                @ input_dict["uni_trans_mat"].T
                @ input_dict["uni_scale_mat"].T
                @ input_dict["uni_rot_mat"].T
        )
        input_dict["uni_rot_aug"] = uni_rot_aug
        if "lidar2img" in input_dict:
            input_dict["lidar2img"] = [
                input_dict["lidar2img"][_idx] @ np.linalg.inv(uni_rot_aug)
                for _idx in range(len(input_dict["lidar2img"]))
            ]
        if "lidar2cam" in input_dict:
            input_dict["lidar2cam"] = [
                input_dict["lidar2cam"][_idx] @ np.linalg.inv(uni_rot_aug)
                for _idx in range(len(input_dict["lidar2cam"]))
            ]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(rot_range={self.rot_range},"
        repr_str += f" scale_ratio_range={self.scale_ratio_range},"
        repr_str += f" translation_std={self.translation_std},"
        repr_str += f" flip_ratio_bev_horizontal={self.flip_ratio_bev_horizontal},"
        repr_str += f" flip_ratio_bev_vertical={self.flip_ratio_bev_vertical},"
        repr_str += f" shift_height={self.shift_height})"
        return repr_str


@PIPELINES.register_module()
class UnifiedObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(
            self, db_sampler, sample_2d=False, sample_method="depth", modify_points=False
    ):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        self.sample_method = sample_method
        self.modify_points = modify_points
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]

        # change to float for blending operation
        points = input_dict["points"]
        if self.sample_2d:
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=True
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, with_img=False
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_points_idx = sampled_dict["points_idx"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points_idx = -1 * np.ones(len(points), dtype=np.int)
            # check the points dimension
            # points = points.cat([sampled_points, points])
            points = points.cat([points, sampled_points])
            points_idx = np.concatenate([points_idx, sampled_points_idx], axis=0)

            if self.sample_2d:
                imgs = input_dict["img"]
                lidar2img = input_dict["lidar2img"]
                sampled_img = sampled_dict["images"]
                sampled_num = len(sampled_gt_bboxes_3d)
                imgs, points_keep = self.unified_sample(
                    imgs,
                    lidar2img,
                    points.tensor.numpy(),
                    points_idx,
                    gt_bboxes_3d.corners.numpy(),
                    sampled_img,
                    sampled_num,
                )

                input_dict["img"] = imgs

                if self.modify_points:
                    points = points[points_keep]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d.astype(np.long)
        input_dict["points"] = points

        return input_dict

    def unified_sample(
            self, imgs, lidar2img, points, points_idx, bboxes_3d, sampled_img, sampled_num
    ):
        # for boxes
        bboxes_3d = np.concatenate([bboxes_3d, np.ones_like(bboxes_3d[..., :1])], -1)
        is_raw = np.ones(len(bboxes_3d))
        is_raw[-sampled_num:] = 0
        is_raw = is_raw.astype(bool)
        raw_num = len(is_raw) - sampled_num
        # for point cloud
        points_3d = points[:, :4].copy()
        points_3d[:, -1] = 1
        points_keep = np.ones(len(points_3d)).astype(np.bool)
        new_imgs = imgs

        assert len(imgs) == len(lidar2img) and len(sampled_img) == sampled_num
        for _idx, (_img, _lidar2img) in enumerate(zip(imgs, lidar2img)):
            assert len(_lidar2img) == 1, "only support sweep == 1"
            _lidar2img = _lidar2img[0]  # (4, 4)
            coord_img = bboxes_3d @ _lidar2img.T
            coord_img[..., :2] /= coord_img[..., 2:3]
            depth = coord_img[..., 2]
            img_mask = (depth > 0).all(axis=-1)
            img_count = img_mask.nonzero()[0]
            if img_mask.sum() == 0:
                continue
            depth = depth.mean(1)[img_mask]
            coord_img = coord_img[..., :2][img_mask]
            minxy = np.min(coord_img, axis=-2)
            maxxy = np.max(coord_img, axis=-2)
            bbox = np.concatenate([minxy, maxxy], axis=-1).astype(int)
            bbox[:, 0::2] = np.clip(bbox[:, 0::2], a_min=0, a_max=_img.shape[1] - 1)
            bbox[:, 1::2] = np.clip(bbox[:, 1::2], a_min=0, a_max=_img.shape[0] - 1)
            img_mask = ((bbox[:, 2:] - bbox[:, :2]) > 1).all(axis=-1)
            if img_mask.sum() == 0:
                continue
            depth = depth[img_mask]
            if "depth" in self.sample_method:
                paste_order = depth.argsort()
                paste_order = paste_order[::-1]
            else:
                paste_order = np.arange(len(depth), dtype=np.int64)
            img_count = img_count[img_mask][paste_order]
            bbox = bbox[img_mask][paste_order]

            paste_mask = -255 * np.ones(_img.shape[:2], dtype=np.int)
            fg_mask = np.zeros(_img.shape[:2], dtype=np.int)
            # first crop image from raw image
            raw_img = []
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    raw_img.append(_img[_box[1]: _box[3], _box[0]: _box[2]])

            # then stitch the crops to raw image
            for _count, _box in zip(img_count, bbox):
                if is_raw[_count]:
                    _img[_box[1]: _box[3], _box[0]: _box[2]] = raw_img.pop(0)
                    fg_mask[_box[1]: _box[3], _box[0]: _box[2]] = 1
                else:
                    img_crop = sampled_img[_count - raw_num]
                    if len(img_crop) == 0:
                        continue
                    img_crop = cv2.resize(img_crop, tuple(_box[[2, 3]] - _box[[0, 1]]))
                    _img[_box[1]: _box[3], _box[0]: _box[2]] = img_crop

                paste_mask[_box[1]: _box[3], _box[0]: _box[2]] = _count

            new_imgs[_idx] = _img

            # calculate modify mask
            if self.modify_points:
                points_img = points_3d @ _lidar2img.T
                points_img[:, :2] /= points_img[:, 2:3]
                depth = points_img[:, 2]
                img_mask = depth > 0
                if img_mask.sum() == 0:
                    continue
                img_mask = (
                        (points_img[:, 0] > 0)
                        & (points_img[:, 0] < _img.shape[1])
                        & (points_img[:, 1] > 0)
                        & (points_img[:, 1] < _img.shape[0])
                        & img_mask
                )
                points_img = points_img[img_mask].astype(int)
                new_mask = paste_mask[points_img[:, 1], points_img[:, 0]] == (
                        points_idx[img_mask] + raw_num
                )
                raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_num)
                raw_bg = (fg_mask == 0) & (paste_mask < 0)
                raw_mask = (
                        raw_fg[points_img[:, 1], points_img[:, 0]]
                        | raw_bg[points_img[:, 1], points_img[:, 0]]
                )
                keep_mask = new_mask | raw_mask
                points_keep[img_mask] = points_keep[img_mask] & keep_mask

        return new_imgs, points_keep

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f" sample_2d={self.sample_2d},"
        repr_str += f" data_root={self.sampler_cfg.data_root},"
        repr_str += f" info_path={self.sampler_cfg.info_path},"
        repr_str += f" rate={self.sampler_cfg.rate},"
        repr_str += f" prepare={self.sampler_cfg.prepare},"
        repr_str += f" classes={self.sampler_cfg.classes},"
        repr_str += f" sample_groups={self.sampler_cfg.sample_groups}"
        return repr_str


@PIPELINES.register_module()
class NormalizeIntensity(object):
    """
    Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self, mean=[127.5], std=[127.5], use_dim=[3]):
        self.mean = mean
        self.std = std
        self.use_dim = use_dim

    def __call__(self, input_dict):
        points = input_dict["points"]
        # overwrite
        mean = points.tensor.new_tensor(self.mean)
        std = points.tensor.new_tensor(self.std)
        points.tensor[:, self.use_dim] = (points.tensor[:, self.use_dim] - mean) / std
        input_dict["points"] = points
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f" mean={self.mean},"
        repr_str += f" std={self.std},"
        repr_str += f" use_dim={self.use_dim}"
        return repr_str


@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'lidar2cam',
                            'depth2img', 'cam2img', 'pad_shape', 'pad_before_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus', 'ego2global_translation', 'ego2global_rotation',
                            'lidar2ego_translation', 'lidar2ego_rotation',
                            'lidar2global_rotation', 'resize_img', 'points', 'conv', 'gt_planning', 'gt_planning_mask',
                            'ego_hist_traj', 'ego_hist_mask', 'ego_command', 'ego_feat', 'location'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        data = {}
        img_metas = {}

        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            if key not in results:
                data[key] = None
            else:
                data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class LoadAnnoatationVQA():
    def __init__(
            self,
            align_desc_path=None,
            base_vqa_path=None,
            base_desc_path=None,
            base_conv_path=None,
            base_key_path=None,
            nusc_qa_path=None,
            future_prediction=False,
            traj_prediction=False,
            use_text_planning=False,
            img_length=2500,
            tokenizer=None,
            max_length=None,
            n_gen=2,
            ignore_type=["v1", "v2", "v3"],
            lane_objs_info=None,
            llm_path=None, ):
        self.n_gen = n_gen
        self.ignore_type = ignore_type
        self.align_desc_path = align_desc_path
        self.base_vqa_path = base_vqa_path
        self.base_desc_path = base_desc_path
        self.base_conv_path = base_conv_path
        self.base_key_path = base_key_path
        self.nusc_qa_path = nusc_qa_path
        self.future_prediction = future_prediction
        self.traj_prediction = traj_prediction
        self.use_text_planning = use_text_planning
        self.img_length = img_length
        CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                   'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                   'barrier')
        self.id2cat = {i: name for i, name in enumerate(CLASSES)}
        self.side = {
            'singapore': 'left',
            'boston': 'right',
        }
        self.template_desc = [
            "What can you tell about the current driving conditions from the images?",
            "What can be observed in the panoramic images provided?",
            "Can you provide a summary of the current driving scenario based on the input images?",
            "What can you observe from the provided images regarding the driving conditions?",
            "Please describe the current driving conditions based on the images provided.",
            "Can you describe the current weather conditions and the general environment depicted in the images?",
            "Please describe the current driving conditions based on the input images.",
            "Could you summarize the current driving conditions based on the input images?",
            "Please provide an overview of the current driving conditions based on the images.",
            "Can you summarize what the panoramic images show?",
            "Can you describe the overall conditions and environment based on the images?",
            "Could you describe the overall environment and objects captured in the images provided?",
            "What insights can be drawn from the current driving conditions in the images?",
            "What does the panoramic image reveal about the driving environment?",
            "How can the current road situation be summarized from the provided images?",
            "What do the images show regarding the driving conditions at present?",
        ]

        self.template_action = [
            "What action should be taken in the current driving scenario?",
            "What action should be taken based on the current driving conditions?",
            "What maneuver is needed in the current driving situation?",
            "What step should be taken based on the current traffic conditions?",
            "What decision should be made according to the present road conditions?",
            "Can you suggest a course of action based on the current driving conditions?",
            "Please provide a recommendation for the current driving scenario.",
            "Can you provide an appropriate response to the current driving scenario?",
            "What action should be taken in the current traffic situation?",
            "Please suggest an appropriate response for the current driving situation.",
            "Please advise on the best course of action based on the current driving scenario.",
            "Can you provide a suggested course of action for the present road conditions?",
            "What is the best course of action based on the current driving conditions?",
            "Could you propose a step to take in light of the current driving environment?",
            "Could you suggest what actions to take based on the prevailing traffic conditions?",
            "Could you provide a suggested step to take considering the present traffic conditions?"
        ]
        self.model_name = llm_path.split("/")[-1]
        if self.model_name == "InternVL2-1B":
            self.template_name = "Hermes-2"
            self.preprocess_func = preprocess_mpt
        elif self.model_name == "InternVL2-2B":
            self.template_name = "internlm2-chat"
            self.preprocess_func = preprocess_internlm
        elif self.model_name == "InternVL2-4B":
            self.template_name = "phi3-chat"
            self.preprocess_func = preprocess_phi3
        elif self.model_name == "InternVL2-8B":
            self.template_name = "internlm2-chat"
            self.preprocess_func = preprocess_internlm

        self.tokenizer_path = llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, add_eos_token=False,
                                                       trust_remote_code=True, use_fast=False)

        # self.tokenizer.pad_token = self.tokenizer.unk_token if self.model_name != "InternVL2-1B" else self.tokenizer.pad_token

        self.tokenizer.tokenizer_path = self.tokenizer_path
        self.tokenizer.model_max_length = 4096
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                      QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                      REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)

    def preprocess(self, conversations=None, img_length=2500):
        ret = self.preprocess_func(self.template_name, [deepcopy(conversations)],
                                   self.tokenizer, [img_length],
                                   group_by_length=True, ds_name='sth')

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
        )
        return ret

    def preprocess_vqa(self, results, traj):
        sources = []
        if self.align_desc_path is not None:
            # the MASKED_CAM is the camera that is masked in the image, "-1" means no camera is masked
            MASKED_CAM = results['MASKED_CAM']
            if os.path.exists(self.align_desc_path + results['sample_idx'] + '.json'):
                with open(self.align_desc_path + results['sample_idx'] + '.json', 'r') as f:
                    align_desc = json.load(f)["{}".format(MASKED_CAM)]["description"]
                question = random.sample(self.template_desc, 1)[0]
                sources.append(
                    [
                        {"from": 'human',
                         "value": question},
                        {"from": 'gpt',
                         "value": align_desc}
                    ]
                )
        else:
            # when use align_desc_path, other paths are not used
            if self.base_key_path is not None:
                if os.path.exists(self.base_key_path + results['sample_idx'] + ".json"):
                    with open(self.base_key_path + results['sample_idx'] + ".json", 'r') as f:
                        action = json.load(f)
                    sources.append(
                        [
                            {"from": 'human',
                             "value": "Please shortly describe your driving action."},
                            {"from": 'gpt',
                             "value": action}
                        ]
                    )
            if self.base_desc_path is not None:
                if os.path.exists(self.base_desc_path + results['sample_idx'] + ".json"):
                    with open(self.base_desc_path + results['sample_idx'] + ".json", 'r') as f:
                        desc = json.load(f)
                    question = random.sample(self.template_desc, 1)[0]
                    sources.append(
                        [
                            {"from": 'human',
                             "value": question},
                            {"from": 'gpt',
                             "value": desc["description"]}
                        ]
                    )
            if self.base_vqa_path is not None:
                if os.path.exists(self.base_vqa_path + results['sample_idx'] + ".json"):
                    with open(self.base_vqa_path + results['sample_idx'] + ".json", 'r') as f:
                        data_qa = json.load(f)
                    for i, pair in enumerate(data_qa):
                        sources.append(
                            [
                                {"from": 'human',
                                 "value": pair["question"]},
                                {"from": 'gpt',
                                 "value": pair["answer"]}
                            ]
                        )
            if self.base_conv_path is not None:
                if os.path.exists(
                        self.base_conv_path + results['sample_idx'] + ".json") and self.base_conv_path is not None:
                    with open(self.base_conv_path + results['sample_idx'] + ".json", 'r') as f:
                        data_qa = json.load(f)
                    for pair in data_qa:
                        sources.append(
                            [
                                {"from": 'human',
                                 "value": pair["question"]},
                                {"from": 'gpt',
                                 "value": pair["answer"]}
                            ]
                        )
            if self.nusc_qa_path is not None:
                if os.path.exists(self.nusc_qa_path + results['sample_idx'] + ".json"):
                    with open(self.nusc_qa_path + results['sample_idx'] + ".json", 'r') as f:
                        data_qa = json.load(f)["questions"]
                    for pair in data_qa:
                        sources.append(
                            [
                                {"from": 'human',
                                 "value": pair["question"] + "Use one or two words to answer the question."},
                                {"from": 'gpt',
                                 "value": pair["answer"]}
                            ]
                        )
        return sources

    def __call__(self, results):
        traj = None
        if 'location' in results.keys():
            results['location'] = "singapore" if results['location'] == 0 else "boston"

        prompt = f"You are driving in {results['location']}. " if 'location' in results.keys() else None

        if 'gt_planning' in results.keys() and self.use_text_planning:
            planning_traj = results['gt_planning'][0, :, :2]
            mask = results['gt_planning_mask'][0].any(axis=1)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                formatted_points = ', '.join(
                    f"({format_number(point[0], 2)}, {format_number(point[1], 2)})" for point in planning_traj)
                traj = f"Here is the planning trajectory [PT, {formatted_points}]."

        conv = self.preprocess_vqa(results, None)
        random.shuffle(conv)
        if self.future_prediction and self.align_desc_path is None:
            if os.path.exists(self.base_desc_path + results['sample_idx'] + ".json"):
                with open(self.base_desc_path + results['sample_idx'] + ".json", 'r') as f:
                    desc = json.load(f)
                question = random.sample(self.template_action, 1)[0]
                conv = conv + [
                    [{"from": 'human',
                      "value": question},
                     {"from": 'gpt',
                      "value": desc["action"]}]
                ]
            if self.use_text_planning:
                if 'gt_planning' in results.keys() and len(planning_traj) == 6:
                    conv = conv + [
                        [{"from": 'human',
                          "value": "Please provide the planning trajectory for the ego car without reasons."},
                         {"from": 'gpt',
                          "value": traj}]
                    ]
            if self.traj_prediction and 'location' in results.keys():
                conv = conv + [
                    [{"from": 'human',
                      "value": "You are driving in {}, please predict the future trajectory of the ego car, and imagine the future scene.".format(
                          results['location'])},
                     {"from": 'gpt',
                      "value": "Sure, I can try to predict the future trajectory and envision the future scene based on the understanding of the driving scenario."}]
                ]
            else:
                conv = conv + [
                    [{"from": 'human',
                      "value": "Please try to imagine the future scene."},
                     {"from": 'gpt',
                      "value": "Sure, I can try to envision the future scene based on the understanding of the driving scenario."}]
                ]
        vqa_anno = [item for pair in conv for item in pair]
        if len(vqa_anno) == 0:
            vqa_anno = [
                {'from': 'human',
                 'value': '<image>\n'},
                {'from': 'gpt',
                 'value': ''}
            ]
        else:
            vqa_anno[0]['value'] = "<image>\n" + vqa_anno[0]['value']

        processed_text = self.preprocess(vqa_anno, img_length=self.img_length)

        input_ids = processed_text['input_ids'].unsqueeze(0)
        attention_mask = processed_text['attention_mask'].unsqueeze(0)
        labels = processed_text['labels'].unsqueeze(0)

        results['conv'] = {'input_ids': input_ids,
                           'attention_mask': attention_mask,
                           'labels': labels}

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
