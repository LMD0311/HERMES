from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.core.merge_all_augs import merge_all_aug_bboxes_3d
import torch.distributed as dist
from ..utils.uni3d_voxelpooldepth import DepthNet


@DETECTORS.register_module()
class UVTR(MVXTwoStageDetector):
    """UVTR."""

    def __init__(
        self,
        use_grid_mask=False,
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
        pretrained_img=None,
        pretrained_pts=None,
        load_img=None,
        load_pts=None,
    ):
        super(UVTR, self).__init__(
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
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
            in_channels = self.img_neck.out_channels
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
            self.use_grid_mask = use_grid_mask
        self.pretrained_img = pretrained_img
        self.pretrained_pts = pretrained_pts
        self.load_img = load_img
        self.load_pts = load_pts

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )

    def _load_state_dict(self, model_state_disk, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def init_weights(self):
        """Initialize weights of the depth head."""
        super().init_weights()
        if not self.with_img_backbone:
            return

        valid_ckpt_load = {}
        # load pretrained pts model
        if self.pretrained_pts is not None:
            ckpt_load = torch.load(
                self.pretrained_pts,
                map_location="cuda:{}".format(torch.cuda.current_device()),
            )["state_dict"]
            print("Loaded pretrained model from: {}".format(self.pretrained_pts))
            for load_key in self.load_pts:
                dict_load = {
                    _key.replace(load_key + ".", ""): ckpt_load[_key]
                    for _key in ckpt_load
                    if load_key in _key
                }
                getattr(self, load_key).load_state_dict(dict_load, strict=False)
                print("Loaded pretrained {}".format(load_key))
                assert len(dict_load) > 0

        # load pretrained img model
        if self.pretrained_img is not None:
            ckpt_load = torch.load(
                self.pretrained_img,
                map_location="cuda:{}".format(torch.cuda.current_device()),
            )["state_dict"]
            if dist.get_rank() == 0:
                print(
                    "Loaded pretrained img model from: {}".format(self.pretrained_img)
                )

            for load_key in self.load_img:
                if "img" not in load_key:
                    continue
                valid_ckpt_load.update(
                    {_key: ckpt_load[_key] for _key in ckpt_load if load_key in _key}
                )

            if "input_proj" in self.load_img:
                valid_ckpt_load.update(
                    {
                        _key: ckpt_load[_key]
                        for _key in ckpt_load
                        if "input_proj" in _key
                    }
                )

            if "depth_head" in self.load_img:
                valid_ckpt_load.update(
                    {_key: ckpt_load[_key] for _key in ckpt_load if "depth_net" in _key}
                )

            if "view_trans" in self.load_img:
                dict_load = {
                    _key.replace("pts_bbox_head.view_trans.", ""): ckpt_load[_key]
                    for _key in ckpt_load
                    if "pts_bbox_head.view_trans" in _key
                }
                self.pts_bbox_head.view_trans.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained view_trans")
                assert len(dict_load) > 0

        if len(valid_ckpt_load) > 0:
            state_dict, update_model_state = self._load_state_dict(
                valid_ckpt_load, strict=False
            )
            if dist.get_rank() == 0:
                for key in state_dict:
                    if key not in update_model_state:
                        print(
                            "Not updated weight %s: %s"
                            % (key, str(state_dict[key].shape))
                        )
                print(
                    "==> Done (loaded %d/%d)"
                    % (len(update_model_state), len(state_dict))
                )

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

            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
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

    def forward_pts_train(
        self,
        pts_feats,
        img_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        depth_prob,
        depth_coord,
        img_metas,
        img_depth=None,
        gt_bboxes_ignore=None,
        points=None,
        img=None,
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
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, img_depth)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, depth_prob, depth_coord, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        if self.with_depth_head and hasattr(self.pts_bbox_head.view_trans, "loss"):
            losses.update(
                self.pts_bbox_head.view_trans.loss(img_depth, points, img, img_metas)
            )
        return losses

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

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        depth_prob=None,
        depth_coord=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
    ):
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
        pts_feat, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )

        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feat,
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            depth_prob,
            depth_coord,
            img_metas,
            img_depth,
            gt_bboxes_ignore,
            points,
            img,
        )
        losses.update(losses_pts)

        return losses

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    "num of augmentations ({}) != num of image meta ({})".format(
                        len(points), len(img_metas)
                    )
                )

        if num_augs == 1:
            if not isinstance(img_metas[0], list):
                img_metas = [img_metas]
            if not isinstance(img, list):
                img = [img]
            results = self.simple_test(img_metas[0], points, img[0], **kwargs)
        else:
            results = self.aug_test(points, img_metas, img, **kwargs)

        return results

    def simple_test_pts(self, pts_feat, img_feats, img_metas, img_depth, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(pts_feat, img_feats, img_metas, img_depth)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, points=None, img=None, rescale=False):
        """Test function without augmentaiton."""
        pts_feat, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feat, img_feats, img_metas, img_depth, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        pts_feats, img_feats, img_depths = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(
                pts_feats, img_feats, img_depths, img_metas, rescale
            )
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

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

    def aug_test_pts(self, pts_feats, img_feats, img_depths, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for _idx, img_meta in enumerate(img_metas):
            outs = self.pts_bbox_head(
                pts_feats[_idx], img_feats[_idx], img_meta, img_depths[_idx]
            )
            bbox_list = self.pts_bbox_head.get_bboxes(outs, img_meta, rescale=rescale)

            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_all_aug_bboxes_3d(
            aug_bboxes, img_metas, self.pts_bbox_head.test_cfg
        )
        return merged_bboxes
