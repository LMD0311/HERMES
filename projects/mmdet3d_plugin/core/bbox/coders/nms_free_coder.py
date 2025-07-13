import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from mmdet3d.core.bbox.structures import get_box_type
from mmdet3d.core import xywhr2xyxyr
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(
        self,
        pc_range,
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        num_classes=10,
    ):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i])
            )
        return predictions_list


@BBOX_CODERS.register_module()
class NMSFreeIoUCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(
        self,
        pc_range,
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        nms_cfg=None,
        num_classes=10,
    ):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.nms_cfg = nms_cfg
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, iou_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        iou_preds = torch.clamp((iou_preds.squeeze(-1) + 1) * 0.5, min=0.0, max=1.0)

        final_box_preds = denormalize_bbox(bbox_preds[bbox_index], self.pc_range)
        final_scores = torch.pow(scores, 0.5) * torch.pow(iou_preds[bbox_index], 0.5)
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

            final_box_preds, final_scores, final_preds = (
                final_box_preds[thresh_mask],
                final_scores[thresh_mask],
                final_preds[thresh_mask],
            )

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            final_box_preds, final_scores, final_preds = (
                final_box_preds[mask],
                final_scores[mask],
                final_preds[mask],
            )
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )

        if self.nms_cfg is not None:
            box_type, _ = get_box_type("lidar")
            boxes_for_nms = xywhr2xyxyr(
                box_type(final_box_preds, final_box_preds.shape[-1]).bev
            )
            # selected = nms_gpu(
            #     boxes_for_nms,
            #     final_scores,
            #     thresh=self.nms_threshold
            # )
            # final_box_preds, final_scores, final_preds = \
            #     final_box_preds[selected], final_scores[selected], final_preds[selected]

            keep_mask = torch.zeros_like(final_scores)
            for task_nms_cfg in self.nms_cfg:
                task_mask = torch.zeros_like(final_scores)
                for cls_idx in task_nms_cfg["indices"]:
                    task_mask += final_preds == cls_idx
                task_mask = task_mask.bool()
                if task_mask.sum() > 0 and task_nms_cfg["nms_threshold"] > 0:
                    task_selected = nms_gpu(
                        boxes_for_nms[task_mask],
                        final_scores[task_mask],
                        thresh=task_nms_cfg["nms_threshold"],
                    )
                else:
                    task_selected = torch.arange(task_mask.sum())

                if task_selected.shape[0] != 0:
                    keep_indices = torch.nonzero(task_mask, as_tuple=True)[0][
                        task_selected
                    ]
                    keep_mask[keep_indices] = 1
            keep_mask = keep_mask.bool()
            final_box_preds, final_scores, final_preds = (
                final_box_preds[keep_mask],
                final_scores[keep_mask],
                final_preds[keep_mask],
            )

        predictions_dict = {
            "bboxes": final_box_preds,
            "scores": final_scores,
            "labels": final_preds,
        }
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_iou_preds = preds_dicts["all_iou_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_iou_preds[i]
                )
            )
        return predictions_list
