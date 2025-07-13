# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from mmdet3d.core.bbox.structures import get_box_type


@IOU_CALCULATORS.register_module()
class PairedBboxOverlaps3D(object):
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate):
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        """Calculate 3D IoU using cuda implementation.

        Note:
            This function calculate the IoU of 3D boxes based on their volumes.
            IoU calculator ``:class:BboxOverlaps3D`` uses this function to
            calculate the actual 3D IoUs of boxes.

        Args:
            bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry].
            bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry].
            mode (str): "iou" (intersection over union) or
                iof (intersection over foreground).

        Return:
            torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2 \
                with shape (M, N) (aligned mode is not supported currently).
        """
        return paired_bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        """str: return a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str


def paired_bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='camera'):
    """Calculate 3D IoU using cuda implementation.

    Note:
        This function calculates the IoU of 3D boxes based on their volumes.
        IoU calculator :class:`BboxOverlaps3D` uses this function to
        calculate the actual IoUs of boxes.

    Args:
        bboxes1 (torch.Tensor): shape (N, 7+C) [x, y, z, h, w, l, ry].
        bboxes2 (torch.Tensor): shape (M, 7+C) [x, y, z, h, w, l, ry].
        mode (str): "iou" (intersection over union) or
            iof (intersection over foreground).
        coordinate (str): 'camera' or 'lidar' coordinate system.

    Return:
        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2 \
            with shape (M, N) (aligned mode is not supported currently).
    """
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7 and bboxes1.size(0) == bboxes2.size(0)

    box_type, _ = get_box_type(coordinate)

    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode, paired=True)
