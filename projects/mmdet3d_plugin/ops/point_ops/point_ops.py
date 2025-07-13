import torch
from . import point_ops_ext


def group_inner_inds(points, inverse_inds, K):
    """
    Args:
        points: (N, C)
        inverse_inds: (N, )
    Return:
        group_points: (valid_voxel_num, K, C)
    """
    valid_voxel_num = inverse_inds.max().item() + 1
    group_inds = torch.full((valid_voxel_num, K), -1, dtype=torch.long, device=points.device)
    point_ops_ext.group_inner_inds_wrapper(inverse_inds.contiguous(), group_inds)
    group_points = points[group_inds]
    return group_points