import torch

from . import voxel_pool_ext

class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coords, ranks, B, X, Y, Z):
        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]
        coords = coords.int()

        out = voxel_pool_ext.voxel_pool_forward(feats, coords, interval_lengths, interval_starts, B, X, Y, Z)

        ctx.save_for_backward(interval_starts, interval_lengths, coords)
        ctx.saved_shapes = B, X, Y, Z
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, coords = ctx.saved_tensors
        B, X, Y, Z = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        feats_grad = voxel_pool_ext.voxel_pool_backward(out_grad, coords, interval_lengths, interval_starts, B, X, Y, Z)

        return feats_grad, None, None, None, None, None, None


def voxel_pool(feats, coords, B, X, Y, Z):
    # coords: [bs_idx, x, y, z]
    assert feats.shape[0] == coords.shape[0]

    ranks = coords[:, 0] * X * Y * Z + coords[:, 1] * Y * Z + coords[:, 2] * Z + coords[:, 3]
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, X, Y, Z)

    return x
