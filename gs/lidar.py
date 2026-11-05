"""LiDAR / laser-scan depth supervision for Gaussian Splatting.

The reference cloud is projected into every training camera to produce a *sparse* depth map
(0 where no point lands). Where several points fall in one pixel the nearest wins (z-buffer),
which removes most of the see-through problem of projecting an unoccluded point cloud; what it
cannot remove is a point behind a *thin* surface that no reference point represents (fences,
foliage). The loss therefore uses a robust L1 and only pixels the model already renders as
opaque, and the remaining wrong supervision is listed in the known-distortions of the metric card.

Depth loss:  L_d = mean_{p in M} | D_pred(p) - D_lidar(p) |,   M = {D_lidar > 0, acc > acc_min}
Total loss:  L = (1 - lambda) L1 + lambda (1 - SSIM) + lambda_d * L_d / depth_scale
`depth_scale` (median LiDAR depth of the scene) makes lambda_d unit-free so the same value
works for a 5 m courtyard and a 300 m aerial block.
"""
import numpy as np
import torch


def project_points_to_depth(xyz, cam, splat=1, z_near=0.05, max_points_chunk=4_000_000):
    """xyz (N,3) numpy or tensor, world frame -> depth (H,W) tensor, 0 where empty.
    splat>1 writes each point into a (splat x splat) pixel block (nearest-wins), useful when the
    cloud is much sparser than the image."""
    dev = cam.w2c.device
    xyz = torch.as_tensor(np.asarray(xyz) if not torch.is_tensor(xyz) else xyz, dtype=torch.float32, device=dev)
    H, W = cam.H, cam.W
    K, w2c = cam.K, cam.w2c
    depth = torch.full((H * W,), float("inf"), device=dev)
    for i in range(0, len(xyz), max_points_chunk):
        p = xyz[i:i + max_points_chunk] @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        m = z > z_near
        p, z = p[m], z[m]
        u = (K[0, 0] * p[:, 0] / z + K[0, 2]).floor().long()
        v = (K[1, 1] * p[:, 1] / z + K[1, 2]).floor().long()
        for du in range(splat):
            for dv in range(splat):
                uu, vv = u + du - splat // 2, v + dv - splat // 2
                ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                idx = vv[ok] * W + uu[ok]
                depth.scatter_reduce_(0, idx, z[ok], reduce="amin")
    depth[torch.isinf(depth)] = 0
    return depth.reshape(H, W)


def attach_lidar_depth(cams, xyz, splat=1, min_pixels=50, verbose=True):
    """Fill cam.depth for every camera; returns the median LiDAR depth (used as depth_scale)."""
    meds = []
    for c in cams:
        d = project_points_to_depth(xyz, c, splat=splat)
        if int((d > 0).sum()) >= min_pixels:
            c.depth = d
            meds.append(float(d[d > 0].median()))
        else:
            c.depth = None
    if verbose:
        n_ok = sum(c.depth is not None for c in cams)
        cover = np.mean([float((c.depth > 0).float().mean()) for c in cams if c.depth is not None]) if n_ok else 0
        print(f"LiDAR depth attached to {n_ok}/{len(cams)} cameras, mean pixel coverage {cover:.3%}")
    return float(np.median(meds)) if meds else 1.0


def depth_loss(pred_depth, acc, lidar_depth, acc_min=0.5, depth_scale=1.0):
    """Robust L1 between rendered expected depth and sparse LiDAR depth on supervised pixels.
    Returns (loss, n_pixels). loss is 0 (with grad) when nothing is supervised."""
    m = (lidar_depth > 0) & (acc.detach() >= acc_min)
    n = int(m.sum())
    if n == 0:
        return pred_depth.sum() * 0.0, 0
    return (pred_depth[m] - lidar_depth[m]).abs().mean() / depth_scale, n


def depth_error_stats(pred_depth, acc, lidar_depth, acc_min=0.5):
    """Per-view diagnostics (no grad): mean |err|, median |err|, relative error, coverage."""
    with torch.no_grad():
        m = (lidar_depth > 0) & (acc >= acc_min)
        if int(m.sum()) == 0:
            return dict(n=0)
        e = (pred_depth[m] - lidar_depth[m]).abs()
        return dict(n=int(m.sum()), abs_mean=float(e.mean()), abs_median=float(e.median()),
                    rel_mean=float((e / lidar_depth[m]).mean()),
                    supervised_frac=float(m.float().mean()), lidar_frac=float((lidar_depth > 0).float().mean()))
