"""Geometric evaluation of a trained Gaussian scene against a laser / LiDAR reference.

Pipeline (see run_geometry.py):
    rendered expected depth per training view  -->  unproject to 3-D points  -->  fuse + voxel-downsample
    -->  Sim(3)-align to the reference cloud (if the frames differ)  -->  metrics + per-cell error map

Metrics follow the ETH3D / Tanks-and-Temples convention:
    accuracy(tau)     = fraction of reconstructed points within tau of the reference   (precision)
    completeness(tau) = fraction of reference points within tau of the reconstruction  (recall)
    F-score(tau)      = harmonic mean of the two
plus the mean / median / RMSE of the nearest-neighbour distance in both directions (Chamfer terms)
and, when reference normals are available, the point-to-plane distance of reconstructed points.
Everything is in the units of the reference cloud (metres for ETH3D and LiDAR surveys).
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- alignment
def umeyama_sim3(src, dst, with_scale=True):
    """Least-squares similarity transform mapping src (N,3) onto dst (N,3) with correspondences.
    Returns s, R (3,3), t (3,) such that  dst ~ s * R @ src + t.
    With with_scale=False this is the SE(3) Umeyama used in the SLAM repo."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    H = xs.T @ xd / len(src)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_s = (xs ** 2).sum() / len(src)
    s = float((D * np.diag(S)).sum() / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(pts, s, R, t):
    return (s * (np.asarray(pts) @ R.T)) + t


def icp_sim3(src, dst, iters=20, max_dist=None, with_scale=True, seed=0, n_sample=200_000):
    """Point-to-point ICP on top of umeyama_sim3 for the case where the two clouds are already
    roughly aligned (e.g. COLMAP frame vs. laser frame after a manual or GPS initialisation).
    Returns (s, R, t) and the final inlier RMSE. Not meant for wide-baseline registration."""
    rng = np.random.default_rng(seed)
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if len(src) > n_sample:
        src = src[rng.choice(len(src), n_sample, replace=False)]
    tree = cKDTree(dst)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    cur = src.copy()
    rmse = np.inf
    for _ in range(iters):
        d, idx = tree.query(cur, k=1)
        thr = max_dist if max_dist is not None else np.percentile(d, 80)
        m = d < thr
        if m.sum() < 10:
            break
        s_i, R_i, t_i = umeyama_sim3(cur[m], dst[idx[m]], with_scale)
        cur = apply_sim3(cur, s_i, R_i, t_i)
        s, R, t = s_i * s, R_i @ R, s_i * (R_i @ t) + t_i
        rmse = float(np.sqrt(np.mean(d[m] ** 2)))
    return (s, R, t), rmse


# ---------------------------------------------------------------- depth -> points
def unproject_depth(depth, cam, acc=None, acc_min=0.5, stride=1, depth_max=None):
    """depth (H,W) tensor of camera-frame z; returns world points (M,3) tensor.
    Pixels with acc < acc_min (background / sky) or depth > depth_max are dropped."""
    H, W = depth.shape
    ys, xs = torch.meshgrid(torch.arange(0, H, stride, device=depth.device),
                            torch.arange(0, W, stride, device=depth.device), indexing="ij")
    z = depth[ys, xs]
    m = z > 0
    if acc is not None:
        m &= acc[ys, xs] >= acc_min
    if depth_max is not None:
        m &= z <= depth_max
    K, w2c = cam.K, cam.w2c
    x = (xs.float() + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys.float() + 0.5 - K[1, 2]) / K[1, 1] * z
    p_cam = torch.stack([x, y, z], -1)[m]
    c2w = torch.linalg.inv(w2c)
    return p_cam @ c2w[:3, :3].T + c2w[:3, 3]


def voxel_downsample(pts, voxel):
    """Keep one point per voxel (the first seen). pts (M,3) numpy -> (M',3)."""
    pts = np.asarray(pts)
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


# ---------------------------------------------------------------- metrics
def nn_distances(a, b, workers=-1):
    """For every point of a, distance to its nearest neighbour in b."""
    return cKDTree(np.asarray(b)).query(np.asarray(a), k=1, workers=workers)[0]


def geometry_metrics(rec, ref, taus=(0.02, 0.05, 0.1, 0.2), ref_normals=None, max_dist=None):
    """rec, ref: (N,3) numpy in the same frame. Returns (summary dict, d_rec, d_ref, p2pl).
    d_rec: distance of every rec point to the reference; d_ref: distance of every ref point to
    the reconstruction; p2pl: point-to-plane distance of every rec point (None without normals).
    Use summarize_distances() to re-aggregate these over subsets (e.g. held-out blocks) so that a
    point near a subset border still finds its true nearest neighbour."""
    tree_ref = cKDTree(np.asarray(ref))
    d_rec, idx = tree_ref.query(np.asarray(rec), k=1, workers=-1)   # accuracy side (index reused below)
    d_ref = nn_distances(ref, rec)                                  # completeness side
    p2pl = None
    if ref_normals is not None:
        p2pl = np.abs(((rec - ref[idx]) * ref_normals[idx]).sum(1))
    return summarize_distances(d_rec, d_ref, taus, p2pl, max_dist), d_rec, d_ref, p2pl


def summarize_distances(d_rec, d_ref, taus=(0.02, 0.05, 0.1, 0.2), p2pl=None, max_dist=None):
    """max_dist clips the distances used for the mean/RMSE so a few far outliers (sky splats
    left after masking) do not dominate; the clipped fraction is reported separately."""
    out = dict(n_rec=int(len(d_rec)), n_ref=int(len(d_ref)))
    for tau in taus:
        acc = float((d_rec < tau).mean()); com = float((d_ref < tau).mean())
        out[f"accuracy@{tau}"] = acc; out[f"completeness@{tau}"] = com
        out[f"fscore@{tau}"] = 0.0 if acc + com == 0 else 2 * acc * com / (acc + com)
    if max_dist is not None:
        out["frac_rec_beyond_max"] = float((d_rec > max_dist).mean())
        d_rec_c, d_ref_c = np.minimum(d_rec, max_dist), np.minimum(d_ref, max_dist)
    else:
        d_rec_c, d_ref_c = d_rec, d_ref
    out.update(acc_mean=float(d_rec_c.mean()), acc_median=float(np.median(d_rec)), acc_rmse=float(np.sqrt((d_rec_c ** 2).mean())),
               comp_mean=float(d_ref_c.mean()), comp_median=float(np.median(d_ref)),
               chamfer=float(d_rec_c.mean() + d_ref_c.mean()))
    if p2pl is not None:
        p = np.minimum(p2pl, max_dist) if max_dist is not None else p2pl
        out.update(p2plane_mean=float(p.mean()), p2plane_median=float(np.median(p2pl)), p2plane_rmse=float(np.sqrt((p ** 2).mean())))
    return out


def estimate_normals(pts, radius=None, knn=30):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, np.float64)))
    if radius:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn))
    else:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pc.normals)


# ---------------------------------------------------------------- spatial block hold-out
def block_ids(xyz, block):
    """Integer id of the (block x block x block) cube each point falls in (spatial hash, no modulo)."""
    k = np.floor(np.asarray(xyz) / block).astype(np.int64)
    return (k[:, 0] * 73856093) ^ (k[:, 1] * 19349663) ^ (k[:, 2] * 83492791)


def hidden_blocks(split_ref, block, frac, seed):
    """Block lottery of the hold-out split, defined on ONE cloud (the reference the supervision was drawn from).
    Returns (known, hidden): the sorted distinct block ids of split_ref and the subset hidden from supervision.
    Each distinct block is hidden with probability frac; the draw is made in sorted-id order with default_rng(seed),
    so the result depends on which blocks split_ref occupies. That is why every other cloud must be assigned by
    membership in these two sets, never by re-running the lottery on its own blocks (see blockwise_metrics)."""
    known = np.unique(block_ids(split_ref, block))
    if frac <= 0:
        return known, known[:0]
    rng = np.random.default_rng(seed)
    return known, known[rng.random(len(known)) < frac]


def holdout_mask(xyz, block, frac, seed):
    """True for points of xyz in blocks hidden from supervision, with the lottery run on xyz itself. Use this only
    on the cloud that defines the split (the training reference); for any other cloud use in_blocks() with the
    sets from hidden_blocks(reference, ...)."""
    if frac <= 0:
        return np.zeros(len(xyz), bool)
    _, hidden = hidden_blocks(xyz, block, frac, seed)
    return np.isin(block_ids(xyz, block), hidden)


def in_blocks(xyz, block, ids):
    return np.isin(block_ids(xyz, block), ids)


def blockwise_metrics(rec, ref, d_rec, d_ref, block, frac, seed, taus=(0.02, 0.05, 0.1, 0.2), p2pl=None,
                      max_dist=None, min_points=100, split_ref=None):
    """Split already-computed per-point distances into supervised / held-out blocks.

    The split is defined on split_ref (default: ref), the cloud the training supervision was drawn from, and both
    rec and ref are assigned to it by block membership. An earlier version ran the block lottery separately on rec
    and on ref; because the two clouds never occupy exactly the same blocks (floaters, holes), the random draws fell
    out of step and rec's "held-out" points were an arbitrary half of the reconstruction, not the hidden blocks.
    Reconstructed points in blocks split_ref does not occupy belong to neither subset (reported as
    rec_frac_unassigned; they are still in the all-points metrics).

    d_rec and d_ref MUST come from nearest-neighbour queries against the FULL other cloud
    (geometry_metrics). Re-running the NN search inside each subset would let a point near a block
    border miss its true neighbour in the adjacent block and inflate both subsets' errors.
    Returns {"supervised_blocks": {...}, "heldout_blocks": {...}, "split": {...}}
    (a subset is omitted when it has fewer than min_points on either side)."""
    known, hidden = hidden_blocks(ref if split_ref is None else split_ref, block, frac, seed)
    rec_ids, ref_ids = block_ids(rec, block), block_ids(ref, block)
    rec_hidden, ref_hidden = np.isin(rec_ids, hidden), np.isin(ref_ids, hidden)
    rec_sup, ref_sup = np.isin(rec_ids, known) & ~rec_hidden, np.isin(ref_ids, known) & ~ref_hidden
    out = {"split": dict(n_blocks=int(len(known)), n_hidden=int(len(hidden)),
                         rec_frac_unassigned=float(1 - (rec_hidden | rec_sup).mean()),
                         ref_frac_unassigned=float(1 - (ref_hidden | ref_sup).mean()))}
    for name, mr, mg in (("supervised_blocks", rec_sup, ref_sup), ("heldout_blocks", rec_hidden, ref_hidden)):
        if mr.sum() >= min_points and mg.sum() >= min_points:
            out[name] = summarize_distances(d_rec[mr], d_ref[mg], taus, p2pl[mr] if p2pl is not None else None, max_dist)
    return out


# ---------------------------------------------------------------- per-cell acceptance map
def error_grid(pts, err, cell, axes=(0, 1), min_count=20):
    """Top-down grid: for each (cell x cell) footprint in the two `axes`, mean error, 90th
    percentile and count. Returns dict with 'origin', 'cell', 'shape' and three 2-D arrays
    (NaN where fewer than min_count points). This is the raw material of the acceptance table:
    a cell whose p90 exceeds the project tolerance is marked 'not to be trusted'."""
    pts = np.asarray(pts); err = np.asarray(err)
    xy = pts[:, list(axes)]
    origin = xy.min(0)
    ij = np.floor((xy - origin) / cell).astype(np.int64)
    shape = ij.max(0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    n = np.bincount(flat, minlength=shape[0] * shape[1]).astype(float)
    s = np.bincount(flat, weights=err, minlength=shape[0] * shape[1])
    mean = np.full(n.shape, np.nan); mean[n >= min_count] = s[n >= min_count] / n[n >= min_count]
    # p90 per cell via sorting
    order = np.argsort(flat, kind="stable")
    p90 = np.full(n.shape, np.nan)
    bounds = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=shape[0] * shape[1]))])
    e_sorted = err[order]
    for c in np.nonzero(n >= min_count)[0]:
        p90[c] = np.percentile(e_sorted[bounds[c]:bounds[c + 1]], 90)
    return dict(origin=origin, cell=cell, shape=tuple(int(v) for v in shape), count=n.reshape(shape),
                mean=mean.reshape(shape), p90=p90.reshape(shape))


def acceptance_table(grid, tolerance):
    """Summarise an error_grid against a tolerance: how many cells pass, fail, or lack data."""
    p90 = grid["p90"]
    valid = ~np.isnan(p90)
    return dict(tolerance=tolerance, cells_total=int(valid.sum()), cells_pass=int((p90[valid] <= tolerance).sum()),
                cells_fail=int((p90[valid] > tolerance).sum()), cells_no_data=int((~valid).sum()),
                pass_fraction=float((p90[valid] <= tolerance).mean()) if valid.any() else float("nan"))
