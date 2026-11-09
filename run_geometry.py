#!/usr/bin/env python3
"""Train 3DGS on a scene with a laser / LiDAR reference, optionally with LiDAR depth supervision,
then evaluate the *geometry* of the result against the reference and produce the per-cell
acceptance map. This is the experiment behind the "can this splat be measured?" question.

    # baseline (photometric only)
    python run_geometry.py --data data/eth3d/courtyard --out runs/courtyard_base
    # LiDAR depth supervision, same everything else
    python run_geometry.py --data data/eth3d/courtyard --out runs/courtyard_d05 --lambda_depth 0.5
    # evaluate an existing model only
    python run_geometry.py --data data/eth3d/courtyard --out runs/courtyard_base --eval_only
    # CPU smoke run (reference rasteriser)
    python run_geometry.py --data data/eth3d/courtyard --out runs/smoke --device cpu --downscale 8 --iters 200 --max_train 6 --max_test 2

Honesty note on supervision vs. reference. The same scan is the only metric truth available, so
the LiDAR used for supervision is a *subsample* of it (--lidar_voxel) and, by default, half of the
scene's 1 m blocks are hidden from supervision (--lidar_holdout 0.5). Metrics are then reported
for all / supervised / held-out blocks separately: improvement on held-out blocks is improvement
the supervision did not simply copy in.
"""
import argparse, json, os, time
import numpy as np
import torch
from gs.data import load_colmap_scene, load_init_points, Camera
from gs.trainer import Config, train, evaluate
from gs.render import render
from gs.geometry import (unproject_depth, voxel_downsample, geometry_metrics, summarize_distances, estimate_normals, error_grid,
                         acceptance_table, icp_sim3, apply_sim3, block_ids, holdout_mask, blockwise_metrics)
from gs.lidar import attach_lidar_depth, project_points_to_depth, depth_error_stats


@torch.no_grad()
def fuse_points(model, cams, backend, acc_min, stride, depth_max, voxel, bg):
    pts = []
    for c in cams:
        _, info = render(model, c, backend=backend, bg=bg, return_depth=True)
        pts.append(unproject_depth(info["depth"], c, info["acc"], acc_min=acc_min, stride=stride, depth_max=depth_max).cpu().numpy())
    pts = np.concatenate(pts)
    n_raw = len(pts)
    pts = voxel_downsample(pts, voxel)
    return pts, n_raw


@torch.no_grad()
def view_depth_errors(model, cams, gt_xyz, backend, acc_min, bg, splat):
    """Rendered depth vs. projected reference depth on (test) views: a second, image-space geometry metric."""
    rows = []
    for c in cams:
        d_ref = project_points_to_depth(gt_xyz, c, splat=splat)
        _, info = render(model, c, backend=backend, bg=bg, return_depth=True)
        rows.append(depth_error_stats(info["depth"], info["acc"], d_ref, acc_min))
    rows = [r for r in rows if r.get("n", 0) > 0]
    if not rows:
        return {}
    return dict(abs_mean=float(np.mean([r["abs_mean"] for r in rows])), abs_median=float(np.median([r["abs_median"] for r in rows])),
                rel_mean=float(np.mean([r["rel_mean"] for r in rows])), n_views=len(rows))


def save_heatmap(grid, path, title, vmax=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key in zip(axes, ("mean", "p90")):
        im = ax.imshow(grid[key].T, origin="lower", cmap="magma", vmax=vmax)
        ax.set_title(f"{title}: {key} error per {grid['cell']} m cell [m]"); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000); ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--max_train", type=int, default=None); ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--grad_thresh", type=float, default=0.0002)
    ap.add_argument("--densify_until", type=int, default=15000); ap.add_argument("--sh_max", type=int, default=3)
    # LiDAR supervision
    ap.add_argument("--lambda_depth", type=float, default=0.0); ap.add_argument("--depth_from", type=int, default=0)
    ap.add_argument("--depth_acc_min", type=float, default=0.5)
    ap.add_argument("--lidar_voxel", type=float, default=0.10, help="supervision cloud = reference subsampled to this spacing [m]")
    ap.add_argument("--lidar_holdout", type=float, default=0.5, help="fraction of blocks hidden from supervision")
    ap.add_argument("--block", type=float, default=1.0, help="block size for the hold-out split [m]")
    ap.add_argument("--splat", type=int, default=2, help="pixel footprint of a projected LiDAR point")
    # geometry evaluation
    ap.add_argument("--gt", default=None, help="reference .ply (default <data>/gt_scan.ply)")
    ap.add_argument("--split_ref", default=None,
                    help="cloud that defines the supervised/held-out block split (default <data>/gt_scan.ply, the "
                         "supervision source); set it when scoring against a second reference with --gt")
    ap.add_argument("--acc_min", type=float, default=0.5); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth_max", type=float, default=None); ap.add_argument("--fuse_voxel", type=float, default=0.02)
    ap.add_argument("--taus", default="0.02,0.05,0.1,0.2"); ap.add_argument("--max_dist", type=float, default=1.0)
    ap.add_argument("--cell", type=float, default=1.0); ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--align", action="store_true", help="Sim(3)-ICP align fused points to the reference before scoring")
    ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--ply", default=None)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--prune_lidar", type=float, default=0.0,
                    help="after training, drop Gaussians sitting more than this many metres in front of the LiDAR depth "
                         "in most views where they are checked (0 = off). Uses only the supervision cloud.")
    ap.add_argument("--prune_min_views", type=int, default=2)
    ap.add_argument("--tolerances", default="0.05,0.1,0.2", help="acceptance table is also reported at these tolerances")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bg = (0.0, 0.0, 0.0)                       # real captures: black background, no alpha

    train_cams = load_colmap_scene(a.data, "train", a.downscale, max_images=a.max_train)
    test_cams = load_colmap_scene(a.data, "test", a.downscale, max_images=a.max_test)
    xyz0, rgb0 = load_init_points(a.data)
    gt_path = a.gt or os.path.join(a.data, "gt_scan.ply")
    import open3d as o3d
    gt = np.asarray(o3d.io.read_point_cloud(gt_path).points, dtype=np.float64)
    print(f"train {len(train_cams)} test {len(test_cams)} {train_cams[0].W}x{train_cams[0].H}; init {len(xyz0)} pts; reference {len(gt)} pts")

    # ---- supervision cloud: subsampled reference minus held-out blocks
    split_path = a.split_ref or os.path.join(a.data, "gt_scan.ply")
    split_ref = gt if os.path.abspath(split_path) == os.path.abspath(gt_path) else \
        np.asarray(o3d.io.read_point_cloud(split_path).points, dtype=np.float64)
    hidden = holdout_mask(split_ref, a.block, a.lidar_holdout, a.seed)
    sup = voxel_downsample(split_ref[~hidden], a.lidar_voxel)
    info_sup = dict(n_reference=int(len(gt)), n_supervision=int(len(sup)), holdout_frac_points=float(hidden.mean()),
                    split_ref=split_path,
                    lidar_voxel=a.lidar_voxel, block=a.block)
    depth_scale = 1.0
    if a.lambda_depth > 0:
        for c in train_cams:
            c.to(a.device)
        depth_scale = attach_lidar_depth(train_cams, sup, splat=a.splat)
        info_sup["depth_scale_median_m"] = depth_scale

    cfg = Config(iters=a.iters, grad_thresh=a.grad_thresh, sh_max=a.sh_max, densify_until=min(a.densify_until, a.iters // 2),
                 white_bg=False, lambda_depth=a.lambda_depth, depth_from=a.depth_from, depth_acc_min=a.depth_acc_min,
                 depth_scale=depth_scale)
    if a.eval_only:
        from gs.model import GaussianModel
        from gs.io import load_ply
        model = load_ply(a.ply or f"{a.out}/point_cloud.ply").to(a.device)
        for c in train_cams + test_cams:
            c.to(a.device)
        info = dict(train_time_min=None)
    else:
        model, log, info = train(train_cams, test_cams, xyz0, rgb0, a.out, cfg, device=a.device, backend=a.backend, seed=a.seed)
        model.save_ply(f"{a.out}/point_cloud.ply")

    # ---- optional post-training floater pruning against the sparse LiDAR depth
    prune_info = None
    if a.prune_lidar > 0:
        from gs.lidar import floater_mask, prune_gaussians
        for c in train_cams:
            c.to(a.device)
        if not any(c.depth is not None for c in train_cams):
            attach_lidar_depth(train_cams, sup, splat=a.splat)
        flag, n_checked = floater_mask(model.means.data, train_cams, a.prune_lidar, a.prune_min_views)
        n_before = model.n
        model = prune_gaussians(model, ~flag)
        prune_info = dict(thr_m=a.prune_lidar, min_views=a.prune_min_views, n_before=int(n_before), n_pruned=int(flag.sum()),
                          frac_pruned=float(flag.float().mean()), frac_checked=float((n_checked >= a.prune_min_views).float().mean()))
        print("PRUNE", prune_info)
        model.save_ply(f"{a.out}/point_cloud_pruned.ply")

    # ---- image metrics
    lp = None
    if not a.no_lpips:
        try:
            import lpips; lp = lpips.LPIPS(net="vgg").to(a.device)
        except Exception as e:
            print("lpips unavailable:", e)
    img_metrics = evaluate(model, test_cams, backend=a.backend, bg=bg, lpips_fn=lp, save_dir=f"{a.out}/test_renders")
    print("IMAGE", img_metrics)

    # ---- geometry: fuse rendered depth of the training views, score against the reference
    t0 = time.time()
    rec, n_raw = fuse_points(model, train_cams, a.backend, a.acc_min, a.stride, a.depth_max, a.fuse_voxel, bg)
    # drop fused points far outside the reference's bounding box (sky / horizon splats that passed the
    # opacity mask); they would otherwise blow up the error grid and are counted as "beyond max" anyway
    lo, hi = gt.min(0) - 2 * a.max_dist, gt.max(0) + 2 * a.max_dist
    inside = np.all((rec >= lo) & (rec <= hi), axis=1)
    frac_outside = float(1 - inside.mean()); rec = rec[inside]
    print(f"fused {n_raw} -> {len(rec)} points after {a.fuse_voxel} m voxel; {frac_outside:.2%} dropped outside reference bbox")
    align_info = None
    if a.align:
        (s, R, t), rmse = icp_sim3(rec, gt, iters=30)
        rec = apply_sim3(rec, s, R, t)
        align_info = dict(scale=s, rmse=rmse, translation_norm=float(np.linalg.norm(t)))
        print(f"aligned: scale {s:.4f} rmse {rmse:.4f}")
    taus = tuple(float(x) for x in a.taus.split(","))
    normals = estimate_normals(gt, knn=20)
    geo_all, d_rec, d_ref, p2pl = geometry_metrics(rec, gt, taus, normals, a.max_dist)
    geo = dict(all=geo_all)
    if a.lidar_holdout > 0:
        geo.update(blockwise_metrics(rec, gt, d_rec, d_ref, a.block, a.lidar_holdout, a.seed, taus, p2pl, a.max_dist,
                                     split_ref=split_ref))
    view_err = view_depth_errors(model, test_cams, gt, a.backend, a.acc_min, bg, a.splat)
    grid = error_grid(rec, np.minimum(d_rec, a.max_dist), a.cell)
    accept = acceptance_table(grid, a.tolerance)
    accept_multi = {str(t): acceptance_table(grid, float(t)) for t in a.tolerances.split(",")}
    np.savez_compressed(f"{a.out}/error_grid.npz", count=grid["count"], mean=grid["mean"], p90=grid["p90"], origin=grid["origin"], cell=grid["cell"])
    save_heatmap(grid, f"{a.out}/error_heatmap.png", os.path.basename(a.out), vmax=a.max_dist * 0.5)
    o3d.io.write_point_cloud(f"{a.out}/fused_points.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec)))

    out = dict(scene=os.path.basename(a.data.rstrip("/")), image=img_metrics, geometry=geo, test_view_depth=view_err,
               acceptance=accept, acceptance_by_tolerance=accept_multi, prune=prune_info, fusion=dict(n_raw=int(n_raw), n_fused=int(len(rec)), frac_outside_bbox=frac_outside, acc_min=a.acc_min, stride=a.stride, voxel=a.fuse_voxel),
               supervision=info_sup, align=align_info, train=info, geometry_eval_min=(time.time() - t0) / 60,
               config=cfg.todict(), n_train=len(train_cams), n_test=len(test_cams), resolution=[train_cams[0].W, train_cams[0].H])
    json.dump(out, open(f"{a.out}/metrics.json", "w"), indent=1)
    g = geo_all
    print(f"GEOMETRY all: acc_mean {g['acc_mean']:.4f} m  comp_mean {g['comp_mean']:.4f} m  F@{taus[1]} {g[f'fscore@{taus[1]}']:.3f}  "
          f"p2plane_median {g.get('p2plane_median', float('nan')):.4f} m | cells pass {accept['cells_pass']}/{accept['cells_total']} at {a.tolerance} m")
    print("ACCEPTANCE " + "  ".join(f"{t}m: {v['cells_pass']}/{v['cells_total']}" for t, v in accept_multi.items()))
    if "heldout_blocks" in geo:
        h = geo["heldout_blocks"]; s_ = geo["supervised_blocks"]
        print(f"  held-out blocks acc_mean {h['acc_mean']:.4f}  supervised blocks acc_mean {s_['acc_mean']:.4f}")


if __name__ == "__main__":
    main()
