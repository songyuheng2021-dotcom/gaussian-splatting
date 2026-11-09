#!/usr/bin/env python3
"""Recompute the supervised / held-out block metrics of finished runs with the corrected split.

Before the fix, blockwise_metrics ran the block lottery separately on the reconstruction and on the reference.
The two clouds occupy different blocks (floaters, holes), so the draws fell out of step and the reconstruction's
"held-out" points were an arbitrary half of it; the reference side (completeness) was right. Training is not
affected: the supervision was always drawn from the reference with its own lottery, which is exactly the split the
corrected code now applies to both clouds.

Inputs per run directory (all written by run_geometry.py): args.json, metrics.json, fused_points.ply (the scored
reconstruction, after the bbox filter and any alignment). The reference is args.gt or <data>/gt_scan.ply.
metrics.json is rewritten in place: geometry.supervised_blocks / heldout_blocks / split are replaced, the old values
are kept under geometry.heldout_before_fix, and the file records rescored_heldout=true.

    python tools/rescore_heldout.py /content/drive/MyDrive/gs_geometry/courtyard_*_s*
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gs.geometry import geometry_metrics, estimate_normals, blockwise_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--repo", default=".", help="directory the runs' relative --data paths are relative to")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    import open3d as o3d
    read = lambda p: np.asarray(o3d.io.read_point_cloud(p).points, dtype=np.float64)
    cache = {}
    for run in a.runs:
        need = [os.path.join(run, f) for f in ("args.json", "metrics.json", "fused_points.ply")]
        if not all(os.path.exists(p) for p in need):
            print(f"skip {run}: missing {[os.path.basename(p) for p in need if not os.path.exists(p)]}")
            continue
        args = json.load(open(need[0])); m = json.load(open(need[1]))
        if args.get("lidar_holdout", 0) <= 0:
            print(f"skip {run}: no hold-out"); continue
        data = args["data"] if os.path.isabs(args["data"]) else os.path.join(a.repo, args["data"])
        gt_path = args.get("gt") or os.path.join(data, "gt_scan.ply")
        split_path = args.get("split_ref") or os.path.join(data, "gt_scan.ply")
        if gt_path not in cache:
            gt = read(gt_path); cache[gt_path] = (gt, estimate_normals(gt, knn=20))
        gt, normals = cache[gt_path]
        split_ref = gt if os.path.abspath(split_path) == os.path.abspath(gt_path) else read(split_path)
        rec = read(need[2])
        taus = tuple(float(x) for x in args["taus"].split(","))
        _, d_rec, d_ref, p2pl = geometry_metrics(rec, gt, taus, normals, args["max_dist"])
        new = blockwise_metrics(rec, gt, d_rec, d_ref, args["block"], args["lidar_holdout"], args["seed"], taus, p2pl,
                                args["max_dist"], split_ref=split_ref)
        geo = m["geometry"]
        if not m.get("rescored_heldout"):
            geo["heldout_before_fix"] = {k: geo.get(k) for k in ("supervised_blocks", "heldout_blocks")}
        for k in ("supervised_blocks", "heldout_blocks", "split"):
            geo.pop(k, None)
        geo.update(new); m["rescored_heldout"] = True
        t = f"fscore@{taus[2]}" if len(taus) > 2 else f"fscore@{taus[-1]}"
        row = lambda d: f"acc {d['acc_mean']:.4f} comp {d['comp_mean']:.4f} {t} {d[t]:.3f}" if d else "-"
        old = geo["heldout_before_fix"]
        print(f"{os.path.basename(run.rstrip('/'))}  split {new['split']}")
        for k in ("supervised_blocks", "heldout_blocks"):
            print(f"   {k:18s} before: {row(old.get(k))}   after: {row(new.get(k))}")
        if not a.dry_run:
            json.dump(m, open(need[1], "w"), indent=1)


if __name__ == "__main__":
    main()
