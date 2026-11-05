#!/usr/bin/env python3
"""Collect every <scene>_<config>_s<seed>/metrics.json under a results root into one table.

Prints a Markdown table with mean ± std over seeds per (scene, config) and writes
summary.csv (one row per run) and summary_by_config.csv (aggregated). Runs that have a
masked_psnr.json next to metrics.json also get the covered-pixel PSNR and train PSNR columns.

    python tools/aggregate_runs.py --root /content/drive/MyDrive/gs_geometry
"""
import argparse, csv, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or a.root
rows = []
for p in sorted(glob.glob(os.path.join(a.root, "*_s[0-9]*", "metrics.json"))):
    name = os.path.basename(os.path.dirname(p))
    mm = re.match(r"(.+?)_(.+)_s(\d+)$", name)
    if not mm:
        continue
    scene, cfg, seed = mm.group(1), mm.group(2), int(mm.group(3))
    m = json.load(open(p)); g = m["geometry"]["all"]; h = m["geometry"].get("heldout_blocks", {}); s = m["geometry"].get("supervised_blocks", {})
    r = dict(scene=scene, config=cfg, seed=seed, lambda_depth=m["config"]["lambda_depth"], psnr=m["image"]["psnr"], ssim=m["image"]["ssim"],
             lpips=m["image"].get("lpips", np.nan), n_gaussians=m["image"]["n_gaussians"],
             acc_mean=g["acc_mean"], acc_median=g["acc_median"], comp_mean=g["comp_mean"], f05=g["fscore@0.05"], f10=g["fscore@0.1"], f20=g["fscore@0.2"],
             beyond_1m=g["frac_rec_beyond_max"], p2plane_median=g.get("p2plane_median", np.nan),
             held_acc=h.get("acc_mean", np.nan), sup_acc=s.get("acc_mean", np.nan), held_comp=h.get("comp_mean", np.nan), sup_comp=s.get("comp_mean", np.nan),
             held_f10=h.get("fscore@0.1", np.nan), sup_f10=s.get("fscore@0.1", np.nan),
             tvd_abs_mean=m["test_view_depth"].get("abs_mean", np.nan), cells_pass_5cm=m["acceptance"]["cells_pass"], cells_total=m["acceptance"]["cells_total"],
             train_min=m["train"].get("train_time_min"))
    for tol, v in m.get("acceptance_by_tolerance", {}).items():
        r[f"cells_pass_{tol}"] = v["cells_pass"]
    mp = os.path.join(os.path.dirname(p), "masked_psnr.json")
    if os.path.exists(mp):
        q = json.load(open(mp)); r.update(psnr_covered=q["psnr_covered"], covered_frac=q["covered_frac"], train_psnr=q.get("train_psnr_last20"))
    rows.append(r)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scene", "config", "seed"), k))
with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
# aggregate
groups = {}
for r in rows:
    groups.setdefault((r["scene"], r["config"]), []).append(r)
agg_rows = []
metrics = ["psnr", "psnr_covered", "acc_mean", "comp_mean", "f05", "f10", "beyond_1m", "held_acc", "sup_acc", "held_comp", "sup_comp", "tvd_abs_mean"]
def fmt(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "", np.nan, np.nan
    mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (f"{mu:.3f} ± {sd:.3f}" if len(vals) > 1 else f"{mu:.3f}"), mu, sd
print("| scene | config | n | " + " | ".join(metrics) + " |")
print("|" + "---|" * (len(metrics) + 3))
for (scene, cfg), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], rs_lambda := kv[1][0]["lambda_depth"], kv[0][1])):
    cells = []; row = dict(scene=scene, config=cfg, n=len(rs), lambda_depth=rs[0]["lambda_depth"])
    for k in metrics:
        txt, mu, sd = fmt([r.get(k) for r in rs]); cells.append(txt); row[k + "_mean"] = mu; row[k + "_std"] = sd
    agg_rows.append(row)
    print(f"| {scene} | {cfg} | {len(rs)} | " + " | ".join(cells) + " |")
with open(os.path.join(out, "summary_by_config.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys())); w.writeheader(); w.writerows(agg_rows)
print(f"\n{len(rows)} runs -> {out}/summary.csv, summary_by_config.csv")
