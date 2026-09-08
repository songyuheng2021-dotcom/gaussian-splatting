#!/usr/bin/env python3
"""Collect runs/*/final.json + log.json into a markdown table and a convergence plot."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = {"tiny": "TinyNeRF (course baseline: freq. enc. + 4x128 MLP)",
         "nerf": "NeRF (8x256 MLP, hierarchical sampling 64+64)",
         "hash": "HashNeRF (multires hash grid + 2 tiny MLPs + occupancy grid)"}
rows, fig = [], plt.figure(figsize=(6, 4))
for run in ["tiny", "nerf", "hash"]:
    f = f"runs/{run}/final.json"
    if not os.path.exists(f):
        continue
    m = json.load(open(f))
    log = json.load(open(f"runs/{run}/log.json")) if os.path.exists(f"runs/{run}/log.json") else []
    log = log + [dict(psnr=m["psnr"], time_min=m["time_min"])]
    rows.append(f"| {names[run]} | {m['params']/1e6:.2f} M | {m['it']} | {m['time_min']:.0f} | "
                f"{m['psnr']:.2f} | {m['ssim']:.3f} |")
    plt.plot([l["time_min"] for l in log], [l["psnr"] for l in log], "o-", label=run)
plt.xlabel("wall-clock training time [min, 2 CPU cores]"); plt.ylabel("val PSNR [dB]"); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig("runs/convergence.png", dpi=140)
tbl = ("| model | params | iters | train time [min] | PSNR | SSIM |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
open("runs/table.md", "w").write(tbl + "\n")
print(tbl)
