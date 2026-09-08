#!/usr/bin/env python3
"""Collect results/ into markdown tables + figures (results/report.md, results/*.png)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Numbers reported in Kerbl et al. 2023 (30k iterations, NeRF-synthetic, PSNR)
PAPER_PSNR = {"chair": 35.83, "drums": 26.15, "ficus": 34.87, "hotdog": 37.72, "lego": 35.78,
              "materials": 30.00, "mic": 35.36, "ship": 30.80}

ABL_NAMES = {"full": "full (30k, densify to 15k, SH deg 3, grad thr 2e-4)",
             "no_densify": "no adaptive density control (300k fixed random Gaussians)",
             "sh0_view_independent": "SH degree 0 (view-independent colour)",
             "grad_thresh_x2": "densification gradient threshold x2 (4e-4)",
             "grad_thresh_half": "densification gradient threshold /2 (1e-4)",
             "densify_until_5k": "densification stopped at 5k iterations",
             "iters_7k": "7k iterations only"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main(root="results"):
    md = []
    # ---- per scene
    rows, ps = [], []
    for f in sorted(glob.glob(f"{root}/scenes/*/metrics.json")):
        m = json.load(open(f)); s = m["scene"]
        rows.append(f"| {s} | {m['psnr']:.2f} | {PAPER_PSNR.get(s, float('nan')):.2f} | {m['ssim']:.3f} | "
                    f"{m.get('lpips', float('nan')):.3f} | {m['n_gaussians']/1e3:.0f}k | {m['train_time_min']:.0f} | {m['fps']:.0f} |")
        ps.append((m["psnr"], PAPER_PSNR.get(s, np.nan)))
    if rows:
        mean = np.nanmean(np.array(ps), 0)
        rows.append(f"| **mean** | **{mean[0]:.2f}** | **{mean[1]:.2f}** | | | | | |")
        md.append("## NeRF-synthetic benchmark (30k iterations, 800x800, 200 test views per scene)\n\n"
                  "| scene | PSNR (ours) | PSNR (paper) | SSIM | LPIPS | #Gaussians | train [min] | render FPS |\n"
                  "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    # ---- ablation
    rows = []
    for k, name in ABL_NAMES.items():
        m = load(f"{root}/ablation/{k}/metrics.json")
        if m:
            rows.append(f"| {name} | {m['psnr']:.2f} | {m['ssim']:.3f} | {m.get('lpips', float('nan')):.3f} | "
                        f"{m['n_gaussians']/1e3:.0f}k | {m['train_time_min']:.0f} | {m['fps']:.0f} |")
    if rows:
        md.append("## Ablation (lego)\n\n| variant | PSNR | SSIM | LPIPS | #Gaussians | train [min] | FPS |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    # ---- NeRF comparison
    cmp = load(f"{root}/nerf_comparison.json")
    if cmp:
        rows = [f"| {r['method']} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['train_min']:.0f} | {r['fps']:.1f} | {r['note']} |" for r in cmp]
        md.append("## Same scene, same GPU: HashNeRF (previous project) vs 3D Gaussian Splatting (lego)\n\n"
                  "| method | PSNR | SSIM | train [min] | render FPS | note |\n|---|---|---|---|---|---|\n" + "\n".join(rows))
    # ---- figures: gaussian count + psnr over training
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for f in sorted(glob.glob(f"{root}/scenes/*/train_log.json")):
        log = json.load(open(f)); s = f.split("/")[-2]
        ax[0].plot([l["it"] for l in log], [l["n"] / 1e3 for l in log], label=s)
        ax[1].plot([l["it"] for l in log], [l["psnr"] for l in log], label=s, lw=0.8)
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("#Gaussians [k]"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("train PSNR"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{root}/training_curves.png", dpi=140)
    open(f"{root}/report.md", "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
