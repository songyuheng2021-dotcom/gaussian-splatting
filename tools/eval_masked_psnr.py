#!/usr/bin/env python3
"""Re-score the test views of finished runs with two image metrics side by side:

  psnr_full    : every pixel (the number reported in metrics.json)
  psnr_covered : only pixels the model renders as opaque (accumulated opacity >= acc_min),
                 i.e. sky / background pixels that no Gaussian covers are excluded
  covered_frac : how many test pixels were kept

Black background with real outdoor photos means uncovered sky pixels are compared against a
bright photo and pull PSNR down for reasons unrelated to geometry. Both numbers are kept; the
covered one is a second reading, not a replacement.

    python tools/eval_masked_psnr.py --data data/eth3d/courtyard --runs /path/run_a /path/run_b
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.data import load_colmap_scene
from gs.io import load_ply
from gs.render import render

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--downscale", type=int, default=1); ap.add_argument("--acc_min", type=float, default=0.5)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--backend", default="auto")
a = ap.parse_args()
test_cams = load_colmap_scene(a.data, "test", a.downscale)
for c in test_cams:
    c.to(a.device)
for run in a.runs:
    ply = os.path.join(run, "point_cloud.ply")
    if not os.path.exists(ply):
        print("skip", run); continue
    model = load_ply(ply).to(a.device)
    full, cov, frac, train_psnr = [], [], [], None
    with torch.no_grad():
        for cam in test_cams:
            img, info = render(model, cam, backend=a.backend, bg=(0.0, 0.0, 0.0), return_depth=True)
            img = img.clamp(0, 1)
            mse_full = ((img - cam.image) ** 2).mean()
            m = info["acc"] >= a.acc_min
            mse_cov = ((img - cam.image) ** 2)[m].mean() if m.any() else torch.tensor(float("nan"))
            full.append(float(-10 * torch.log10(mse_full))); cov.append(float(-10 * torch.log10(mse_cov))); frac.append(float(m.float().mean()))
    log = os.path.join(run, "train_log.json")
    if os.path.exists(log):
        try:
            L = json.load(open(log))
            rows = L if isinstance(L, list) else L.get("log", [])
            vals = [r.get("psnr") for r in rows[-20:] if isinstance(r, dict) and r.get("psnr") is not None]
            train_psnr = float(np.mean(vals)) if vals else None
        except Exception:
            pass
    out = dict(psnr_full=float(np.mean(full)), psnr_covered=float(np.nanmean(cov)), covered_frac=float(np.mean(frac)),
               train_psnr_last20=train_psnr, n_test=len(test_cams), acc_min=a.acc_min)
    json.dump(out, open(os.path.join(run, "masked_psnr.json"), "w"), indent=1)
    print(os.path.basename(run), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
