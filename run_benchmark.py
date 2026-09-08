#!/usr/bin/env python3
"""Run the full experiment set, resumable (a run is skipped if its metrics.json exists).

    python run_benchmark.py --data data/nerf_synthetic --out results --scenes all --ablations
"""
import argparse, os, subprocess, sys

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nerf_synthetic")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenes", nargs="+", default=["all"])
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--ablation_scene", default="lego")
    ap.add_argument("--ablation_iters", type=int, default=30000)
    a = ap.parse_args()
    scenes = SCENES if a.scenes == ["all"] else a.scenes
    py = sys.executable
    for s in scenes:
        out = f"{a.out}/scenes/{s}"
        if os.path.exists(f"{out}/metrics.json"):
            print("skip", s); continue
        run([py, "train_scene.py", "--data", f"{a.data}/{s}", "--out", out, "--iters", str(a.iters), "--lpips"])
    if a.ablations:
        s = a.ablation_scene
        variants = {
            "full": [],
            "no_densify": ["--no_densify", "--n_init", "300000"],
            "sh0_view_independent": ["--sh_max", "0"],
            "grad_thresh_x2": ["--grad_thresh", "0.0004"],
            "grad_thresh_half": ["--grad_thresh", "0.0001"],
            "densify_until_5k": ["--densify_until", "5000"],
            "iters_7k": ["--iters", "7000"],
        }
        for name, extra in variants.items():
            out = f"{a.out}/ablation/{name}"
            if os.path.exists(f"{out}/metrics.json"):
                print("skip", name); continue
            cmd = [py, "train_scene.py", "--data", f"{a.data}/{s}", "--out", out, "--iters", str(a.ablation_iters),
                   "--lpips", "--no_turntable"] + extra
            run(cmd)


if __name__ == "__main__":
    main()
