#!/usr/bin/env python3
"""Train the HashNeRF from the previous project on the same lego scene and GPU, then write
results/nerf_comparison.json with PSNR / SSIM / training time / render FPS next to 3DGS.

    python nerf_compare.py --data data/nerf_synthetic/lego --results results --res 400 --iters 15000
"""
import argparse, json, os, subprocess, sys, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nerf_synthetic/lego")
    ap.add_argument("--results", default="results")
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--iters", type=int, default=15000)
    a = ap.parse_args()
    out = f"{a.results}/hashnerf_lego"
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f"{out}/final.json"):
        subprocess.run([sys.executable, "train.py", "--data", os.path.abspath(a.data), "--model", "hash", "--res", str(a.res),
                        "--iters", str(a.iters), "--batch", "4096", "--eval_every", "2500", "--test_json", "transforms_test.json",
                        "--n_test", "25", "--out", os.path.abspath(out)], cwd=f"{here}/hashnerf", check=True)
        subprocess.run([sys.executable, "render.py", "--run", os.path.abspath(out), "--data", os.path.abspath(a.data), "--spiral"],
                       cwd=f"{here}/hashnerf", check=True)
    nerf = json.load(open(f"{out}/final.json"))
    # render speed of the NeRF
    sys.path.insert(0, f"{here}/hashnerf")
    from nerf.data import load_dataset, get_rays
    from nerf.model import build_model
    from nerf.render import render_image, OccupancyGrid
    dev = "cuda"; torch.set_default_device(dev)
    args = json.load(open(f"{out}/args.json"))
    data = load_dataset(a.data, split_json="transforms_test.json", res=a.res, n_val=1)
    model = build_model("hash").to(dev); model.load_state_dict(torch.load(f"{out}/model.pt", map_location=dev)); model.eval()
    grid = OccupancyGrid(res=args["grid_res"]); grid.occupied = torch.load(f"{out}/grid.pt", map_location=dev)
    ro, rd = get_rays(a.res, a.res, data["focal"], data["poses"][0].to(dev))
    render_image(model, ro, rd, 2.0, 6.0, n_samples=64, grid=grid)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3):
        render_image(model, ro, rd, 2.0, 6.0, n_samples=64, grid=grid)
    torch.cuda.synchronize(); fps_nerf = 3 / (time.time() - t)
    gs = json.load(open(f"{a.results}/scenes/lego/metrics.json"))
    rows = [dict(method="HashNeRF (Instant-NGP style, pure PyTorch, previous project)", psnr=nerf["psnr"], ssim=nerf["ssim"],
                 train_min=nerf["time_min"], fps=fps_nerf, note=f"{a.res}x{a.res}, {nerf['it']} iters, 25 test views"),
            dict(method="3D Gaussian Splatting (this project)", psnr=gs["psnr"], ssim=gs["ssim"], train_min=gs["train_time_min"],
                 fps=gs["fps"], note=f"800x800, {gs['iters']} iters, {gs['n_test']} test views, {gs['n_gaussians']//1000}k Gaussians")]
    json.dump(rows, open(f"{a.results}/nerf_comparison.json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
