#!/usr/bin/env python3
"""Train and evaluate 3D Gaussian Splatting on one scene.

    python train_scene.py --data data/nerf_synthetic/lego --out runs/lego
    python train_scene.py --data data/nerf_synthetic/lego --out runs/lego_nodensify --no_densify
CPU smoke test with the pure-PyTorch reference rasteriser:
    python train_scene.py --data data/nerf_synthetic/lego --out runs/smoke --device cpu --downscale 8 --iters 300 --n_init 2000
"""
import argparse, json, os, time
import numpy as np
import torch
from gs.data import load_scene, load_init_points
from gs.trainer import Config, train, evaluate
from gs.render import render


def turntable(model, cams, out, n=60, backend="auto", bg=(1.0, 1.0, 1.0)):
    """Render a circular camera path around the scene (uses the first camera's intrinsics/height)."""
    import imageio
    from gs.data import Camera
    ref = cams[0]
    c2ws = torch.stack([torch.linalg.inv(c.w2c) for c in cams])
    centers = c2ws[:, :3, 3]
    radius = float(centers.norm(dim=1).mean()); elev = float(torch.asin((centers[:, 2] / centers.norm(dim=1)).mean()))
    up = torch.tensor([0, 0, 1.0], device=centers.device)
    frames = []
    with torch.no_grad():
        for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
            pos = torch.tensor([radius * np.cos(elev) * np.cos(th), radius * np.cos(elev) * np.sin(th), radius * np.sin(elev)],
                               dtype=torch.float32, device=centers.device)
            z = -pos / pos.norm()                       # OpenCV camera looks along +z toward the origin
            x = torch.cross(z, up, dim=0); x = x / x.norm()
            y = torch.cross(z, x, dim=0)
            c2w = torch.eye(4, device=centers.device); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
            cam = Camera(ref.image, torch.linalg.inv(c2w), ref.K)
            img, _ = render(model, cam, backend=backend, bg=bg)
            frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.mimsave(out, frames, duration=0.05, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--n_init", type=int, default=100000)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--no_densify", action="store_true")
    ap.add_argument("--sh_max", type=int, default=3)
    ap.add_argument("--grad_thresh", type=float, default=0.0002)
    ap.add_argument("--densify_until", type=int, default=15000)
    ap.add_argument("--no_turntable", action="store_true")
    ap.add_argument("--lpips", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

    train_cams = load_scene(a.data, "train", a.downscale, max_images=a.max_train)
    test_cams = load_scene(a.data, "test", a.downscale, max_images=a.max_test)
    xyz, rgb = load_init_points(a.data, n_random=a.n_init)
    print(f"train {len(train_cams)} test {len(test_cams)} images {train_cams[0].W}x{train_cams[0].H}, init {len(xyz)} points, device {a.device}")
    cfg = Config(iters=a.iters, densify=not a.no_densify, sh_max=a.sh_max, grad_thresh=a.grad_thresh,
                 densify_until=min(a.densify_until, a.iters // 2))
    # NOTE: the schedule (densify 500..15k every 100, opacity reset every 3k, 30k LR decay) is never
    # rescaled for short runs; a short run is simply the first N iterations of the real schedule.
    model, log, info = train(train_cams, test_cams, xyz, rgb, a.out, cfg, device=a.device, backend=a.backend)

    lp = None
    if a.lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="vgg").to(a.device)
        except Exception as e:
            print("lpips unavailable:", e)
    metrics = evaluate(model, test_cams, backend=a.backend, lpips_fn=lp, save_dir=f"{a.out}/test_renders")
    metrics.update(info, iters=a.iters, scene=os.path.basename(a.data.rstrip("/")), n_train=len(train_cams),
                   n_test=len(test_cams), resolution=[train_cams[0].W, train_cams[0].H], config=cfg.todict())
    print("FINAL", {k: v for k, v in metrics.items() if k != "config"})
    json.dump(metrics, open(f"{a.out}/metrics.json", "w"), indent=1)
    model.save_ply(f"{a.out}/point_cloud.ply")
    if not a.no_turntable:
        turntable(model, test_cams, f"{a.out}/turntable.gif", backend=a.backend)


if __name__ == "__main__":
    main()
