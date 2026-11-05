#!/usr/bin/env python3
"""Build a small synthetic scene in the COLMAP-text layout used by run_geometry.py, with a known
geometric truth, so the whole geometry pipeline can be exercised on a CPU in a minute.

    python tools/make_synthetic_scene.py --out data/synthetic/wave --n_cams 24 --res 96

Truth: ~1200 opaque Gaussians on a wavy surface z = 0.3 sin(2x) + 0.2 cos(2y) over [-2,2]^2, with a
0.6 m box standing on it. Cameras orbit above at radius 5. Images are rendered with the reference
rasteriser; the truth centres form gt_scan.ply (plus a dense sampling of the analytic surface);
colmap/points3D.txt holds a sparse noisy subset as "SfM" initialisation.
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.model import GaussianModel
from gs.render import render_reference
from gs.data import Camera


def surface(x, y):
    return 0.3 * np.sin(2 * x) + 0.2 * np.cos(2 * y)


def look_at(pos, target, up=np.array([0, 0, 1.0])):
    z = target - pos; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return np.linalg.inv(c2w)


def rotmat_to_qvec(R):
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sign(R[2, 1] - R[1, 2]) * np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sign(R[0, 2] - R[2, 0]) * np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sign(R[1, 0] - R[0, 1]) * np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n_cams", type=int, default=24)
    ap.add_argument("--res", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    g = np.linspace(-2, 2, 32)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), surface(xx.ravel(), yy.ravel())], 1)
    # a box 0.6 m wide standing at (0.8, -0.8), 0.6 m tall: 5 faces sampled
    s = np.linspace(-0.3, 0.3, 6)
    faces = []
    for u in s:
        for v in np.linspace(0, 0.6, 6):
            faces += [[0.8 + u, -0.8 - 0.3, v], [0.8 + u, -0.8 + 0.3, v], [0.8 - 0.3, -0.8 + u, v], [0.8 + 0.3, -0.8 + u, v]]
        for v in s:
            faces.append([0.8 + u, -0.8 + v, 0.6])
    faces = np.array(faces); faces[:, 2] += surface(0.8, -0.8)
    pts = np.concatenate([pts, faces])
    col = np.stack([0.5 + 0.5 * np.sin(3 * pts[:, 0]), 0.5 + 0.5 * np.cos(3 * pts[:, 1]), 0.3 + 0.7 * (pts[:, 2] - pts[:, 2].min()) / np.ptp(pts[:, 2])], 1).clip(0, 1)
    model = GaussianModel(torch.tensor(pts, dtype=torch.float32), torch.tensor(col, dtype=torch.float32), sh_degree=0, init_scale=0.09)
    model.opacity_logits.data[:] = 8.0
    W = H = a.res; f = 0.9 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    os.makedirs(f"{a.out}/images", exist_ok=True); os.makedirs(f"{a.out}/colmap", exist_ok=True)
    lines = ["# Image list"]
    for i in range(a.n_cams):
        th = 2 * np.pi * i / a.n_cams
        pos = np.array([4.5 * np.cos(th), 4.5 * np.sin(th), 3.0 + 0.5 * np.sin(3 * th)])
        w2c = look_at(pos, np.array([0, 0, 0.0]))
        cam = Camera(torch.zeros(H, W, 3), torch.tensor(w2c, dtype=torch.float32), torch.tensor(K, dtype=torch.float32))
        with torch.no_grad():
            img, _ = render_reference(model, cam, bg=(0, 0, 0))
        name = f"cam_{i:03d}.png"
        Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(f"{a.out}/images/{name}")
        q = rotmat_to_qvec(w2c[:3, :3]); t = w2c[:3, 3]
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}")
        lines.append("")
    open(f"{a.out}/colmap/images.txt", "w").write("\n".join(lines) + "\n")
    open(f"{a.out}/colmap/cameras.txt", "w").write(f"# cams\n1 PINHOLE {W} {H} {f} {f} {W / 2} {H / 2}\n")
    sel = rng.choice(len(pts), 300, replace=False)
    noisy = pts[sel] + rng.normal(scale=0.05, size=(300, 3))
    open(f"{a.out}/colmap/points3D.txt", "w").write("\n".join(
        f"{i + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0] * 255)} {int(c[1] * 255)} {int(c[2] * 255)} 0.1 1 0" for i, (p, c) in enumerate(zip(noisy, col[sel]))) + "\n")
    # dense analytic reference: surface at 2 cm plus box faces
    gd = np.arange(-2, 2, 0.02)
    gx, gy = np.meshgrid(gd, gd, indexing="ij")
    dense = np.stack([gx.ravel(), gy.ravel(), surface(gx.ravel(), gy.ravel())], 1)
    inside = (np.abs(dense[:, 0] - 0.8) < 0.3) & (np.abs(dense[:, 1] + 0.8) < 0.3)
    dense = np.concatenate([dense[~inside], faces])
    import open3d as o3d
    o3d.io.write_point_cloud(f"{a.out}/gt_scan.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense)))
    json.dump(dict(n_images=a.n_cams, n_gt_points=int(len(dense)), truth_gaussians=int(len(pts))), open(f"{a.out}/scene_info.json", "w"), indent=1)
    print(f"wrote {a.n_cams} images {W}x{H}, {len(dense)} reference points, {len(noisy)} init points -> {a.out}")


if __name__ == "__main__":
    main()
