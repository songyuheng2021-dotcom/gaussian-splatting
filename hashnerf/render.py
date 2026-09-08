#!/usr/bin/env python3
"""Render products from a trained run: validation comparisons, 360-degree turntable GIF,
depth maps and a marching-cubes mesh.

    python render.py --run runs/hash --spiral --mesh
"""
import argparse, json, os
import numpy as np
import torch
import imageio
from PIL import Image
from nerf.data import load_dataset, get_rays, pose_spherical
from nerf.model import build_model
from nerf.render import render_image, OccupancyGrid


def to8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def colorize_depth(d, acc=None):
    d = d.cpu().numpy()
    lo, hi = 2.0, 6.0
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    img = np.stack([1 - x, 1 - x, 1 - x], -1)
    if acc is not None:
        img = img * acc.cpu().numpy()[..., None] + (1 - acc.cpu().numpy()[..., None])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="../data")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--spiral", action="store_true")
    ap.add_argument("--n_spiral", type=int, default=40)
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--mesh_res", type=int, default=192)
    ap.add_argument("--mesh_thresh", type=float, default=5.0)
    a = ap.parse_args()
    args = json.load(open(f"{a.run}/args.json"))
    dev = args.get("device", "cpu") if torch.cuda.is_available() else "cpu"
    torch.set_default_device(dev)
    res = a.res or args["res"]
    data = load_dataset(a.data, split_json=args.get("data_json", "transforms_train.json"), res=res)
    model = (build_model("hash", log2_T=args.get("log2_T", 19), n_levels=args.get("n_levels", 16), use_dirs=not args.get("no_dirs", False))
             if args["model"] == "hash" else build_model(args["model"]))
    model = model.to(dev)
    model.load_state_dict(torch.load(f"{a.run}/model.pt", map_location=dev))
    data["images"], data["poses"] = data["images"].to(dev), data["poses"].to(dev)
    model.eval()
    grid = None
    if os.path.exists(f"{a.run}/grid.pt"):
        grid = OccupancyGrid(res=args["grid_res"])
        grid.occupied = torch.load(f"{a.run}/grid.pt", map_location=dev)
    kw = dict(n_samples=args["n_samples"], n_importance=args["n_importance"], grid=grid)
    out = f"{a.run}/renders"; os.makedirs(out, exist_ok=True)

    # validation views: GT | prediction | depth
    rows = []
    for i in data["val_idx"][:4]:
        ro, rd = get_rays(res, res, data["focal"], data["poses"][i])
        rgb, depth = render_image(model, ro, rd, args["near"], args["far"], **kw)
        acc = (depth > 0).float()
        rows.append(np.concatenate([to8(data["images"][i].cpu().numpy()), to8(rgb.cpu().numpy()), to8(colorize_depth(depth))], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f"{out}/val_gt_pred_depth.png")
    print("saved val comparison")

    if a.spiral:
        frames = []
        for th in np.linspace(-180, 180, a.n_spiral, endpoint=False):
            c2w = pose_spherical(th, -30.0, 4.0)
            ro, rd = get_rays(res, res, data["focal"], c2w)
            rgb, depth = render_image(model, ro, rd, args["near"], args["far"], **kw)
            frames.append(np.concatenate([to8(rgb.cpu().numpy()), to8(colorize_depth(depth))], 1))
            print(f"spiral {len(frames)}/{a.n_spiral}", flush=True)
        imageio.mimsave(f"{out}/turntable.gif", frames, duration=0.08, loop=0)
        Image.fromarray(frames[0]).save(f"{out}/turntable_frame0.png")

    if a.mesh:
        from skimage import measure
        import open3d as o3d
        N, b = a.mesh_res, 1.5
        g = torch.linspace(-b, b, N)
        pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        sig = []
        with torch.no_grad():
            for i in range(0, len(pts), 131072):
                p = pts[i:i + 131072]
                if hasattr(model, "density"):
                    s = model.density(p)[0]
                else:
                    s = model(p, p)[0]
                sig.append(s)
        sig = torch.cat(sig).reshape(N, N, N).cpu().numpy()
        verts, faces, normals, _ = measure.marching_cubes(sig, level=a.mesh_thresh, spacing=(2 * b / (N - 1),) * 3)
        verts = verts - b
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()
        # keep the largest connected component to drop floaters
        tri_ids, counts, _ = mesh.cluster_connected_triangles()
        tri_ids = np.asarray(tri_ids); counts = np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[tri_ids] < counts.max() * 0.01)
        mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(f"{out}/mesh.ply", mesh)
        print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} faces")
        # quick shaded preview from three angles
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        fig = plt.figure(figsize=(12, 4))
        for k, (el, az) in enumerate([(20, -60), (20, 30), (60, 120)]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color="lightgray", edgecolor="none", shade=True)
            ax.view_init(el, az); ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))
        fig.tight_layout(); fig.savefig(f"{out}/mesh_preview.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
