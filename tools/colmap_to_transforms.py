#!/usr/bin/env python3
"""Convert a COLMAP sparse reconstruction (text or binary model) into the transforms.json
format used by gs/data.py, plus points3d.ply for Gaussian initialisation.

    colmap automatic_reconstructor --workspace_path capture --image_path capture/images --sparse 1 --dense 0
    python tools/colmap_to_transforms.py --sparse capture/sparse/0 --images capture/images --out data/my_object

Camera-to-world matrices are written in the OpenGL convention (x right, y up, z back), the
same as NeRF-synthetic, so the loader treats both identically. The scene is recentred on the
camera centroid and scaled so that the mean camera distance is ~4 (like the Blender scenes).
"""
import argparse, json, os, shutil
import numpy as np
import pycolmap


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_radius", type=float, default=4.0)
    a = ap.parse_args()
    rec = pycolmap.Reconstruction(a.sparse)
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = cam.params[0], cam.params[1] if len(cam.params) > 3 else cam.params[0], cam.params[-2], cam.params[-1]
    frames, centers = [], []
    for im in rec.images.values():
        R = qvec2rotmat(im.cam_from_world.rotation.quat[[3, 0, 1, 2]]) if hasattr(im, "cam_from_world") else qvec2rotmat(im.qvec)
        t = np.array(im.cam_from_world.translation) if hasattr(im, "cam_from_world") else im.tvec
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1                    # OpenCV -> OpenGL
        frames.append(dict(file_path=f"./images/{im.name}", transform_matrix=c2w))
        centers.append(c2w[:3, 3])
    centers = np.array(centers)
    centroid = centers.mean(0)
    scale = a.target_radius / np.linalg.norm(centers - centroid, axis=1).mean()
    for f in frames:
        f["transform_matrix"][:3, 3] = (f["transform_matrix"][:3, 3] - centroid) * scale
        f["transform_matrix"] = f["transform_matrix"].tolist()
    os.makedirs(f"{a.out}/images", exist_ok=True)
    for f in frames:
        shutil.copy(os.path.join(a.images, os.path.basename(f["file_path"])), f"{a.out}/images/")
    json.dump(dict(fl_x=fx, fl_y=fy, cx=cx, cy=cy, w=cam.width, h=cam.height, frames=frames),
              open(f"{a.out}/transforms.json", "w"), indent=1)
    # sparse points -> ply
    pts = np.array([(p.xyz - centroid) * scale for p in rec.points3D.values()])
    cols = np.array([p.color / 255.0 for p in rec.points3D.values()])
    with open(f"{a.out}/points3d.ply", "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(pts))
        for p, c in zip(pts, cols):
            fh.write(f"{p[0]} {p[1]} {p[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")
    print(f"{len(frames)} images, {len(pts)} points -> {a.out}")


if __name__ == "__main__":
    main()
