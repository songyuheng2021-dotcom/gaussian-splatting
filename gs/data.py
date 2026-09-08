"""Datasets for 3D Gaussian Splatting.

Supports
  * NeRF-synthetic (Blender) scenes: transforms_{train,test}.json + RGBA PNGs
  * COLMAP captures converted with tools/colmap_to_transforms.py (same json format,
    plus an optional points3d.ply used to initialise the Gaussians)
Cameras are returned in the OpenCV convention expected by the rasteriser
(x right, y down, z forward), with world-to-camera matrices and intrinsics.
"""
import json, os
import numpy as np
import torch
from PIL import Image


def _load_json(root, name):
    p = os.path.join(root, name)
    return json.load(open(p)) if os.path.exists(p) else None


class Camera:
    def __init__(self, image, w2c, K, name=""):
        self.image, self.w2c, self.K, self.name = image, w2c, K, name
        self.H, self.W = image.shape[:2]


def opengl_to_opencv(c2w):
    c2w = c2w.copy()
    c2w[:3, 1:3] *= -1          # flip y and z axes of the camera frame
    return c2w


def load_scene(root, split="train", downscale=1, white_bg=True, max_images=None):
    """Return a list of Camera. `split` is 'train' or 'test'. If the scene has no test
    split (e.g. only transforms_train.json), every 8th image is held out as test."""
    meta_tr, meta_te = _load_json(root, "transforms_train.json"), _load_json(root, "transforms_test.json")
    if meta_tr is None:
        meta_tr = _load_json(root, "transforms.json")
    frames = None
    if meta_te is not None and split == "test":
        frames, meta = meta_te["frames"], meta_te
    elif meta_te is not None:
        frames, meta = meta_tr["frames"], meta_tr
    else:                                            # single json: hold out every 8th
        fr = meta_tr["frames"]
        frames = [f for i, f in enumerate(fr) if (i % 8 == 0) == (split == "test")]
        meta = meta_tr
    if max_images:
        frames = frames[:max_images]
    cams = []
    for fr in frames:
        p = os.path.join(root, fr["file_path"])
        if not os.path.exists(p):
            p = p + ".png"
        im = Image.open(p)
        if downscale > 1:
            im = im.resize((im.width // downscale, im.height // downscale), Image.LANCZOS)
        im = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        rgb = im[..., :3] * im[..., 3:] + (1 - im[..., 3:]) * (1.0 if white_bg else 0.0)
        H, W = rgb.shape[:2]
        if "camera_angle_x" in meta:
            fx = fy = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
            cx, cy = W / 2, H / 2
        else:
            s = 1.0 / downscale
            fx, fy, cx, cy = meta["fl_x"] * s, meta["fl_y"] * s, meta["cx"] * s, meta["cy"] * s
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        c2w = opengl_to_opencv(np.array(fr["transform_matrix"], dtype=np.float32))
        w2c = np.linalg.inv(c2w).astype(np.float32)
        cams.append(Camera(torch.from_numpy(rgb), torch.from_numpy(w2c), torch.from_numpy(K), os.path.basename(p)))
    return cams


def load_init_points(root, n_random=100_000, bound=1.3, seed=0):
    """Initial Gaussian centres: COLMAP sparse points if available, else random in a cube."""
    p = os.path.join(root, "points3d.ply")
    if os.path.exists(p):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(p)
        xyz = np.asarray(pc.points, dtype=np.float32)
        rgb = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else np.full_like(xyz, 0.5)
        return torch.from_numpy(xyz), torch.from_numpy(rgb)
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n_random, 3, generator=g) * 2 - 1) * bound
    rgb = torch.rand(n_random, 3, generator=g)
    return xyz, rgb


def scene_extent(cams):
    """Radius of the camera set: used to scale densification thresholds like the paper."""
    centers = torch.stack([torch.linalg.inv(c.w2c)[:3, 3] for c in cams])
    return float((centers - centers.mean(0)).norm(dim=1).max() * 1.1)
