"""NeRF-synthetic style dataset (transforms json + RGBA PNGs), with train/val split."""
import json, os
import numpy as np
import torch
from PIL import Image


def load_dataset(root, split_json="transforms_train.json", res=200, n_val=10, white_bg=True):
    meta = json.load(open(os.path.join(root, split_json)))
    imgs, poses = [], []
    for fr in meta["frames"]:
        rel = fr["file_path"].replace("./", "")
        p = os.path.join(root, rel + ".png")                      # NeRF-synthetic layout: train/r_0.png
        if not os.path.exists(p):
            p = os.path.join(root, "images", rel + ".png")        # this repo's layout: images/train/r_0.png
        im = Image.open(p).convert("RGBA").resize((res, res), Image.LANCZOS)
        im = np.asarray(im, dtype=np.float32) / 255.0
        rgb = im[..., :3] * im[..., 3:] + (1 - im[..., 3:]) if white_bg else im[..., :3]
        imgs.append(rgb)
        poses.append(np.array(fr["transform_matrix"], dtype=np.float32))
    imgs, poses = np.stack(imgs), np.stack(poses)
    focal = 0.5 * res / np.tan(0.5 * meta["camera_angle_x"])
    idx = np.arange(len(imgs))
    val = idx[::max(1, len(imgs) // n_val)][:n_val]
    train = np.setdiff1d(idx, val) if n_val < len(imgs) else idx
    return dict(images=torch.from_numpy(imgs), poses=torch.from_numpy(poses), focal=float(focal),
                H=res, W=res, train_idx=train, val_idx=val)


def get_rays(H, W, focal, c2w):
    """OpenGL camera convention (x right, y up, looking down -z). Returns (H,W,3) origins and dirs."""
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing="xy")
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def pose_spherical(theta, phi, radius):
    """Camera-to-world for a camera on a sphere looking at the origin (NeRF convention)."""
    def trans_t(t): return torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=torch.float32)
    def rot_phi(p): return torch.tensor([[1, 0, 0, 0], [0, np.cos(p), -np.sin(p), 0], [0, np.sin(p), np.cos(p), 0], [0, 0, 0, 1]], dtype=torch.float32)
    def rot_theta(t): return torch.tensor([[np.cos(t), 0, -np.sin(t), 0], [0, 1, 0, 0], [np.sin(t), 0, np.cos(t), 0], [0, 0, 0, 1]], dtype=torch.float32)
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32) @ c2w
    return c2w.to(torch.get_default_device()) if hasattr(torch, 'get_default_device') else c2w
