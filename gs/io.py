"""Load a GaussianModel back from the standard 3DGS .ply layout written by GaussianModel.save_ply."""
import numpy as np
import torch
from .model import GaussianModel


def _read_ply_binary(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        n = int([l for l in header if l.startswith("element vertex")][0].split()[-1])
        names = [l.split()[-1] for l in header if l.startswith("property")]
        arr = np.frombuffer(f.read(), dtype=np.dtype([(nm, "<f4") for nm in names]), count=n)
    return arr, names


def load_ply(path):
    arr, names = _read_ply_binary(path)
    n = len(arr)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1)
    n_rest = len([nm for nm in names if nm.startswith("f_rest_")])
    K = 1 + n_rest // 3
    deg = int(round(np.sqrt(K))) - 1
    m = GaussianModel(torch.from_numpy(xyz.copy()), torch.full((n, 3), 0.5), sh_degree=deg, init_scale=0.01)
    sh = np.zeros((n, K, 3), np.float32)
    sh[:, 0] = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], 1)
    if n_rest:
        rest = np.stack([arr[f"f_rest_{i}"] for i in range(n_rest)], 1).reshape(n, 3, K - 1).transpose(0, 2, 1)
        sh[:, 1:] = rest
    with torch.no_grad():
        m.sh.data = torch.from_numpy(sh)
        m.opacity_logits.data = torch.from_numpy(arr["opacity"].copy())
        m.log_scales.data = torch.from_numpy(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], 1).copy())
        m.quats.data = torch.from_numpy(np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], 1).copy())
    m.active_sh_degree = deg
    return m
