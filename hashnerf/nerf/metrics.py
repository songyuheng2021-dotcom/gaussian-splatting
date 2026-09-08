import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(a, b):
    mse = float(((a - b) ** 2).mean().cpu())
    return -10 * np.log10(mse)


def ssim(a, b):
    return float(structural_similarity(a.cpu().numpy(), b.cpu().numpy(), channel_axis=-1, data_range=1.0))


def lpips_fn(lp, a, b):
    with torch.no_grad():
        x = a.permute(2, 0, 1)[None] * 2 - 1
        y = b.permute(2, 0, 1)[None] * 2 - 1
        return float(lp(x, y))
