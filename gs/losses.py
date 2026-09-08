import math
import torch
from torch.nn import functional as F


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)], device=device)
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim(img1, img2, window_size=11):
    """Differentiable SSIM for (H,W,3) images in [0,1] (same formulation as the 3DGS reference)."""
    a = img1.permute(2, 0, 1)[None]; b = img2.permute(2, 0, 1)[None]
    w = _gaussian_window(window_size, device=a.device)
    pad = window_size // 2
    mu1 = F.conv2d(a, w, padding=pad, groups=3); mu2 = F.conv2d(b, w, padding=pad, groups=3)
    s11 = F.conv2d(a * a, w, padding=pad, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=pad, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=pad, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m.mean()


def rgb_loss(pred, gt, lambda_dssim=0.2):
    l1 = (pred - gt).abs().mean()
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(pred, gt)), l1


def psnr(pred, gt):
    return float(-10 * torch.log10(((pred - gt) ** 2).mean()))
