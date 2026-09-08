import torch
from torch import nn
from torch.nn import functional as F
from .encoding import FrequencyEncoding, SHEncoding, HashGridEncoding


class TinyNeRF(nn.Module):
    """Course baseline: frequency encoding + 4-layer MLP, view-independent colour."""

    def __init__(self, n_freqs=6, width=128):
        super().__init__()
        self.enc = FrequencyEncoding(n_freqs)
        self.mlp = nn.Sequential(nn.Linear(self.enc.out_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, 4))
        self.view_dependent = False

    def forward(self, x, d=None):
        out = self.mlp(self.enc(x))
        return F.relu(out[..., 0]), torch.sigmoid(out[..., 1:])


class NeRF(nn.Module):
    """Original NeRF MLP (8x256, skip at 4, view-dependent colour branch)."""

    def __init__(self, n_freqs_pos=10, n_freqs_dir=4, width=256):
        super().__init__()
        self.enc_x, self.enc_d = FrequencyEncoding(n_freqs_pos), FrequencyEncoding(n_freqs_dir)
        dx, dd = self.enc_x.out_dim, self.enc_d.out_dim
        self.l1 = nn.ModuleList([nn.Linear(dx, width)] + [nn.Linear(width, width) for _ in range(3)])
        self.l2 = nn.ModuleList([nn.Linear(width + dx, width)] + [nn.Linear(width, width) for _ in range(3)])
        self.sigma = nn.Linear(width, 1)
        self.feat = nn.Linear(width, width)
        self.rgb = nn.Sequential(nn.Linear(width + dd, width // 2), nn.ReLU(), nn.Linear(width // 2, 3))
        self.view_dependent = True

    def forward(self, x, d):
        ex = self.enc_x(x)
        h = ex
        for l in self.l1:
            h = F.relu(l(h))
        h = torch.cat([h, ex], -1)
        for l in self.l2:
            h = F.relu(l(h))
        sigma = F.relu(self.sigma(h)[..., 0])
        h = torch.cat([self.feat(h), self.enc_d(d)], -1)
        return sigma, torch.sigmoid(self.rgb(h))


class HashNeRF(nn.Module):
    """Instant-NGP style field: hash-grid encoding -> tiny density MLP -> tiny colour MLP with SH dirs."""

    def __init__(self, bound=1.5, log2_T=19, n_levels=16, max_res=512, width=64, geo_feat=15, use_dirs=True):
        super().__init__()
        self.enc = HashGridEncoding(n_levels=n_levels, log2_T=log2_T, max_res=max_res, bound=bound)
        self.dir_enc = SHEncoding()
        self.use_dirs = use_dirs
        self.sigma_net = nn.Sequential(nn.Linear(self.enc.out_dim, width), nn.ReLU(), nn.Linear(width, 1 + geo_feat))
        self.color_net = nn.Sequential(nn.Linear(geo_feat + (16 if use_dirs else 0), width), nn.ReLU(),
                                       nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 3))
        self.view_dependent = use_dirs

    def density(self, x):
        h = self.sigma_net(self.enc(x))
        return F.softplus(h[..., 0] - 1.0), h[..., 1:]

    def forward(self, x, d):
        sigma, feat = self.density(x)
        h = torch.cat([feat, self.dir_enc(d)], -1) if self.use_dirs else feat
        return sigma, torch.sigmoid(self.color_net(h))


def build_model(name, **kw):
    return {"tiny": TinyNeRF, "nerf": NeRF, "hash": HashNeRF}[name](**kw)
