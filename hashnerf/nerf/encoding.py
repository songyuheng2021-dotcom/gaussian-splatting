"""Input encodings: frequency (NeRF, Mildenhall et al. 2020), spherical harmonics for
view directions, and a multiresolution hash grid (Instant-NGP, Mueller et al. 2022)
implemented from scratch in pure PyTorch (all levels vectorised into one gather)."""
import math
import torch
from torch import nn


class FrequencyEncoding(nn.Module):
    def __init__(self, n_freqs=10, include_input=True):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs))
        self.include_input = include_input
        self.out_dim = 3 * (2 * n_freqs + int(include_input))

    def forward(self, x):
        xb = x[..., None] * self.freqs                          # (..., 3, F)
        enc = torch.cat([torch.sin(xb), torch.cos(xb)], -1).flatten(-2)
        return torch.cat([x, enc], -1) if self.include_input else enc


class SHEncoding(nn.Module):
    """Real spherical harmonics up to degree 4 (16 coefficients) of a unit direction."""
    out_dim = 16

    def forward(self, d):
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        return torch.stack([
            0.28209479177387814 * torch.ones_like(x),
            -0.48860251190291987 * y, 0.48860251190291987 * z, -0.48860251190291987 * x,
            1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.31539156525252005 * (2 * zz - xx - yy),
            -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy),
            -0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
            -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
            -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3 * yy)], -1)


class HashGridEncoding(nn.Module):
    """Multiresolution hash encoding.
    L levels of 3-D grids growing geometrically from base_res to max_res; every level owns
    a table of 2**log2_T feature vectors of size F (one flat parameter, level offsets added
    to the indices). Coarse levels that fit in the table are indexed densely, finer levels
    through the spatial hash of Instant-NGP. The 8 corner features are trilinearly
    interpolated and the levels concatenated -> (L*F) dims."""

    PRIMES = (1, 2654435761, 805459861)

    def __init__(self, n_levels=16, n_features=2, log2_T=19, base_res=16, max_res=512, bound=1.5):
        super().__init__()
        self.L, self.F, self.T, self.bound = n_levels, n_features, 2 ** log2_T, bound
        b = math.exp((math.log(max_res) - math.log(base_res)) / (n_levels - 1))
        res = torch.tensor([int(math.floor(base_res * b ** l)) for l in range(n_levels)])
        self.register_buffer("res", res)
        self.table = nn.Parameter(torch.empty(n_levels * self.T, n_features).uniform_(-1e-4, 1e-4))
        self.register_buffer("corners", torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]))
        self.register_buffer("primes", torch.tensor(self.PRIMES))
        self.register_buffer("dense", (res + 1) ** 3 <= self.T)
        self.register_buffer("offset", torch.arange(n_levels) * self.T)
        self.out_dim = n_levels * n_features

    def forward(self, x):
        """x: (N,3) world coords in [-bound, bound]^3 -> (N, L*F)."""
        u = ((x / self.bound + 1) * 0.5).clamp(0, 1)
        p = u[:, None, :] * self.res[None, :, None]                              # (N,L,3)
        p0 = torch.floor(p)
        w = p - p0
        cell = (p0.long()[:, :, None, :] + self.corners[None, None]).clamp_(max=self.res[None, :, None, None])
        r1 = (self.res + 1)[None, :, None]
        dense_idx = (cell[..., 0] * r1 + cell[..., 1]) * r1 + cell[..., 2]
        h = (cell[..., 0] * self.primes[0]) ^ (cell[..., 1] * self.primes[1]) ^ (cell[..., 2] * self.primes[2])
        idx = torch.where(self.dense[None, :, None], dense_idx, h % self.T) + self.offset[None, :, None]
        feats = self.table[idx]                                                  # (N,L,8,F)
        cw = torch.where(self.corners[None, None].bool(), w[:, :, None, :], 1 - w[:, :, None, :]).prod(-1)
        return (feats * cw[..., None]).sum(2).flatten(1)
