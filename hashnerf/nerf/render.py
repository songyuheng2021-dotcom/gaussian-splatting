"""Ray sampling, occupancy-grid empty-space skipping and differentiable volume rendering."""
import torch
from torch.nn import functional as F


class OccupancyGrid:
    """Binary grid over [-bound,bound]^3 maintained from the field's density (Instant-NGP sec. 5.2).
    Samples in empty cells are skipped so the MLP is evaluated only where it matters."""

    def __init__(self, bound=1.5, res=64, decay=0.9, thresh=0.1):
        self.bound, self.res, self.decay, self.thresh = bound, res, decay, thresh
        self.density = torch.zeros(res, res, res)
        self.occupied = torch.ones(res, res, res, dtype=torch.bool)
        g = (torch.arange(res) + 0.5) / res
        self.centers = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * 2 * bound - bound
        self.cell = 2 * bound / res

    @torch.no_grad()
    def update(self, model, chunk=65536):
        pts = self.centers + (torch.rand_like(self.centers) - 0.5) * self.cell
        sig = torch.cat([model.density(pts[i:i + chunk])[0] for i in range(0, len(pts), chunk)])
        sig = sig.reshape(self.res, self.res, self.res)
        self.density = torch.maximum(self.density * self.decay, sig)
        occ = self.density > self.thresh
        occ = F.max_pool3d(occ[None, None].float(), 3, 1, 1)[0, 0] > 0   # dilate by one cell
        self.occupied = occ

    def query(self, x):
        i = ((x + self.bound) / (2 * self.bound) * self.res).long().clamp(0, self.res - 1)
        inside = (x.abs() < self.bound).all(-1)
        return self.occupied[i[..., 0], i[..., 1], i[..., 2]] & inside

    def occupancy_ratio(self):
        return float(self.occupied.float().mean())


def sample_along_rays(rays_o, rays_d, near, far, n_samples, perturb):
    t = torch.linspace(near, far, n_samples)
    t = t.expand(rays_o.shape[0], n_samples)
    if perturb:
        mids = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = torch.cat([mids, t[:, -1:]], -1)
        lower = torch.cat([t[:, :1], mids], -1)
        t = lower + (upper - lower) * torch.rand_like(t)
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t[..., None]
    return pts, t


def sample_pdf(bins, weights, n, perturb):
    """Hierarchical sampling: inverse-CDF sample `n` new depths from coarse weights."""
    w = weights + 1e-5
    pdf = w / w.sum(-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    u = torch.rand(cdf.shape[0], n) if perturb else torch.linspace(0, 1, n).expand(cdf.shape[0], n)
    u = u.contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    lo = (idx - 1).clamp(min=0)
    hi = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_lo, cdf_hi = torch.gather(cdf, 1, lo), torch.gather(cdf, 1, hi)
    b_lo, b_hi = torch.gather(bins, 1, lo), torch.gather(bins, 1, hi)
    denom = torch.where(cdf_hi - cdf_lo < 1e-5, torch.ones_like(cdf_lo), cdf_hi - cdf_lo)
    return b_lo + (u - cdf_lo) / denom * (b_hi - b_lo)


def composite(sigma, rgb, t, rays_d, white_bg=True):
    """sigma (R,S), rgb (R,S,3), t (R,S) -> rgb_map, depth, acc, weights."""
    dists = t[:, 1:] - t[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], -1) * rays_d.norm(dim=-1, keepdim=True)
    alpha = 1 - torch.exp(-sigma * dists)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    w = alpha * T
    rgb_map = (w[..., None] * rgb).sum(1)
    depth = (w * t).sum(1)
    acc = w.sum(1)
    if white_bg:
        rgb_map = rgb_map + (1 - acc[..., None])
    return rgb_map, depth, acc, w


def render_rays(model, rays_o, rays_d, near, far, n_samples=64, n_importance=0, perturb=False,
                grid=None, chunk=None, white_bg=True):
    """Render a batch of rays. If `grid` is given, samples in empty cells are skipped."""
    pts, t = sample_along_rays(rays_o, rays_d, near, far, n_samples, perturb)
    R, S = t.shape
    d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    dirs = d[:, None, :].expand(R, S, 3)

    def query(pts, dirs):
        flat_p, flat_d = pts.reshape(-1, 3), dirs.reshape(-1, 3)
        sigma = torch.zeros(flat_p.shape[0])
        rgb = torch.zeros(flat_p.shape[0], 3)
        mask = grid.query(flat_p) if grid is not None else torch.ones(flat_p.shape[0], dtype=torch.bool)
        if mask.any():
            s, c = model(flat_p[mask], flat_d[mask])
            sigma = sigma.masked_scatter(mask, s)
            rgb = rgb.masked_scatter(mask[:, None].expand(-1, 3), c)
        return sigma.reshape(pts.shape[:-1]), rgb.reshape(*pts.shape[:-1], 3), int(mask.sum())

    sigma, rgb, n_eval = query(pts, dirs)
    out = {}
    if n_importance > 0:
        rgb0, depth0, acc0, w0 = composite(sigma, rgb, t, rays_d, white_bg)
        out.update(rgb_coarse=rgb0)
        mids = 0.5 * (t[:, 1:] + t[:, :-1])
        t_fine = sample_pdf(mids, w0[:, 1:-1].detach(), n_importance, perturb)
        t, _ = torch.sort(torch.cat([t, t_fine], -1), -1)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[..., None]
        dirs = d[:, None, :].expand(R, t.shape[1], 3)
        sigma, rgb, n2 = query(pts, dirs)
        n_eval += n2
    rgb_map, depth, acc, w = composite(sigma, rgb, t, rays_d, white_bg)
    out.update(rgb=rgb_map, depth=depth, acc=acc, n_eval=n_eval, n_total=t.numel())
    return out


@torch.no_grad()
def render_image(model, rays_o, rays_d, near, far, chunk=4096, **kw):
    H, W = rays_o.shape[:2]
    ro, rd = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    rgbs, depths = [], []
    for i in range(0, ro.shape[0], chunk):
        o = render_rays(model, ro[i:i + chunk], rd[i:i + chunk], near, far, **kw)
        rgbs.append(o["rgb"]); depths.append(o["depth"])
    return torch.cat(rgbs).reshape(H, W, 3), torch.cat(depths).reshape(H, W)
