"""The Gaussian scene representation and its optimiser, with adaptive density control
(clone / split / prune / opacity reset) as in Kerbl et al. 2023."""
import math
import numpy as np
import torch
from torch import nn

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / SH_C0


def sh0_to_rgb(sh):
    return sh * SH_C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def quat_to_rotmat(q):
    """q: (N,4) (w,x,y,z), not necessarily normalised -> (N,3,3)."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


class GaussianModel(nn.Module):
    """Parameters (all leaf tensors so the optimiser can be rebuilt after densification):
        means (N,3), log_scales (N,3), quats (N,4), opacity_logits (N,), sh (N,K,3)
    Covariance = R(q) diag(exp(s))^2 R(q)^T, guaranteed positive semi-definite by construction.
    Colour = spherical harmonics of the viewing direction, degree grown during training."""

    def __init__(self, xyz, rgb, sh_degree=3, init_scale=None):
        super().__init__()
        n = xyz.shape[0]
        self.sh_degree, self.active_sh_degree = sh_degree, 0
        K = (sh_degree + 1) ** 2
        if init_scale is None:                       # mean distance to 3 nearest neighbours
            d = torch.cdist(xyz[:min(n, 20000)], xyz[:min(n, 20000)])
            knn = d.topk(4, largest=False).values[:, 1:].mean(1).mean()
            init_scale = float(knn)
        self.means = nn.Parameter(xyz.clone())
        self.log_scales = nn.Parameter(torch.full((n, 3), math.log(max(init_scale, 1e-4))))
        q = torch.zeros(n, 4); q[:, 0] = 1
        self.quats = nn.Parameter(q)
        self.opacity_logits = nn.Parameter(inverse_sigmoid(torch.full((n,), 0.1)))
        sh = torch.zeros(n, K, 3); sh[:, 0] = rgb_to_sh0(rgb)
        self.sh = nn.Parameter(sh)
        # densification statistics
        self.register_buffer("grad_accum", torch.zeros(n))
        self.register_buffer("grad_count", torch.zeros(n))
        self.register_buffer("max_radii", torch.zeros(n))

    # ---- derived quantities
    @property
    def n(self):
        return self.means.shape[0]

    def scales(self):
        return torch.exp(self.log_scales)

    def opacities(self):
        return torch.sigmoid(self.opacity_logits)

    def covariances(self):
        R = quat_to_rotmat(self.quats)
        S = torch.diag_embed(self.scales())
        M = R @ S
        return M @ M.transpose(1, 2)

    def param_groups(self, lr_means):
        return [dict(params=[self.means], lr=lr_means, name="means"),
                dict(params=[self.log_scales], lr=5e-3, name="log_scales"),
                dict(params=[self.quats], lr=1e-3, name="quats"),
                dict(params=[self.opacity_logits], lr=5e-2, name="opacity_logits"),
                dict(params=[self.sh], lr=2.5e-3, name="sh")]

    # ---- adaptive density control
    @torch.no_grad()
    def accumulate_stats(self, grad2d_norm, radii, visible):
        self.grad_accum[visible] += grad2d_norm[visible]
        self.grad_count[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], radii[visible].float())

    @torch.no_grad()
    def _replace(self, optimizer, keep_mask=None, extra=None):
        """Rebuild parameters (and Adam state) keeping rows in keep_mask and appending `extra`."""
        names = ["means", "log_scales", "quats", "opacity_logits", "sh"]
        for group in optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = optimizer.state.pop(old, None)
            new = getattr(self, name).data
            if keep_mask is not None:
                new = new[keep_mask]
            if extra is not None:
                new = torch.cat([new, extra[name]], 0)
            p = nn.Parameter(new.contiguous())
            if state is not None:
                for k in ("exp_avg", "exp_avg_sq"):
                    s = state[k]
                    if keep_mask is not None:
                        s = s[keep_mask]
                    if extra is not None:
                        s = torch.cat([s, torch.zeros_like(extra[name])], 0)
                    state[k] = s
                optimizer.state[p] = state
            group["params"][0] = p
            setattr(self, name, p)
        n = self.n
        dev = self.means.device
        self.grad_accum = torch.zeros(n, device=dev)
        self.grad_count = torch.zeros(n, device=dev)
        self.max_radii = torch.zeros(n, device=dev)

    @torch.no_grad()
    def densify_and_prune(self, optimizer, grad_thresh, scene_extent, min_opacity=0.005,
                          max_screen=None, percent_dense=0.01, split_n=2):
        grads = self.grad_accum / self.grad_count.clamp(min=1)
        grads[self.grad_count == 0] = 0
        big = grads >= grad_thresh
        small_scale = self.scales().max(1).values <= percent_dense * scene_extent
        clone_mask = big & small_scale
        split_mask = big & ~small_scale
        n_clone, n_split = int(clone_mask.sum()), int(split_mask.sum())
        extra = {}
        names = ["means", "log_scales", "quats", "opacity_logits", "sh"]
        # clone: copy small under-reconstructed Gaussians (moved along the gradient is optional; we copy in place)
        for k in names:
            extra[k] = getattr(self, k).data[clone_mask]
        # split: replace large Gaussians by `split_n` smaller ones sampled from the original
        if n_split > 0:
            R = quat_to_rotmat(self.quats[split_mask]).repeat(split_n, 1, 1)
            s = self.scales()[split_mask].repeat(split_n, 1)
            samples = torch.randn_like(s) * s
            new_means = (R @ samples[..., None])[..., 0] + self.means[split_mask].repeat(split_n, 1)
            new_scales = torch.log(s / (0.8 * split_n))
            extra["means"] = torch.cat([extra["means"], new_means])
            extra["log_scales"] = torch.cat([extra["log_scales"], new_scales])
            for k in ("quats", "opacity_logits", "sh"):
                extra[k] = torch.cat([extra[k], getattr(self, k).data[split_mask].repeat(split_n, *([1] * (self.state_ndim(k) - 1)))])
        # prune: transparent, or huge on screen / in world, and the originals that were split
        prune = (self.opacities() < min_opacity) | split_mask
        if max_screen is not None:
            prune |= self.max_radii > max_screen
        prune |= self.scales().max(1).values > 0.1 * scene_extent
        keep = ~prune
        self._replace(optimizer, keep_mask=keep, extra=extra)
        return dict(cloned=n_clone, split=n_split, pruned=int(prune.sum()), n=self.n)

    def state_ndim(self, k):
        return getattr(self, k).data.dim()

    @torch.no_grad()
    def reset_opacity(self, optimizer, value=0.01):
        new = inverse_sigmoid(torch.full_like(self.opacity_logits, value))
        self.opacity_logits.data.copy_(torch.minimum(self.opacity_logits.data, new))
        for group in optimizer.param_groups:
            if group["name"] == "opacity_logits":
                st = optimizer.state.get(group["params"][0])
                if st:
                    st["exp_avg"].zero_(); st["exp_avg_sq"].zero_()

    # ---- I/O in the standard 3DGS .ply layout (loadable by common viewers)
    @torch.no_grad()
    def save_ply(self, path):
        import numpy as np
        n = self.n
        sh = self.sh.detach().cpu().numpy()
        f_dc = sh[:, 0]                                   # (N,3)
        f_rest = sh[:, 1:].transpose(0, 2, 1).reshape(n, -1)  # (N, 3*(K-1)) channel-major like the reference
        cols = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
        cols += [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])]
        cols += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                 ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
        arr = np.empty(n, dtype=cols)
        xyz = self.means.detach().cpu().numpy()
        arr["x"], arr["y"], arr["z"] = xyz.T
        arr["nx"] = arr["ny"] = arr["nz"] = 0
        arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc.T
        for i in range(f_rest.shape[1]):
            arr[f"f_rest_{i}"] = f_rest[:, i]
        arr["opacity"] = self.opacity_logits.detach().cpu().numpy()
        ls = self.log_scales.detach().cpu().numpy(); arr["scale_0"], arr["scale_1"], arr["scale_2"] = ls.T
        q = self.quats.detach().cpu().numpy(); arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = q.T
        header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        header += "".join(f"property float {c[0]}\n" for c in cols) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii")); f.write(arr.tobytes())
