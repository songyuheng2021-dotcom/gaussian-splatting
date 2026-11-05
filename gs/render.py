"""Rendering front-ends.

  * `render_gsplat`   : the CUDA tile rasteriser from gsplat (used for all real training)
  * `render_reference`: a from-scratch, pure-PyTorch, fully differentiable rasteriser that
                        implements exactly the maths of the paper (EWA projection of the 3-D
                        covariance, per-pixel depth-sorted alpha compositing). It has no tiles
                        or culling so it is only practical for small images; it is used for
                        the unit tests, for CPU smoke runs and as a readable reference.
Both return (image (H,W,3), info) where info has 'means2d' (N,2, grad retained),
'radii' (N,), 'visible' (N,) so that the trainer is backend-agnostic.
"""
import math
import torch
from .model import quat_to_rotmat

# real SH basis constants (degree <= 3), same ordering as the reference implementation
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def eval_sh(sh, dirs, degree):
    """sh (N,K,3), dirs (N,3) unit -> rgb (N,3) (clamped to >= 0 after +0.5 like the paper)."""
    x, y, z = dirs.unbind(-1)
    res = 0.28209479177387814 * sh[:, 0]
    if degree >= 1:
        res = res - C1 * y[:, None] * sh[:, 1] + C1 * z[:, None] * sh[:, 2] - C1 * x[:, None] * sh[:, 3]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        res = (res + C2[0] * (xy)[:, None] * sh[:, 4] + C2[1] * (yz)[:, None] * sh[:, 5]
               + C2[2] * (2 * zz - xx - yy)[:, None] * sh[:, 6] + C2[3] * (xz)[:, None] * sh[:, 7]
               + C2[4] * (xx - yy)[:, None] * sh[:, 8])
    if degree >= 3:
        res = (res + C3[0] * (y * (3 * xx - yy))[:, None] * sh[:, 9] + C3[1] * (xy * z)[:, None] * sh[:, 10]
               + C3[2] * (y * (4 * zz - xx - yy))[:, None] * sh[:, 11]
               + C3[3] * (z * (2 * zz - 3 * xx - 3 * yy))[:, None] * sh[:, 12]
               + C3[4] * (x * (4 * zz - xx - yy))[:, None] * sh[:, 13]
               + C3[5] * (z * (xx - yy))[:, None] * sh[:, 14] + C3[6] * (x * (xx - 3 * yy))[:, None] * sh[:, 15])
    return (res + 0.5).clamp(min=0)


def project(model, w2c, K):
    """EWA projection: returns means2d (N,2), cov2d (N,2,2), depth (N,), colours (N,3)."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    p_cam = model.means @ R.T + t
    x, y, z = p_cam.unbind(-1)
    z = z.clamp(min=1e-3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    means2d = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero, -fx * x / z ** 2], -1),
                     torch.stack([zero, fy / z, -fy * y / z ** 2], -1)], 1)          # (N,2,3)
    cov3d = model.covariances()
    cov2d = J @ R @ cov3d @ R.T @ J.transpose(1, 2)
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device)                      # low-pass filter
    cam_center = -R.T @ t
    dirs = model.means - cam_center
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = eval_sh(model.sh, dirs, model.active_sh_degree)
    return means2d, cov2d, z, colors


def render_reference(model, cam, bg=(1.0, 1.0, 1.0), chunk_px=4096, return_depth=False):
    """With return_depth=True, info also carries 'depth' (H,W): the expected depth
    sum_i w_i z_i / sum_i w_i along each pixel (z = camera-frame depth of the Gaussian
    centre), and 'acc' (H,W): the accumulated opacity sum_i w_i. Pixels with small acc
    are background and their depth is meaningless; callers mask on acc."""
    w2c, K, H, W = cam.w2c, cam.K, cam.H, cam.W
    means2d, cov2d, depth, colors = project(model, w2c, K)
    if means2d.requires_grad:
        means2d.retain_grad()
    opac = model.opacities()
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    inv = torch.stack([torch.stack([cov2d[:, 1, 1], -cov2d[:, 0, 1]], -1),
                       torch.stack([-cov2d[:, 1, 0], cov2d[:, 0, 0]], -1)], 1) / det[:, None, None]
    lam = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lam_max = lam + torch.sqrt((lam ** 2 - det).clamp(min=0.1))
    radii = torch.ceil(3 * torch.sqrt(lam_max))
    in_front = depth > 0.2
    on_screen = (means2d[:, 0] + radii > 0) & (means2d[:, 0] - radii < W) & (means2d[:, 1] + radii > 0) & (means2d[:, 1] - radii < H)
    visible = in_front & on_screen
    order = torch.argsort(depth)                                   # front to back
    order = order[visible[order]]
    m2, iv, op, col, dz = means2d[order], inv[order], opac[order], colors[order], depth[order]
    ys, xs = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    px = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5
    bg_t = torch.tensor(bg, device=depth.device)
    out, out_d, out_a = [], [], []
    for i in range(0, px.shape[0], chunk_px):
        d = px[i:i + chunk_px, None, :] - m2[None]                 # (P,N,2)
        power = -0.5 * (iv[None, :, 0, 0] * d[..., 0] ** 2 + iv[None, :, 1, 1] * d[..., 1] ** 2
                        + 2 * iv[None, :, 0, 1] * d[..., 0] * d[..., 1])
        alpha = (op[None] * torch.exp(power)).clamp(max=0.99)
        alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
        T = torch.cumprod(1 - alpha, dim=1)
        T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], 1)  # transmittance before each Gaussian
        w = alpha * T
        acc = w.sum(1, keepdim=True)
        rgb = w @ col + (1 - acc) * bg_t
        out.append(rgb)
        if return_depth:
            out_d.append((w @ dz[:, None]) / acc.clamp(min=1e-6))
            out_a.append(acc)
    img = torch.cat(out).reshape(H, W, 3)
    info = dict(means2d=means2d, radii=radii, visible=visible, n_visible=int(visible.sum()))
    if return_depth:
        info["depth"] = torch.cat(out_d).reshape(H, W)
        info["acc"] = torch.cat(out_a).reshape(H, W)
    return img, info


def render_gsplat(model, cam, bg=(1.0, 1.0, 1.0), absgrad=False, return_depth=False):
    """absgrad=True (AbsGS) needs a ~4x higher densification threshold (gsplat docs: 8e-4); we keep
    the paper's averaged gradient with 2e-4 by default.
    return_depth=True uses render_mode 'RGB+ED' so the 4th channel is gsplat's expected depth
    (alpha-normalised, same definition as the reference rasteriser); 'acc' comes from alphas."""
    from gsplat import rasterization
    dev = model.means.device
    bg_t = torch.tensor(bg, device=dev)[None]
    renders, alphas, info = rasterization(
        means=model.means, quats=model.quats, scales=model.scales(), opacities=model.opacities(),
        colors=model.sh, viewmats=cam.w2c[None].to(dev), Ks=cam.K[None].to(dev), width=cam.W, height=cam.H,
        sh_degree=model.active_sh_degree, backgrounds=bg_t, packed=False, absgrad=absgrad,
        near_plane=0.01, far_plane=1e10, render_mode="RGB+ED" if return_depth else "RGB",
        rasterize_mode="classic")
    if info["means2d"].requires_grad:
        info["means2d"].retain_grad()
    radii = info["radii"][0]                                   # (N,) in gsplat<=1.4, (N,2) in >=1.5
    visible = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
    radii = radii.max(-1).values if radii.dim() == 2 else radii
    out = dict(means2d=info["means2d"], radii=radii, visible=visible, n_visible=int(visible.sum()), absgrad=absgrad)
    if return_depth:
        out["depth"] = renders[0, ..., 3]
        out["acc"] = alphas[0, ..., 0]
        return renders[0, ..., :3], out
    return renders[0], out


def grad2d_norm(info, W, H):
    """Screen-space positional gradient magnitude per Gaussian, in NDC-like units."""
    m = info["means2d"]
    g = m.absgrad if info.get("absgrad") and hasattr(m, "absgrad") and m.absgrad is not None else m.grad
    if g is None:
        return torch.zeros(info["radii"].shape[0], device=info["radii"].device)
    if g.dim() == 3:
        g = g[0]
    g = g.clone()
    g[:, 0] *= 0.5 * W; g[:, 1] *= 0.5 * H          # both backends give pixel-space gradients; this converts to NDC (paper threshold 2e-4)
    return g.norm(dim=-1)


def render(model, cam, backend="auto", **kw):
    if backend == "auto":
        backend = "gsplat" if model.means.is_cuda else "reference"
    return (render_gsplat if backend == "gsplat" else render_reference)(model, cam, **kw)
