"""Training loop with the schedule of the paper (30k iterations):
  * L1 + 0.2 * D-SSIM loss, white background for Blender scenes
  * SH degree increased every 1000 iterations up to 3
  * densification every 100 its between 500 and 15000 (clone/split by screen-space gradient),
    pruning of near-transparent Gaussians, opacity reset every 3000 its
  * exponential learning-rate decay on the positions (scaled by the scene extent)
"""
import json, math, os, random, time
import numpy as np
import torch
from .model import GaussianModel
from .render import render, grad2d_norm
from .losses import rgb_loss, psnr, ssim
from .data import scene_extent


class Config:
    iters = 30000
    sh_degree = 3
    densify_from, densify_until, densify_every = 500, 15000, 100
    grad_thresh = 0.0002
    opacity_reset_every = 3000
    min_opacity = 0.005
    percent_dense = 0.01
    lr_means_init, lr_means_final = 1.6e-4, 1.6e-6
    lambda_dssim = 0.2
    white_bg = True
    max_gaussians = 6_000_000
    eval_every = 0
    densify = True          # ablation switch
    sh_max = 3              # ablation: 0 = view-independent colour
    log_every = 100
    # ---- LiDAR depth supervision (gs/lidar.py); 0 disables it and the trainer never renders depth
    lambda_depth = 0.0
    depth_from = 0          # iteration from which the depth term is active
    depth_acc_min = 0.5     # only pixels the model renders as at least this opaque are supervised
    depth_scale = 1.0       # median LiDAR depth of the scene (set by the caller) so lambda_depth is unit-free

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)

    def todict(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith("_") and not callable(getattr(self, k))}


def lr_schedule(it, cfg, extent):
    """Always the 30k-iteration schedule of the paper, regardless of how long this run is."""
    t = min(it / 30000, 1.0)
    return extent * math.exp(math.log(cfg.lr_means_init) * (1 - t) + math.log(cfg.lr_means_final) * t)


def train(train_cams, test_cams, init_xyz, init_rgb, out, cfg=None, device="cuda", backend="auto", seed=0):
    cfg = cfg or Config()
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); random.seed(seed)
    extent = scene_extent(train_cams)
    model = GaussianModel(init_xyz, init_rgb, sh_degree=cfg.sh_max).to(device)
    for c in train_cams + test_cams:
        c.to(device)
    use_depth = cfg.lambda_depth > 0 and any(c.depth is not None for c in train_cams)
    depth_log = []
    opt = torch.optim.Adam(model.param_groups(lr_schedule(0, cfg, extent)), eps=1e-15)
    bg = (1.0, 1.0, 1.0) if cfg.white_bg else (0.0, 0.0, 0.0)
    log, t0 = [], time.time()
    order = []
    for it in range(1, cfg.iters + 1):
        if not order:
            order = list(range(len(train_cams))); random.shuffle(order)
        cam = train_cams[order.pop()]
        if it % 1000 == 0 and model.active_sh_degree < cfg.sh_max:
            model.active_sh_degree += 1
        for g in opt.param_groups:
            if g["name"] == "means":
                g["lr"] = lr_schedule(it, cfg, extent)
        want_depth = use_depth and cam.depth is not None and it >= cfg.depth_from
        img, info = render(model, cam, backend=backend, bg=bg, return_depth=want_depth)
        loss, l1 = rgb_loss(img, cam.image, cfg.lambda_dssim)
        if want_depth:
            from .lidar import depth_loss
            ld, n_sup = depth_loss(info["depth"], info["acc"], cam.depth, cfg.depth_acc_min, cfg.depth_scale)
            loss = loss + cfg.lambda_depth * ld
            if it % cfg.log_every == 0:
                depth_log.append(dict(it=it, depth_l1_scaled=float(ld), n_supervised=n_sup))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # ---- density control
        if cfg.densify and it < cfg.densify_until:
            model.accumulate_stats(grad2d_norm(info, cam.W, cam.H), info["radii"], info["visible"])
            if it > cfg.densify_from and it % cfg.densify_every == 0 and model.n < cfg.max_gaussians:
                stats = model.densify_and_prune(opt, cfg.grad_thresh, extent, cfg.min_opacity,
                                                max_screen=20 if it > cfg.opacity_reset_every else None,
                                                percent_dense=cfg.percent_dense)
                if it % (cfg.densify_every * 10) == 0:
                    print(f"  [densify @{it}] {stats}", flush=True)
            if it % cfg.opacity_reset_every == 0:
                model.reset_opacity(opt)
        opt.step()
        if it % cfg.log_every == 0 or it == 1:
            rec = dict(it=it, loss=float(loss), l1=float(l1), psnr=psnr(img.detach(), cam.image),
                       n=model.n, time_min=(time.time() - t0) / 60)
            log.append(rec)
            if it % (cfg.log_every * 10) == 0:
                print(f"it {it:6d} loss {rec['loss']:.4f} psnr {rec['psnr']:.2f} N {model.n:7d} {rec['time_min']:.1f} min", flush=True)
        if cfg.eval_every and it % cfg.eval_every == 0:
            m = evaluate(model, test_cams[:5], backend=backend, bg=bg)
            print(f"  VAL @{it}: {m}", flush=True)
    json.dump(log, open(f"{out}/train_log.json", "w"))
    if depth_log:
        json.dump(depth_log, open(f"{out}/depth_log.json", "w"))
    return model, log, dict(train_time_min=(time.time() - t0) / 60, extent=extent, depth_supervised=use_depth)


@torch.no_grad()
def evaluate(model, cams, backend="auto", bg=(1.0, 1.0, 1.0), lpips_fn=None, save_dir=None):
    model.eval()
    ps, ss, lp, times = [], [], [], []
    for i, cam in enumerate(cams):
        if model.means.is_cuda:
            torch.cuda.synchronize()
        t = time.time()
        img, _ = render(model, cam, backend=backend, bg=bg)
        if model.means.is_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - t)
        img = img.clamp(0, 1)
        ps.append(psnr(img, cam.image)); ss.append(float(ssim(img, cam.image)))
        if lpips_fn is not None:
            lp.append(float(lpips_fn(img.permute(2, 0, 1)[None] * 2 - 1, cam.image.permute(2, 0, 1)[None] * 2 - 1)))
        if save_dir is not None and i < 8:
            from PIL import Image
            os.makedirs(save_dir, exist_ok=True)
            both = torch.cat([cam.image, img], 1).cpu().numpy()
            Image.fromarray((both * 255).astype(np.uint8)).save(f"{save_dir}/{i:03d}_gt_pred.png")
    model.train()
    out = dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), fps=float(1 / np.mean(times[1:] if len(times) > 1 else times)),
               n_gaussians=model.n, mean_opacity=float(model.opacities().mean()), mean_scale=float(model.scales().mean()),
               frac_opacity_gt_0p5=float((model.opacities() > 0.5).float().mean()))
    if lp:
        out["lpips"] = float(np.mean(lp))
    return out
