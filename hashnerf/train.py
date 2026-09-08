#!/usr/bin/env python3
"""Train a NeRF variant on the synthetic Lego scene.

    python train.py --model hash --iters 4000 --out runs/hash
    python train.py --model tiny --iters 4000 --out runs/tiny
"""
import argparse, json, os, time
import numpy as np
import torch
from nerf.data import load_dataset, get_rays
from nerf.model import build_model
from nerf.render import render_rays, render_image, OccupancyGrid
from nerf.metrics import psnr, ssim, lpips_fn

torch.manual_seed(0)
np.random.seed(0)


def evaluate(model, data, idx, args, grid, lp=None):
    model.eval()
    res = []
    for i in idx:
        ro, rd = get_rays(data["H"], data["W"], data["focal"], data["poses"][i])
        rgb, _ = render_image(model, ro, rd, args.near, args.far, n_samples=args.n_samples,
                              n_importance=args.n_importance, grid=grid)
        gt = data["images"][i]
        r = dict(psnr=psnr(rgb, gt), ssim=ssim(rgb, gt))
        if lp is not None:
            r["lpips"] = lpips_fn(lp, rgb, gt)
        res.append(r)
    model.train()
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--model", default="hash", choices=["tiny", "nerf", "hash"])
    ap.add_argument("--res", type=int, default=200)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--n_importance", type=int, default=0)
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=6.0)
    ap.add_argument("--no_grid", action="store_true")
    ap.add_argument("--grid_res", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--time_budget_min", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log2_T", type=int, default=19)
    ap.add_argument("--n_levels", type=int, default=16)
    ap.add_argument("--no_dirs", action="store_true")
    ap.add_argument("--data_json", default="transforms_train.json")
    ap.add_argument("--test_json", default=None, help="use this split as validation (e.g. transforms_test.json)")
    ap.add_argument("--n_test", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    torch.set_num_threads(max(1, os.cpu_count()))
    torch.set_default_device(args.device)

    data = load_dataset(args.data, split_json=args.data_json, res=args.res)
    data["images"], data["poses"] = data["images"].to(args.device), data["poses"].to(args.device)
    if args.test_json:                                   # standard split: train on all, validate on test views
        data["train_idx"] = np.arange(len(data["images"]))
        val = load_dataset(args.data, split_json=args.test_json, res=args.res, n_val=1)
        val["images"], val["poses"] = val["images"].to(args.device), val["poses"].to(args.device)
        val["val_idx"] = np.arange(min(args.n_test, len(val["images"])))
    else:
        val = data
    model = (build_model("hash", log2_T=args.log2_T, n_levels=args.n_levels, use_dirs=not args.no_dirs)
             if args.model == "hash" else build_model(args.model)).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params/1e6:.2f}M  train={len(data['train_idx'])} val={len(data['val_idx'])}")
    lr = args.lr or (1e-2 if args.model == "hash" else 5e-4)
    if args.model == "hash":
        opt = torch.optim.Adam([{"params": [model.enc.table], "weight_decay": 0},
                                {"params": list(model.sigma_net.parameters()) + list(model.color_net.parameters()),
                                 "weight_decay": 1e-6}], lr=lr, betas=(0.9, 0.99), eps=1e-15)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 ** (it / args.iters))
    grid = None if (args.no_grid or args.model != "hash") else OccupancyGrid(res=args.grid_res)

    # precompute all training rays
    ro_all, rd_all, rgb_all = [], [], []
    for i in data["train_idx"]:
        ro, rd = get_rays(data["H"], data["W"], data["focal"], data["poses"][i])
        ro_all.append(ro.reshape(-1, 3)); rd_all.append(rd.reshape(-1, 3)); rgb_all.append(data["images"][i].reshape(-1, 3))
    ro_all, rd_all, rgb_all = torch.cat(ro_all), torch.cat(rd_all), torch.cat(rgb_all)
    N = ro_all.shape[0]

    log, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        if grid is not None and (it % 16 == 0 or it == 1) and it > 64:
            grid.update(model)
        sel = torch.randint(0, N, (args.batch,))
        out = render_rays(model, ro_all[sel], rd_all[sel], args.near, args.far, n_samples=args.n_samples,
                          n_importance=args.n_importance, perturb=True, grid=grid)
        loss = ((out["rgb"] - rgb_all[sel]) ** 2).mean()
        if "rgb_coarse" in out:
            loss = loss + ((out["rgb_coarse"] - rgb_all[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 50 == 0:
            el = time.time() - t0
            frac = out["n_eval"] / out["n_total"]
            print(f"it {it:5d} loss {loss.item():.4f} psnr {-10*np.log10(loss.item()):.2f} "
                  f"eval_frac {frac:.2f} {el/it:.2f}s/it  elapsed {el/60:.1f}min", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            m = evaluate(model, val, val["val_idx"][:3], args, grid)
            m.update(it=it, time_min=(time.time() - t0) / 60)
            log.append(m); print("  VAL", m, flush=True)
            json.dump(log, open(f"{args.out}/log.json", "w"), indent=1)
            torch.save(model.state_dict(), f"{args.out}/model.pt")
        if args.time_budget_min and (time.time() - t0) / 60 > args.time_budget_min:
            print("time budget reached at it", it); break

    # final evaluation with LPIPS
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(args.device)
    except Exception as e:
        print("lpips unavailable:", e)
    final = evaluate(model, val, val["val_idx"], args, grid, lp)
    final.update(it=it, time_min=(time.time() - t0) / 60, params=n_params, model=args.model, res=args.res)
    print("FINAL", final)
    json.dump(final, open(f"{args.out}/final.json", "w"), indent=1)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    if grid is not None:
        torch.save(grid.occupied, f"{args.out}/grid.pt")


if __name__ == "__main__":
    main()
