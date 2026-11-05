"""Generate run_geometry_colab.ipynb: ETH3D download, baseline vs LiDAR-supervised 3DGS, geometry acceptance tables."""
import json

cells = []


def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))


md("""# Geometric acceptance of Gaussian splats with a laser reference (Colab GPU)

Branch `aerial-geometry` of the repo. Runtime → GPU (A100 or L4 preferred, T4 works at `--downscale 8`).
Everything lands in `MyDrive/gs_geometry/` and finished runs are skipped on re-run.

Scenes: ETH3D high-res multi-view **courtyard** (38 images) and **terrace** (23 images), each with a Faro laser
scan as the metric reference. Experiments per scene: photometric baseline, LiDAR depth supervision at three
weights, 3 seeds for the two main configs.""")

code("""!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
import os
RES = '/content/drive/MyDrive/gs_geometry'
os.makedirs(RES, exist_ok=True)""")

code("""# 1) code
%cd /content
!rm -rf gaussian-splatting && git clone -q -b aerial-geometry https://github.com/songyuheng2021-dotcom/gaussian-splatting.git
%cd /content/gaussian-splatting
!git log --oneline | head -3""")

code("""# 2) dependencies (gsplat builds its CUDA kernels on first use: 3-8 min once per session)
!pip -q install ninja gsplat lpips imageio scikit-image open3d matplotlib
!apt-get -qq install -y p7zip-full > /dev/null
import torch, gsplat, time
print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())
t = time.time()
from gsplat import rasterization
means = torch.randn(10, 3, device='cuda'); quats = torch.randn(10, 4, device='cuda'); scales = torch.rand(10, 3, device='cuda')
K = torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]], device='cuda')
rasterization(means, quats, scales, torch.rand(10, device='cuda'), torch.rand(10, 3, device='cuda'),
              torch.eye(4, device='cuda')[None], K, 64, 64, render_mode='RGB+ED')
print('kernels ready in %.0f s' % (time.time() - t))
!python -m pytest -q tests""")

code("""# 3) ETH3D scenes (cached on Drive; ~0.7 GB courtyard, ~0.5 GB terrace)
import os
SCENES = ['courtyard', 'terrace']
DOWNSCALE = 4          # 6048x4032 -> 1512x1008; use 8 on a T4
os.makedirs(f'{RES}/eth3d_raw', exist_ok=True)
for s in SCENES:
    for part in ('dslr_undistorted', 'dslr_scan_eval'):
        f = f'{RES}/eth3d_raw/{s}_{part}.7z'
        if not os.path.exists(f):
            !wget -q -O {f} https://www.eth3d.net/data/{s}_{part}.7z
    if not os.path.exists(f'data/eth3d/{s}/gt_scan.ply'):
        !mkdir -p /content/raw/{s} && 7z x -y -o/content/raw/{s} {RES}/eth3d_raw/{s}_dslr_undistorted.7z > /dev/null && 7z x -y -o/content/raw/{s} {RES}/eth3d_raw/{s}_dslr_scan_eval.7z > /dev/null
        !python tools/prepare_eth3d.py --raw /content/raw/{s} --out data/eth3d/{s} --downscale {DOWNSCALE} --gt_voxel 0.02
    !cat data/eth3d/{s}/scene_info.json""")

code("""# 4) smoke: 1000 iterations on courtyard, geometry evaluation included (~2 min)
!python run_geometry.py --data data/eth3d/courtyard --out /tmp/smoke --iters 1000 --max_test 3 --no_lpips --lambda_depth 0.5""")

code("""# 5) the experiment grid (resumable). Edit ITERS / SEEDS to trade time for completeness.
import os, subprocess, json
ITERS = 30000
GRID = [  # (name, extra args)
    ('base',  '--lambda_depth 0'),
    ('d0p1',  '--lambda_depth 0.1'),
    ('d0p5',  '--lambda_depth 0.5'),
    ('d2',    '--lambda_depth 2.0'),
]
SEEDS = {'base': [0, 1, 2], 'd0p5': [0, 1, 2]}   # others: seed 0 only
for scene in SCENES:
    for name, extra in GRID:
        for seed in SEEDS.get(name, [0]):
            out = f'{RES}/{scene}_{name}_s{seed}'
            if os.path.exists(f'{out}/metrics.json'):
                print('skip', out); continue
            cmd = f'python run_geometry.py --data data/eth3d/{scene} --out {out} --iters {ITERS} --seed {seed} {extra} --cell 1.0 --tolerance 0.05'
            print(cmd); subprocess.run(cmd, shell=True, check=False)""")

code("""# 6) summary table (markdown) -> Drive
import glob, json, os
rows = []
for f in sorted(glob.glob(f'{RES}/*/metrics.json')):
    m = json.load(open(f)); g = m['geometry']['all']; h = m['geometry'].get('heldout_blocks', {}); a = m['acceptance']
    rows.append(dict(run=os.path.basename(os.path.dirname(f)), psnr=m['image']['psnr'], ssim=m['image']['ssim'], lpips=m['image'].get('lpips', float('nan')),
                     n=m['image']['n_gaussians'], acc_mean=g['acc_mean'], acc_med=g['acc_median'], comp_mean=g['comp_mean'],
                     f05=g['fscore@0.05'], f10=g['fscore@0.1'], p2pl_med=g.get('p2plane_median', float('nan')),
                     heldout_acc=h.get('acc_mean', float('nan')), pass_frac=a['pass_fraction'], train_min=m['train']['train_time_min']))
hdr = '| run | PSNR | SSIM | LPIPS | #G | acc mean [m] | acc median [m] | comp mean [m] | F@5cm | F@10cm | p2plane med [m] | held-out acc [m] | cells pass @5cm | train [min] |'
lines = [hdr, '|' + '---|' * 14]
for r in rows:
    lines.append(f"| {r['run']} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | {r['n']//1000}k | {r['acc_mean']:.3f} | {r['acc_med']:.3f} | {r['comp_mean']:.3f} | {r['f05']:.3f} | {r['f10']:.3f} | {r['p2pl_med']:.3f} | {r['heldout_acc']:.3f} | {r['pass_frac']:.2f} | {r['train_min'] or 0:.0f} |")
open(f'{RES}/summary.md', 'w').write('\\n'.join(lines) + '\\n')
print('\\n'.join(lines))""")

code("""# 7) share bundle (metrics, heatmaps, logs, test renders; no .ply)
!cd {RES} && zip -q -r /content/drive/MyDrive/gs_geometry_share.zip . -x '*.ply' 'eth3d_raw/*'
!ls -la /content/drive/MyDrive/gs_geometry_share.zip""")

nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3"),
          accelerator="GPU", colab=dict(provenance=[], gpuType="A100")), cells=cells)
json.dump(nb, open("run_geometry_colab.ipynb", "w"), indent=1)
print("notebook written")
