import json

cells = []


def md(s): cells.append(dict(cell_type="markdown", metadata={}, source=s))
def code(s): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s))


md("""# 3D Gaussian Splatting — full experiment run (Colab GPU)

**Before running:** Runtime → Change runtime type → **GPU (T4 is fine)**, and upload
`gaussian_splatting_package.zip` to the root of your Google Drive (`MyDrive/`).

Everything is saved under `MyDrive/gs_results/`, and every step is resumable: if Colab
disconnects, just run all cells again and finished runs are skipped.

Total time on a T4: ~2 h for the 8 scenes, ~1.5 h for the ablations, ~30 min for the NeRF comparison.
Run the cells in order; you can leave the tab alone in between.""")

code("""!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/gs_results', exist_ok=True)""")

code("""# 1) unpack the package next to the results dir on Drive (code + bundled lego data)
%cd /content
!rm -rf gs && mkdir gs && unzip -q /content/drive/MyDrive/gaussian_splatting_package.zip -d gs
%cd /content/gs/gaussian_splatting
!ln -sfn /content/drive/MyDrive/gs_results results
!ls""")

code("""# 2) dependencies (gsplat compiles its CUDA kernels on first import: 3-8 minutes, once per session)
!pip -q install ninja gsplat lpips imageio scikit-image gdown
import torch, gsplat, time
print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())
t = time.time()
from gsplat import rasterization
means = torch.randn(10, 3, device='cuda'); quats = torch.randn(10, 4, device='cuda'); scales = torch.rand(10, 3, device='cuda')
K = torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]], device='cuda')
rasterization(means, quats, scales, torch.rand(10, device='cuda'), torch.rand(10, 3, device='cuda'),
              torch.eye(4, device='cuda')[None], K, 64, 64)
print('kernels ready in %.0f s' % (time.time() - t))""")

code("""# 3) NeRF-synthetic dataset (8 scenes, ~700 MB). Cached on Drive after the first download.
import os, subprocess
zip_path = '/content/drive/MyDrive/nerf_synthetic.zip'
if not os.path.exists(zip_path):
    !gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG -O {zip_path}
if not os.path.exists('data/nerf_synthetic/lego/transforms_test.json'):
    !unzip -q {zip_path} -d data/ && ls data
    # some mirrors nest the folder one level deeper
    if os.path.exists('data/nerf_synthetic/nerf_synthetic'):
        !mv data/nerf_synthetic/nerf_synthetic/* data/nerf_synthetic/
!ls data/nerf_synthetic
# Fallback if the download fails: upload nerf_synthetic.zip (from https://www.matthewtancik.com/nerf) to MyDrive and re-run.""")

code("""# 4) sanity: unit tests + a 1-minute GPU run on lego
!python -m pytest -q tests
!python train_scene.py --data data/nerf_synthetic/lego --out /tmp/quick --iters 1000 --max_test 5 --no_turntable""")

code("""# 5) the 8-scene benchmark (resumable; ~10-15 min per scene on a T4)
!python run_benchmark.py --data data/nerf_synthetic --out results --scenes all""")

code("""# 6) ablations on lego (resumable)
!python run_benchmark.py --data data/nerf_synthetic --out results --scenes lego --ablations""")

code("""# 7) HashNeRF (previous project) on the same scene and GPU -> comparison table
!python nerf_compare.py --data data/nerf_synthetic/lego --results results --res 400 --iters 15000""")

code("""# 8) tables + figures, then a zip of everything small enough to share
!python make_report.py results
!cat results/report.md
!cd results && zip -q -r /content/drive/MyDrive/gs_results_share.zip . -x '*.ply' 'hashnerf_lego/model.pt' 'hashnerf_lego/grid.pt'
!ls -la /content/drive/MyDrive/gs_results_share.zip
print('DONE: download MyDrive/gs_results_share.zip and send it back')""")

md("""## Optional: your own object (phone capture)

Take 40-80 photos walking around an object (all sides, overlapping views, no motion blur),
put them in `MyDrive/my_capture/images/`, then run the cell below. COLMAP estimates the
camera poses; the converter writes `transforms.json` + `points3d.ply`; training is identical.""")

code("""#!apt-get -qq install -y colmap > /dev/null && pip -q install pycolmap
#!mkdir -p /content/cap && cp -r /content/drive/MyDrive/my_capture/images /content/cap/
#!colmap automatic_reconstructor --workspace_path /content/cap --image_path /content/cap/images --sparse 1 --dense 0 --quality medium
#!python tools/colmap_to_transforms.py --sparse /content/cap/sparse/0 --images /content/cap/images --out data/my_object
#!python train_scene.py --data data/my_object --out results/my_object --iters 15000 --downscale 2 --lpips""")

nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3"),
          accelerator="GPU", colab=dict(provenance=[], gpuType="T4")), cells=cells)
json.dump(nb, open("run_all_colab.ipynb", "w"), indent=1)
print("notebook written")
