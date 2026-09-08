# 3D Gaussian Splatting: a from-scratch training system with a pure-PyTorch reference rasteriser

A complete 3DGS implementation (Kerbl et al., SIGGRAPH 2023) built around the `gsplat` CUDA
rasteriser for speed, with **every other part written from scratch**: the Gaussian
representation (quaternion-scale covariance, spherical-harmonic colour), EWA projection,
adaptive density control (clone / split / prune / opacity reset with optimiser-state
surgery), the L1 + D-SSIM objective, the learning-rate schedule, evaluation (PSNR / SSIM /
LPIPS / FPS), standard `.ply` export, and a **readable pure-PyTorch reference rasteriser**
(depth-sorted per-pixel alpha compositing) that reproduces the maths of the paper, is used by
the unit tests and lets the whole pipeline run on a CPU at small resolution.

<p align="center">
<img src="results/scenes/lego/turntable.gif" width="40%"/>&nbsp;
<img src="results/scenes/ship/turntable.gif" width="40%"/>
</p>

## Results (Tesla T4, Colab)

<p align="center"><img src="results/gallery.png" width="100%"/></p>
<p align="center"><i>Turntable renders of the trained models (chair, drums, ficus, hotdog / lego, materials, mic, ship); full GIFs in <code>results/scenes/*/turntable.gif</code></i></p>

### NeRF-synthetic benchmark (30k iterations, 800x800, 200 test views per scene)

| scene | PSNR (ours) | PSNR (paper) | SSIM | LPIPS | #Gaussians | train [min] | render FPS |
|---|---|---|---|---|---|---|---|
| chair | 34.39 | 35.83 | 0.981 | 0.026 | 143k | 14 | 235 |
| drums | 25.89 | 26.15 | 0.947 | 0.061 | 155k | 14 | 229 |
| ficus | 29.79 | 34.87 | 0.968 | 0.040 | 79k | 12 | 312 |
| hotdog | 34.97 | 37.72 | 0.975 | 0.055 | 33k | 11 | 337 |
| lego | 31.28 | 35.78 | 0.960 | 0.052 | 110k | 14 | 263 |
| materials | 27.47 | 30.00 | 0.936 | 0.085 | 52k | 11 | 366 |
| mic | 33.93 | 35.36 | 0.988 | 0.015 | 105k | 13 | 256 |
| ship | 27.76 | 30.80 | 0.865 | 0.214 | 79k | 13 | 258 |
| **mean** | **30.68** | **33.31** | | | | | |

### Ablation (lego)

| variant | PSNR | SSIM | LPIPS | #Gaussians | train [min] | FPS |
|---|---|---|---|---|---|---|
| full (30k, densify to 15k, SH deg 3, grad thr 2e-4) | 31.45 | 0.960 | 0.051 | 110k | 13 | 271 |
| no adaptive density control (300k fixed random Gaussians) | 13.52 | 0.725 | 0.425 | 300k | 29 | 68 |
| SH degree 0 (view-independent colour) | 31.03 | 0.958 | 0.053 | 158k | 13 | 245 |
| densification gradient threshold x2 (4e-4) | 28.38 | 0.929 | 0.097 | 20k | 11 | 451 |
| densification gradient threshold /2 (1e-4) | 33.54 | 0.974 | 0.030 | 444k | 22 | 129 |
| densification stopped at 5k iterations | 31.22 | 0.958 | 0.056 | 84k | 13 | 293 |
| 7k iterations only | 28.63 | 0.932 | 0.094 | 36k | 3 | 369 |

### Same scene, same GPU: HashNeRF (previous project) vs 3D Gaussian Splatting (lego)

| method | PSNR | SSIM | train [min] | render FPS | note |
|---|---|---|---|---|---|
| HashNeRF (Instant-NGP style, pure PyTorch, previous project) | 24.36 | 0.883 | 29 | 0.4 | 400x400, 15000 iters, 25 test views |
| 3D Gaussian Splatting (this project) | 31.28 | 0.960 | 14 | 263.0 | 800x800, 30000 iters, 200 test views, 109k Gaussians |

Reading the numbers: with the paper's default densification threshold (2e-4) this implementation
ends up with far fewer Gaussians than the reference (lego: 110k vs. several hundred thousand) and
lands 2.6 dB below the paper on average. The ablation shows that this gap is a densification
setting, not a rendering or optimisation defect: halving the threshold takes lego to 33.5 dB with
444k Gaussians, close to the paper's 35.8, while removing density control altogether drops it to
13.5 dB. The HashNeRF comparison uses the same GPU and scene; the NeRF is trained at 400x400 for
15k iterations (its own budget), 3DGS at 800x800 for 30k, so the PSNR gap is indicative rather than
exact, but the 650x rendering-speed gap is the point of the comparison.

<p align="center"><img src="results/training_curves.png" width="90%"/></p>
<p align="center"><i>Gaussian count (left) and training PSNR (right) over the 30k iterations. The sawtooth in the
count is the periodic opacity reset followed by pruning; densification stops at 15k.</i></p>

## What is implemented

| component | file | notes |
|---|---|---|
| Gaussian representation | `gs/model.py` | means, log-scales, quaternions, opacity logits, SH coefficients (degree 0-3); covariance `R S S^T R^T` is PSD by construction |
| Adaptive density control | `gs/model.py` | screen-space gradient statistics; clone small / split large Gaussians (children sampled from the parent, scale / 1.6); prune transparent, oversized and split parents; periodic opacity reset; Adam moments are sliced / extended so the optimiser stays consistent |
| EWA projection + SH | `gs/render.py` | 3-D -> 2-D covariance through the projective Jacobian, 0.3 px low-pass, real SH evaluation with the reference ordering |
| Reference rasteriser | `gs/render.py` | pure PyTorch, differentiable, depth-sorted front-to-back compositing with early alpha cutoff; no tiles, meant for tests and small images |
| CUDA rasteriser | `gs/render.py` | `gsplat.rasterization` with `absgrad` for densification statistics; identical interface to the reference |
| Loss / metrics | `gs/losses.py` | L1 + 0.2 D-SSIM with an 11x11 Gaussian window; PSNR, SSIM; LPIPS (VGG) at evaluation |
| Training schedule | `gs/trainer.py` | 30k iterations, SH degree +1 every 1k, densify every 100 from 500 to 15k, opacity reset every 3k, exponential position LR scaled by scene extent |
| Data | `gs/data.py` | NeRF-synthetic json + RGBA; COLMAP captures via `tools/colmap_to_transforms.py` (poses + sparse points for initialisation) |
| Experiments | `run_benchmark.py`, `nerf_compare.py`, `make_report.py`, `run_all_colab.ipynb` | resumable benchmark + ablations, NeRF comparison, tables and figures |
| Tests | `tests/test_gs.py` | rotation/PSD, single-Gaussian rendering, depth ordering, gradient flow, SH, densification + optimiser consistency, SSIM |

## Run it

```bash
pip install torch gsplat lpips imageio scikit-image
python -m pytest -q tests                                                   # CPU, seconds
python train_scene.py --data data/nerf_synthetic/lego --out runs/lego       # GPU, ~10 min on a T4
python train_scene.py --data data/nerf_synthetic/lego --out runs/smoke --device cpu --downscale 10 --iters 120 --n_init 500   # CPU smoke run
python run_benchmark.py --data data/nerf_synthetic --out results --scenes all --ablations
python nerf_compare.py --data data/nerf_synthetic/lego --results results
python make_report.py results
```

`run_all_colab.ipynb` runs the whole set on a free Colab GPU with results saved to Google Drive.

## What I learned

1. **Density control is the algorithm.** Without it (300k fixed random Gaussians) the same
   optimiser reaches 13.5 dB; with it, 31.5 dB. The gradient threshold is the single most
   important knob: 4e-4 gives 20k Gaussians / 28.4 dB / 451 FPS, 1e-4 gives 444k / 33.5 dB /
   129 FPS. Quality, memory and frame rate are all set by this one number.
2. **Optimiser surgery matters.** Cloning / splitting / pruning changes the parameter
   tensors; Adam's first and second moments have to be sliced and extended in the same way
   or the new Gaussians inherit stale statistics and training destabilises.
3. **A reference implementation pays for itself.** When the first GPU runs produced washed-out
   renders, rendering the same Gaussians through the pure-PyTorch rasteriser and through gsplat
   (mean pixel difference 7e-5, identical gradients) ruled out the whole rendering path in ten
   minutes and pointed at the training schedule: I had rescaled the 30k learning-rate decay to
   the length of a short debugging run, which froze the positions after a few hundred steps.
   The paper's schedule is now used unchanged whatever the run length.
4. **Short runs must be prefixes of the real schedule, not compressed versions of it.** Scaling
   the opacity-reset interval down to 100 iterations made the first reset coincide with the
   large-Gaussian pruning pass and deleted 99.99% of the scene.

## References

Kerbl, Kopanas, Leimkuehler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) ·
Ye et al., *gsplat: An Open-Source Library for Gaussian Splatting* (2024) · Zwicker et al., *EWA Splatting* (2002)
