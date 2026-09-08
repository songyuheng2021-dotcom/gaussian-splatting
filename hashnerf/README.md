# HashNeRF: an Instant-NGP-style neural radiance field from scratch in PyTorch

A from-scratch re-implementation of the core ideas of Instant-NGP (multiresolution hash
encoding, tiny MLPs, occupancy-grid empty-space skipping) next to the original NeRF and the
course TinyNeRF baseline, trained and evaluated on the synthetic Lego scene **on a 2-core CPU
with no GPU**. Everything (encodings, volume rendering, hierarchical sampling, occupancy
grid, metrics, mesh extraction) is implemented in this repo; the only dependencies are
PyTorch, NumPy, scikit-image and Open3D (mesh I/O).

<p align="center">
<img src="runs/hash/renders/turntable.gif" width="60%"/>
</p>
<p align="center"><i>HashNeRF 360-degree render (RGB | depth) after 79 min of CPU training</i></p>

## Results (10 held-out views, 200x200, white background, ~45-75 min wall-clock on 2 CPU cores)

| model | params | iters | train time [min] | PSNR | SSIM |
|---|---|---|---|---|---|
| TinyNeRF (course baseline: freq. enc. + 4x128 MLP) | 0.04 M | 3005 | 46 | 21.26 | 0.790 |
| NeRF (8x256 MLP, hierarchical sampling 64+64) | 0.60 M | 520 | 58 | 19.90 | 0.742 |
| HashNeRF (multires hash grid + 2 tiny MLPs + occupancy grid) | 16.79 M | 2416 | 79 | 25.27 | 0.891 |

Training time includes periodic validation. HashNeRF had already reached **25.0 dB after 42 min** (1500 iterations), i.e. within the baselines' budget it beats TinyNeRF by ~3.8 dB and the full NeRF MLP by ~5 dB; the full NeRF is limited to ~520 iterations in 45 min on CPU (5.2 s/iteration with 128 samples/ray), which is why it lands below the much smaller TinyNeRF here.


<p align="center">
<img src="runs/convergence.png" width="55%"/>
</p>

<p align="center">
<img src="runs/hash/renders/val_gt_pred_depth.png" width="60%"/>
</p>
<p align="center"><i>Held-out views: ground truth | HashNeRF prediction | depth</i></p>

<p align="center">
<img src="runs/hash/renders/mesh_preview.png" width="80%"/>
</p>
<p align="center"><i>Mesh extracted from the density field with marching cubes (largest connected component)</i></p>

## What is implemented

| component | file | notes |
|---|---|---|
| Multiresolution hash encoding | `nerf/encoding.py` | 16 levels, 2 features, 2^19 entries per level, base 16 -> 512 resolution; coarse levels indexed densely, fine levels through the Instant-NGP spatial hash; all levels vectorised into a single gather; trilinear interpolation |
| Spherical-harmonics direction encoding | `nerf/encoding.py` | degree-4 real SH (16 coefficients) |
| Frequency encoding | `nerf/encoding.py` | NeRF positional encoding for the baselines |
| Fields | `nerf/model.py` | `HashNeRF` (hash enc -> 64-wide density MLP -> 64-wide colour MLP), `NeRF` (8x256, skip connection, view-dependent branch), `TinyNeRF` (course baseline) |
| Occupancy grid | `nerf/render.py` | 64^3 binary grid updated every 16 iterations from the density field with EMA decay and one-cell dilation; samples in empty cells are not evaluated (90% of samples skipped after warm-up) |
| Volume rendering | `nerf/render.py` | stratified sampling, alpha compositing with white background, hierarchical (inverse-CDF) importance sampling for the NeRF baseline |
| Training | `train.py` | Adam with per-parameter-group weight decay (none on the hash tables), exponential LR decay, wall-clock budget, periodic validation |
| Evaluation / products | `render.py`, `nerf/metrics.py` | PSNR, SSIM; GT/prediction/depth comparison sheet, 360-degree turntable GIF, marching-cubes mesh export (`.ply`) |
| Tests | `tests/` | encoding shapes and interpolation continuity, SH orthonormality sanity check, compositing invariants |

## What I learned

1. **Where the speed comes from.** The hash grid moves most of the capacity from the MLP
   into a lookup table (16.8 M parameters, but each sample touches only 16x8 entries), so a
   64-wide MLP is enough. On CPU the per-iteration cost is dominated by the gather, not by
   matrix multiplies; vectorising all levels into a single gather gave ~1.6x over a
   per-level loop.
2. **The occupancy grid only helps once the density has separated.** With the default
   threshold the grid stayed 30% occupied because the freshly initialised density
   (softplus(-1) ~ 0.3) is above it everywhere; raising the threshold to the alpha-equivalent
   of one step brought the evaluated fraction to ~10% within the first 100 iterations.
3. **Compare at equal wall-clock, not equal iterations.** Per iteration the hash grid is slower on CPU
   than TinyNeRF (1.5 s vs 0.9 s), yet at 42 min it is 3.8 dB ahead and still climbing; the full NeRF MLP
   cannot even finish warm-up in that budget. Iteration counts alone would tell the opposite story.

## Run it

```bash
pip install torch numpy scikit-image open3d imageio pillow
python train.py --model hash --iters 3000 --time_budget_min 75 --out runs/hash
python train.py --model tiny --iters 20000 --time_budget_min 45 --out runs/tiny
python train.py --model nerf --iters 20000 --n_importance 64 --time_budget_min 45 --out runs/nerf
python render.py --run runs/hash --spiral --mesh
python make_report.py
pytest -q
```

Data layout expected by `nerf/data.py`: `data/transforms_train.json` (NeRF-synthetic format,
`camera_angle_x` + per-frame `transform_matrix`) and `data/images/train/r_*.png` (RGBA).
The 100 Lego views are split 90/10 into train/validation.

## References

Mildenhall et al., *NeRF* (ECCV 2020) · Mueller et al., *Instant Neural Graphics Primitives* (SIGGRAPH 2022)
