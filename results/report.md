## NeRF-synthetic benchmark (30k iterations, 800x800, 200 test views per scene)

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

## Ablation (lego)

| variant | PSNR | SSIM | LPIPS | #Gaussians | train [min] | FPS |
|---|---|---|---|---|---|---|
| full (30k, densify to 15k, SH deg 3, grad thr 2e-4) | 31.45 | 0.960 | 0.051 | 110k | 13 | 271 |
| no adaptive density control (300k fixed random Gaussians) | 13.52 | 0.725 | 0.425 | 300k | 29 | 68 |
| SH degree 0 (view-independent colour) | 31.03 | 0.958 | 0.053 | 158k | 13 | 245 |
| densification gradient threshold x2 (4e-4) | 28.38 | 0.929 | 0.097 | 20k | 11 | 451 |
| densification gradient threshold /2 (1e-4) | 33.54 | 0.974 | 0.030 | 444k | 22 | 129 |
| densification stopped at 5k iterations | 31.22 | 0.958 | 0.056 | 84k | 13 | 293 |
| 7k iterations only | 28.63 | 0.932 | 0.094 | 36k | 3 | 369 |

## Same scene, same GPU: HashNeRF (previous project) vs 3D Gaussian Splatting (lego)

| method | PSNR | SSIM | train [min] | render FPS | note |
|---|---|---|---|---|---|
| HashNeRF (Instant-NGP style, pure PyTorch, previous project) | 24.36 | 0.883 | 29 | 0.4 | 400x400, 15000 iters, 25 test views |
| 3D Gaussian Splatting (this project) | 31.28 | 0.960 | 14 | 263.0 | 800x800, 30000 iters, 200 test views, 109k Gaussians |
