#!/usr/bin/env python3
"""Turn one run of UAVScenes (ICCV 2025, built on MARS-LVIG: DJI M300 RTK, Hikvision 2448x2048 global-shutter
camera and Livox Avia LiDAR, hardware-synchronised, 80-130 m altitude) into the scene layout run_geometry.py reads.

Input (as extracted from the dataset's interval5_CAM_LIDAR.zip and terra_3dmap_pointcloud_mesh.zip):
    <root>/cam/<stamp>.jpg                         images
    <root>/lidar/image<stamp>_lidar<stamp>.txt     Livox Avia scan, x y z in the LiDAR frame, one per image
    <root>/sampleinfos_interpolated.json           per image: T4x4 (camera-to-map, OpenCV camera axes), P3x3, W, H
    <root>/cloud_merged.ply                        DJI Terra map of the whole scene (dense, coloured)

Output (same layout as tools/prepare_eth3d.py):
    <out>/colmap/{cameras,images,points3D}.txt     one PINHOLE camera per image, world-to-camera poses
    <out>/images/<stamp>.jpg                       downscaled images
    <out>/gt_scan.ply                              LiDAR reference: Avia scans accumulated in the map frame, kept where
                                                   the cameras can see them (front-most in >= --min_views images)
    <out>/gt_scan_all.ply                          the same accumulation before the visibility filter
    <out>/terra_ref.ply                            second reference: the Terra map cropped to the same window
    <out>/scene_info.json

Conventions verified on AMvalley01 before writing this (see the project notes):
  * T4x4 is camera-to-map: rendering the Terra map through inv(T4x4) and P3x3 and correlating it with the photo
    peaks at zero shift (NCC 0.52-0.65 on three frames); the opposite convention gives NCC ~0.
  * camera_ext_R / camera_ext_t from the dataset's calibration_results.py map LiDAR points into the camera
    (p_cam = R p_lidar + t); a LiDAR point then lands at the depth of the Terra surface along the same ray.
  * the images are used with P3x3 and no distortion model, as the dataset's own json provides.

Independence of the references. The Terra map is a dense coloured reconstruction and is likely photogrammetric,
i.e. derived from the same images; the Avia scans are an independent sensor. gt_scan.ply (LiDAR) is therefore the
primary reference and the supervision source; terra_ref.ply is reported as a second reading only. The LiDAR map is
registered with the image poses and the camera-LiDAR extrinsic, so its absolute accuracy is bounded by both; the
LiDAR-vs-Terra depth agreement printed at the end is the sanity check for that.

Visibility filter. The Avia returns also come from under tree canopy and from oblique angles no camera sees; on
AMvalley01 the vertical spread of LiDAR points in 0.5 m columns is ~6x that of the Terra surface. Completeness against
such points would penalise the reconstruction for geometry no image constrains, so the reference keeps a LiDAR point
only if it is front-most (z-buffer on a coarse pixel grid, tolerance --vis_tol + 0.5 % of depth) in at least
--min_views of the selected cameras. The same filtered cloud is what the depth supervision is drawn from.

Reference noise floor. Scans are split into two interleaved halves (odd / even) and accumulated separately; the
nearest-neighbour distance between the two halves is the repeatability of the LiDAR map (ranging + pose + extrinsic).
Metrics at tolerances below that floor are not meaningful and the aerial runs use tolerances above it.

Initial Gaussians come from the Terra map subsampled to --init_voxel and --n_init points, standing in for the SfM
tie points a normal 3DGS pipeline would start from. Both arms of every experiment use the same initialisation.
"""
import argparse, json, os, re
import numpy as np
from PIL import Image

# calibration_results.py of the UAVScenes repository (camera <- LiDAR), by scene family
CALIB = {
    "AMvalley": dict(R=[0.00298068, -0.999735, -0.0231428, -0.00504595, 0.023132, -0.99974, 0.999985, 0.00309701, -0.00497598],
                     t=[-0.0025563, 0.0567484, -0.0512149]),
    "AMtown": dict(R=[0.00298088, -0.999728, -0.0231416, -0.00504636, 0.0231263, -0.99972, 0.999983, 0.00309683, -0.00497605],
                   t=[0.0025563, 0.0567484, -0.0512149]),
    "HKisland": dict(R=[0.00352762, -0.999765, -0.0213775, -0.0111803, 0.0213369, -0.99971, 0.999931, 0.00376561, -0.0111025],
                     t=[-0.0025563, 0.0470454, -0.0513375]),
    "HK_GNSS": dict(R=[0.00363212, -0.999819, -0.0213618, -0.0111679, 0.0214512, -0.999591, 0.999879, 0.00375613, -0.0111134],
                    t=[-0.0021928, 0.0470312, -0.0513126]),
}


def rotmat_to_qvec(R):
    """Shepperd's method, robust for any rotation (nadir cameras are close to 180-degree rotations)."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def voxel(pts, v, cols=None):
    key = np.floor(pts / v).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return (pts[idx], cols[idx]) if cols is not None else pts[idx]


def read_xyz(path):
    try:
        import pandas as pd
        return pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 2], dtype=np.float64).to_numpy()
    except ImportError:
        return np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    except Exception:                                            # empty file
        return np.zeros((0, 3))


def count_visible(X, cams, grid, tol):
    """For each point, the number of cameras in which it is front-most: z-buffer on the image downsampled by grid,
    a point counts if its depth is within tol + 0.5 % of the pixel's minimum depth."""
    nv = np.zeros(len(X), np.int32)
    for info, T in cams:
        K = np.array(info["P3x3"], dtype=np.float64); W, H = info["Width"] // grid, info["Height"] // grid
        w2c = np.linalg.inv(T); pc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]; front = z > 1.0
        u = np.full(len(X), -1, np.int64); v = np.full(len(X), -1, np.int64)
        u[front] = np.floor((K[0, 0] * pc[front, 0] / z[front] + K[0, 2]) / grid).astype(np.int64)
        v[front] = np.floor((K[1, 1] * pc[front, 1] / z[front] + K[1, 2]) / grid).astype(np.int64)
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pix = v[inb] * W + u[inb]
        zbuf = np.full(W * H, np.inf); np.minimum.at(zbuf, pix, z[inb])
        zi = z[inb]
        nv[np.flatnonzero(inb)[zi <= zbuf[pix] + tol + 0.005 * zi]] += 1
    return nv


def col_sd(X, cell=0.5, min_pts=4):
    """Median over 0.5 m xy columns of the z standard deviation: how 'thick' a cloud is vertically."""
    key = np.floor(X[:, :2] / cell).astype(np.int64); kk = key[:, 0] * 1_000_003 + key[:, 1]
    o = np.argsort(kk, kind="stable"); kk = kk[o]; z = X[o, 2]
    _, start, cnt = np.unique(kk, return_index=True, return_counts=True)
    s1 = np.add.reduceat(z, start); s2 = np.add.reduceat(z * z, start)
    keep = cnt >= min_pts
    sd = np.sqrt(np.maximum(s2[keep] / cnt[keep] - (s1[keep] / cnt[keep]) ** 2, 0))
    return float(np.median(sd))


def pick_window(centres, size):
    """Square window (in map x, y) containing the most camera centres."""
    best, best_c = -1, None
    for c in centres[:: max(1, len(centres) // 400)]:
        n = int((np.abs(centres[:, :2] - c[:2]) <= size / 2).all(1).sum())
        if n > best:
            best, best_c = n, c[:2].copy()
    return best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="AMvalley", choices=sorted(CALIB))
    ap.add_argument("--window", type=float, default=200.0, help="side of the square map window [m]")
    ap.add_argument("--center", type=float, nargs=2, default=None, help="window centre in map x y; default = densest")
    ap.add_argument("--max_images", type=int, default=120)
    ap.add_argument("--min_spacing", type=float, default=0.0, help="minimum distance between selected cameras [m]")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--lidar_voxel", type=float, default=0.2, help="voxel of the accumulated LiDAR reference [m]")
    ap.add_argument("--lidar_margin", type=float, default=60.0,
                    help="also accumulate scans from cameras this far outside the window (their footprint reaches in)")
    ap.add_argument("--min_views", type=int, default=2, help="LiDAR point must be visible in this many cameras")
    ap.add_argument("--vis_grid", type=int, default=8, help="z-buffer grid = full-resolution image / vis_grid")
    ap.add_argument("--vis_tol", type=float, default=0.5, help="visibility depth tolerance [m] (+0.5 %% of depth)")
    ap.add_argument("--init_voxel", type=float, default=0.5); ap.add_argument("--n_init", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "colmap"), exist_ok=True); os.makedirs(os.path.join(a.out, "images"), exist_ok=True)

    infos = {s["OriginalImageName"]: s for s in json.load(open(os.path.join(a.root, "sampleinfos_interpolated.json")))}
    names = sorted(n for n in os.listdir(os.path.join(a.root, "cam")) if n in infos)
    lidar_of = {}
    for f in os.listdir(os.path.join(a.root, "lidar")):
        m = re.match(r"image([\d.]+)_lidar[\d.]+\.txt$", f)
        if m:
            lidar_of[m.group(1) + ".jpg"] = f
    c2w = {n: np.array(infos[n]["T4x4"], dtype=np.float64) for n in names}
    centres = np.array([c2w[n][:3, 3] for n in names])
    ctr = np.array(a.center) if a.center else pick_window(centres, a.window)
    half = a.window / 2
    inside = (np.abs(centres[:, :2] - ctr) <= half).all(1)
    cand = [n for n, ok in zip(names, inside) if ok]
    # spatially even subsample: greedy farthest-point on camera centres
    pts = np.array([c2w[n][:3, 3] for n in cand])
    chosen = [int(rng.integers(len(cand)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < min(a.max_images, len(cand)):
        j = int(d.argmax())
        if d[j] < a.min_spacing:
            break
        chosen.append(j); d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    sel = sorted(cand[j] for j in chosen)
    print(f"{len(names)} images in run, window centre {ctr.round(1)} side {a.window} m: {len(cand)} inside, {len(sel)} selected")

    # ---- cameras / images
    with open(os.path.join(a.out, "colmap", "cameras.txt"), "w") as fc, open(os.path.join(a.out, "colmap", "images.txt"), "w") as fi:
        fc.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"); fi.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i, n in enumerate(sel, 1):
            s = infos[n]; K = np.array(s["P3x3"])
            fc.write(f"{i} PINHOLE {s['Width']} {s['Height']} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
            w2c = np.linalg.inv(c2w[n]); q = rotmat_to_qvec(w2c[:3, :3]); t = w2c[:3, 3]
            fi.write(f"{i} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {i} {n}\n\n")
            im = Image.open(os.path.join(a.root, "cam", n)).convert("RGB")
            if a.downscale > 1:
                im = im.resize((im.width // a.downscale, im.height // a.downscale), Image.LANCZOS)
            im.save(os.path.join(a.out, "images", n), quality=95)

    # ---- LiDAR reference: accumulate Avia scans in the map frame
    R = np.array(CALIB[a.calib]["R"]).reshape(3, 3); t = np.array(CALIB[a.calib]["t"])
    near = (np.abs(centres[:, :2] - ctr) <= half + a.lidar_margin).all(1)
    acc, halves, n_scans = [], ([], []), 0
    lo_w, hi_w = ctr - half, ctr + half
    for n, ok in zip(names, near):
        if not ok or n not in lidar_of:
            continue
        P = read_xyz(os.path.join(a.root, "lidar", lidar_of[n]))
        if len(P) == 0:
            continue
        pc = P @ R.T + t
        pc = pc[pc[:, 2] > 1.0]                                  # in front of the camera
        pw = pc @ c2w[n][:3, :3].T + c2w[n][:3, 3]
        pw = voxel(pw[((pw[:, :2] >= lo_w) & (pw[:, :2] <= hi_w)).all(1)], a.lidar_voxel)
        acc.append(pw); halves[n_scans % 2].append(pw); n_scans += 1
    lidar_all = voxel(np.concatenate(acc), a.lidar_voxel)
    import open3d as o3d
    from scipy.spatial import cKDTree
    save = lambda path, X: o3d.io.write_point_cloud(path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)))
    save(os.path.join(a.out, "gt_scan_all.ply"), lidar_all)
    nviews = count_visible(lidar_all, [(infos[n], c2w[n]) for n in sel], a.vis_grid, a.vis_tol)
    lidar = lidar_all[nviews >= a.min_views]
    save(os.path.join(a.out, "gt_scan.ply"), lidar)
    print(f"LiDAR reference: {n_scans} Avia scans, {len(lidar_all)} points after {a.lidar_voxel} m voxel, "
          f"{len(lidar)} visible in >= {a.min_views} cameras")

    # reference noise floor: odd vs even scans, restricted to the visible region
    A = voxel(np.concatenate(halves[0]), a.lidar_voxel); B = voxel(np.concatenate(halves[1]), a.lidar_voxel)
    vkey = lambda X: (lambda k: k[:, 0] * 73856093 ^ k[:, 1] * 19349663 ^ k[:, 2] * 83492791)(
        np.floor(X / a.lidar_voxel).astype(np.int64))
    vis_keys = vkey(lidar)
    A = A[np.isin(vkey(A), vis_keys)]; B = B[np.isin(vkey(B), vis_keys)]
    d_ab = np.concatenate([cKDTree(B).query(A, k=1, workers=-1)[0], cKDTree(A).query(B, k=1, workers=-1)[0]])
    floor = dict(p50=float(np.median(d_ab)), p90=float(np.percentile(d_ab, 90)),
                 **{f"frac_lt_{tau}": float((d_ab < tau).mean()) for tau in (0.1, 0.2, 0.3, 0.5, 1.0)})
    print("LiDAR repeatability (odd vs even scans):", {k: round(v, 3) for k, v in floor.items()})

    # ---- Terra map: second reference + initial points
    pcd = o3d.io.read_point_cloud(os.path.join(a.root, "cloud_merged.ply"))
    M = np.asarray(pcd.points); C = np.asarray(pcd.colors) if pcd.has_colors() else np.full((len(M), 3), 0.5)
    pad = half + 30.0
    m = (np.abs(M[:, :2] - ctr) <= pad).all(1)
    Mw, Cw = M[m], C[m]
    win = (np.abs(Mw[:, :2] - ctr) <= half).all(1)
    terra = voxel(Mw[win], a.lidar_voxel)
    save(os.path.join(a.out, "terra_ref.ply"), terra)
    init, icol = voxel(Mw, a.init_voxel, Cw)
    if len(init) > a.n_init:
        k = rng.choice(len(init), a.n_init, replace=False); init, icol = init[k], icol[k]
    with open(os.path.join(a.out, "colmap", "points3D.txt"), "w") as fp:
        fp.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, (p, c) in enumerate(zip(init, (icol * 255).astype(int)), 1):
            fp.write(f"{i} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n")
    print(f"Terra reference: {len(terra)} points; initial Gaussians: {len(init)} (Terra map, {a.init_voxel} m voxel)")

    # ---- sanity check: LiDAR vs Terra surface along the vertical (nearest neighbour in xy, height difference)
    tr = cKDTree(terra[:, :2]); dxy, j = tr.query(lidar[:, :2], k=1, workers=-1)
    ok = dxy < a.lidar_voxel
    dz = lidar[ok, 2] - terra[j[ok], 2]
    gsd_f = np.median([np.array(infos[n]["P3x3"])[0, 0] for n in sel]) / a.downscale
    cam_h = np.median([c2w[n][2, 3] for n in sel]) - np.median(lidar[:, 2])
    info = dict(run=os.path.basename(a.root.rstrip("/")), calib=a.calib, window_center=ctr.tolist(), window_m=a.window,
                n_images=len(sel), n_candidates=len(cand), downscale=a.downscale, n_lidar=int(len(lidar)), n_scans=n_scans,
                n_terra=int(len(terra)), n_init=int(len(init)), lidar_voxel=a.lidar_voxel,
                n_lidar_all=int(len(lidar_all)), min_views=a.min_views,
                camera_height_above_lidar_median_m=float(cam_h), gsd_m=float(cam_h / gsd_f),
                lidar_repeatability=floor,
                column_z_sd_p50=dict(lidar_all=col_sd(lidar_all), lidar_visible=col_sd(lidar), terra=col_sd(terra)),
                lidar_minus_terra_dz=dict(n=int(ok.sum()), median=float(np.median(dz)) if ok.any() else None,
                                          abs_p50=float(np.median(np.abs(dz))) if ok.any() else None,
                                          abs_p90=float(np.percentile(np.abs(dz), 90)) if ok.any() else None),
                bbox_min=lidar.min(0).tolist(), bbox_max=lidar.max(0).tolist())
    json.dump(info, open(os.path.join(a.out, "scene_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
