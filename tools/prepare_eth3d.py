#!/usr/bin/env python3
"""Turn an extracted ETH3D high-res multi-view training scene into the layout read by
gs.data.load_colmap_scene and run_geometry.py.

    # downloads (eth3d.net/datasets), ~0.7 GB for courtyard
    wget https://www.eth3d.net/data/courtyard_dslr_undistorted.7z https://www.eth3d.net/data/courtyard_dslr_scan_eval.7z
    7z x courtyard_dslr_undistorted.7z -o raw && 7z x courtyard_dslr_scan_eval.7z -o raw
    python tools/prepare_eth3d.py --raw raw/courtyard --out data/eth3d/courtyard --downscale 4

Output
    out/colmap/{cameras,images,points3D}.txt   copied from dslr_calibration_undistorted (full-res intrinsics;
                                               the loader rescales K from the image size on disk)
    out/images/<name>                          undistorted JPGs, downscaled
    out/gt_scan.ply                            all laser scans of dslr_scan_eval merged (scan_alignment.mlp
                                               transforms applied), voxel-downsampled to --gt_voxel metres
    out/scene_info.json                        counts, bounding box, voxel size
The laser scans and the COLMAP model are in the same metric frame (ETH3D registers the images to
the scans), so no Sim(3) alignment is needed; run_geometry.py can still verify this with --align.
"""
import argparse, glob, json, os, re, shutil
import numpy as np
from PIL import Image


def read_mlp(path):
    """MeshLab project: {ply filename: 4x4 matrix}."""
    txt = open(path).read()
    out = {}
    for m in re.finditer(r'<MLMesh[^>]*filename="([^"]+)"[^>]*>(.*?)</MLMesh>', txt, re.S):
        fn, body = m.group(1), m.group(2)
        mm = re.search(r"<MLMatrix44>(.*?)</MLMatrix44>", body, re.S)
        M = np.array(list(map(float, mm.group(1).split()))).reshape(4, 4) if mm else np.eye(4)
        out[os.path.basename(fn)] = M
    return out


def load_scans(folder, voxel):
    import open3d as o3d
    mlp = glob.glob(os.path.join(folder, "*.mlp"))
    mats = read_mlp(mlp[0]) if mlp else {}
    clouds = []
    for ply in sorted(glob.glob(os.path.join(folder, "*.ply"))):
        pc = o3d.io.read_point_cloud(ply)
        M = mats.get(os.path.basename(ply), np.eye(4))
        pc.transform(M)
        pc = pc.voxel_down_sample(voxel)
        clouds.append(pc)
        print(f"  {os.path.basename(ply)}: {len(pc.points)} pts after {voxel} m voxel, transform {'from mlp' if os.path.basename(ply) in mats else 'identity'}")
    merged = clouds[0]
    for c in clouds[1:]:
        merged += c
    return merged.voxel_down_sample(voxel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="extracted scene folder, e.g. raw/courtyard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--downscale", type=int, default=4, help="6048x4032 -> 1512x1008 at 4")
    ap.add_argument("--gt_voxel", type=float, default=0.02, help="metres; the scans have mm spacing, 2 cm keeps ~10M pts for courtyard")
    a = ap.parse_args()
    calib = glob.glob(os.path.join(a.raw, "**", "dslr_calibration_undistorted"), recursive=True)
    imgs = glob.glob(os.path.join(a.raw, "**", "dslr_images_undistorted"), recursive=True)
    scans = glob.glob(os.path.join(a.raw, "**", "dslr_scan_eval"), recursive=True)
    assert calib and imgs, f"could not find dslr_calibration_undistorted / dslr_images_undistorted under {a.raw}"
    os.makedirs(f"{a.out}/colmap", exist_ok=True); os.makedirs(f"{a.out}/images", exist_ok=True)
    for f in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(os.path.join(calib[0], f), f"{a.out}/colmap/{f}")
    # images.txt names look like "dslr_images_undistorted/DSC_0286.JPG"; keep the basename on disk and in images.txt
    lines = open(f"{a.out}/colmap/images.txt").read().splitlines()
    fixed = []
    for l in lines:
        f = l.split()
        if len(f) >= 10 and not l.startswith("#") and "/" in f[9]:
            f[9] = os.path.basename(f[9]); l = " ".join(f)
        fixed.append(l)
    open(f"{a.out}/colmap/images.txt", "w").write("\n".join(fixed) + "\n")
    n = 0
    for p in sorted(glob.glob(os.path.join(imgs[0], "*"))):
        im = Image.open(p)
        if a.downscale > 1:
            im = im.resize((im.width // a.downscale, im.height // a.downscale), Image.LANCZOS)
        im.save(f"{a.out}/images/{os.path.basename(p)}", quality=95); n += 1
    print(f"{n} images -> {a.out}/images at 1/{a.downscale}")
    info = dict(n_images=n, downscale=a.downscale, gt_voxel=a.gt_voxel)
    if scans:
        import open3d as o3d
        pc = load_scans(scans[0], a.gt_voxel)
        o3d.io.write_point_cloud(f"{a.out}/gt_scan.ply", pc)
        pts = np.asarray(pc.points)
        info.update(n_gt_points=int(len(pts)), bbox_min=pts.min(0).tolist(), bbox_max=pts.max(0).tolist())
        print(f"ground truth: {len(pts)} points, bbox {pts.min(0).round(1)} .. {pts.max(0).round(1)}")
    else:
        print("no dslr_scan_eval folder found: geometry evaluation will not be possible for this scene")
    json.dump(info, open(f"{a.out}/scene_info.json", "w"), indent=1)


if __name__ == "__main__":
    main()
