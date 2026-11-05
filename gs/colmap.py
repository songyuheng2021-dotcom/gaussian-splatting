"""Pure-Python reader for COLMAP text models (cameras.txt / images.txt / points3D.txt).

COLMAP's camera convention is already OpenCV (x right, y down, z forward) and images.txt
stores the world-to-camera rotation as a quaternion (qw, qx, qy, qz) plus translation, so no
axis flipping is needed. ETH3D ships its undistorted DSLR calibration in exactly this format
(`dslr_calibration_undistorted/`) with PINHOLE cameras, in the metric frame of the laser scans.
"""
import os
import numpy as np


def _qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_cameras(path):
    """-> {camera_id: dict(model, width, height, params)}"""
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        cams[int(f[0])] = dict(model=f[1], width=int(f[2]), height=int(f[3]), params=np.array(list(map(float, f[4:]))))
    return cams


def intrinsics(cam):
    """3x3 K for PINHOLE / SIMPLE_PINHOLE / (the pinhole part of) SIMPLE_RADIAL and RADIAL."""
    p, m = cam["params"], cam["model"]
    if m == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        raise ValueError(f"camera model {m} needs undistortion first (run `colmap image_undistorter`)")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def read_images(path):
    """-> list of dict(name, camera_id, w2c (4x4), image_id), sorted by name.
    images.txt has two lines per image; the second (2-D points) is skipped."""
    out, lines = [], [l for l in open(path) if not l.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    for i in range(0, len(lines), 2):             # keep blank 2nd lines: an image with no 2-D points has one
        f = lines[i].split()
        if len(f) < 10:
            continue
        q = np.array(list(map(float, f[1:5]))); t = np.array(list(map(float, f[5:8])))
        w2c = np.eye(4); w2c[:3, :3] = _qvec2rotmat(q); w2c[:3, 3] = t
        out.append(dict(image_id=int(f[0]), camera_id=int(f[8]), name=f[9], w2c=w2c))
    return sorted(out, key=lambda d: d["name"])


def read_points3d(path):
    """-> xyz (N,3) float32, rgb (N,3) float32 in [0,1]"""
    xyz, rgb = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        xyz.append([float(f[1]), float(f[2]), float(f[3])]); rgb.append([int(f[4]), int(f[5]), int(f[6])])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


def read_model(folder):
    cams = read_cameras(os.path.join(folder, "cameras.txt"))
    ims = read_images(os.path.join(folder, "images.txt"))
    p3 = os.path.join(folder, "points3D.txt")
    pts = read_points3d(p3) if os.path.exists(p3) else (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    return cams, ims, pts
