"""Unit tests for depth rendering, LiDAR projection, Sim(3) alignment and the geometry metrics (CPU, seconds)."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.model import GaussianModel
from gs.render import render_reference
from gs.data import Camera
from gs.geometry import umeyama_sim3, apply_sim3, icp_sim3, unproject_depth, geometry_metrics, error_grid, acceptance_table, voxel_downsample
from gs.lidar import project_points_to_depth, depth_loss, attach_lidar_depth
from gs.colmap import read_model, intrinsics


def _cam(W=48, H=48, f=40.0, dist=5.0):
    w2c = torch.eye(4); w2c[2, 3] = dist                     # camera 5 m in front of the origin, looking down +z
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return Camera(torch.ones(H, W, 3), w2c, K)


def test_expected_depth_of_single_opaque_gaussian_is_its_distance():
    m = GaussianModel(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0]]), init_scale=0.3)
    m.opacity_logits.data[:] = 10.0
    cam = _cam(dist=5.0)
    _, info = render_reference(m, cam, bg=(0, 0, 0), return_depth=True)
    assert abs(float(info["depth"][24, 24]) - 5.0) < 1e-3
    assert float(info["acc"][24, 24]) > 0.9                     # alpha is capped at 0.99 per Gaussian
    assert float(info["acc"][0, 0]) < 1e-3                   # corner: background, no depth support


def test_expected_depth_is_alpha_weighted_mix_of_two_gaussians():
    xyz = torch.tensor([[0, 0, 0.0], [0, 0, 1.0]])            # front at 5 m, back at 6 m
    m = GaussianModel(xyz, torch.rand(2, 3), init_scale=0.5)
    m.opacity_logits.data[:] = torch.logit(torch.tensor([0.5, 0.99]))
    cam = _cam(dist=5.0)
    _, info = render_reference(m, cam, bg=(0, 0, 0), return_depth=True)
    d = float(info["depth"][24, 24])
    # centre pixel: alpha_front ~0.5, back ~0.99*(1-0.5): depth = (0.5*5 + 0.495*6)/0.995 ~= 5.497
    assert 5.3 < d < 5.7


def test_unproject_roundtrip_matches_rendered_surface():
    m = GaussianModel(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0]]), init_scale=0.3)
    m.opacity_logits.data[:] = 10.0
    cam = _cam()
    _, info = render_reference(m, cam, bg=(0, 0, 0), return_depth=True)
    pts = unproject_depth(info["depth"], cam, info["acc"], acc_min=0.5)
    assert len(pts) > 10
    assert pts[:, 2].abs().max() < 0.2                        # the splat sits on the z=0 plane in world frame


def test_lidar_projection_hits_right_pixel_and_depth_and_zbuffers():
    cam = _cam()
    pts = np.array([[0, 0, 0.0],       # on the optical axis, 5 m away -> pixel (24,24)
                    [0, 0, 2.0],       # same pixel, 7 m away -> must lose the z-buffer
                    [1.0, 0, 0.0]])    # x=+1 m at 5 m -> u = 24 + 40*1/5 = 32
    d = project_points_to_depth(pts, cam)
    assert abs(float(d[24, 24]) - 5.0) < 1e-5
    assert abs(float(d[24, 32]) - 5.0) < 1e-5
    assert int((d > 0).sum()) == 2


def test_depth_loss_has_gradient_and_masks_unsupervised_pixels():
    m = GaussianModel(torch.zeros(1, 3), torch.tensor([[1.0, 0, 0]]), init_scale=0.3)
    m.opacity_logits.data[:] = 10.0
    cam = _cam()
    lidar = torch.zeros(48, 48); lidar[24, 24] = 5.5           # one supervised pixel, 0.5 m farther
    _, info = render_reference(m, cam, bg=(0, 0, 0), return_depth=True)
    loss, n = depth_loss(info["depth"], info["acc"], lidar, acc_min=0.5, depth_scale=1.0)
    assert n == 1 and abs(float(loss) - 0.5) < 1e-3
    loss.backward()
    assert m.means.grad is not None and torch.isfinite(m.means.grad).all() and m.means.grad.abs().sum() > 0
    scale = attach_lidar_depth([cam], np.array([[0, 0, 0.0]]), min_pixels=1, verbose=False)
    assert cam.depth is not None and abs(scale - 5.0) < 1e-5


def test_umeyama_sim3_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3))
    ang = 0.7; Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    s, t = 2.5, np.array([1.0, -2.0, 0.5])
    dst = apply_sim3(src, s, Rz, t)
    s2, R2, t2 = umeyama_sim3(src, dst)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, Rz, atol=1e-9) and np.allclose(t2, t, atol=1e-9)
    # ICP refinement: unknown correspondences, start from a small residual misalignment (scale 1.05, 3 deg, 5 cm)
    a2 = 0.05; Rp = np.array([[np.cos(a2), -np.sin(a2), 0], [np.sin(a2), np.cos(a2), 0], [0, 0, 1]])
    dense = rng.uniform(-3, 3, size=(30000, 3)); dense[:, 2] = 0.5 * np.sin(2 * dense[:, 0]) + 0.3 * np.cos(2 * dense[:, 1])
    (s3, R3, t3), rmse = icp_sim3(apply_sim3(dense, 1.05, Rp, np.array([0.05, 0, 0])), dense, iters=30)
    assert abs(s3 - 1 / 1.05) < 1e-3 and rmse < 1e-3


def test_metrics_zero_for_identical_clouds_and_grid_flags_bad_cell():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 10, size=(5000, 3)); ref[:, 2] = 0
    out, d_rec, d_ref, _ = geometry_metrics(ref, ref, taus=(0.05,))
    assert out["fscore@0.05"] == 1.0 and out["chamfer"] == 0.0
    rec = ref.copy(); bad = (rec[:, 0] < 2) & (rec[:, 1] < 2); rec[bad, 2] += 0.5      # one corner lifted by 0.5 m
    out, d_rec, _, _ = geometry_metrics(rec, ref, taus=(0.1,))
    assert out["accuracy@0.1"] < 1.0 and 0.0 < out["acc_mean"] < 0.5
    g = error_grid(rec, d_rec, cell=2.0, min_count=5)
    assert g["shape"] == (5, 5) and g["p90"][0, 0] > 0.4 and g["p90"][4, 4] < 1e-9
    tab = acceptance_table(g, tolerance=0.1)
    assert tab["cells_fail"] == 1 and tab["cells_pass"] == 24
    assert len(voxel_downsample(ref, 1.0)) <= 121


def test_colmap_text_reader(tmp_path):
    (tmp_path / "cameras.txt").write_text("# cams\n1 PINHOLE 100 80 50.0 50.0 50.0 40.0\n")
    (tmp_path / "images.txt").write_text("# ims\n1 1 0 0 0 0.1 0.2 0.3 1 b.jpg\n\n2 1 0 0 0 0 0 0 1 a.jpg\n\n")
    (tmp_path / "points3D.txt").write_text("1 0.5 0.5 0.5 255 0 0 0.1 1 2\n")
    cams, ims, (xyz, rgb) = read_model(str(tmp_path))
    assert intrinsics(cams[1])[0, 2] == 50.0
    assert [im["name"] for im in ims] == ["a.jpg", "b.jpg"] and np.allclose(ims[1]["w2c"][:3, 3], [0.1, 0.2, 0.3])
    assert xyz.shape == (1, 3) and rgb[0, 0] == 1.0
