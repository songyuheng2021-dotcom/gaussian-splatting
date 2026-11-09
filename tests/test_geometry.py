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


def test_floater_mask_flags_gaussians_in_front_of_lidar_surface():
    from gs.lidar import floater_mask, prune_gaussians
    # LiDAR "surface": a plane of points at z=0 (5 m from the camera); two Gaussians on it, one 1 m in front
    g = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 41), torch.linspace(-2, 2, 41), indexing="ij"), -1).reshape(-1, 2)
    plane = torch.cat([g, torch.zeros(len(g), 1)], 1)
    cams = [_cam(dist=5.0), _cam(dist=5.0)]
    cams[1].w2c[0, 3] = 0.3                                  # second view shifted sideways
    attach_lidar_depth(cams, plane, splat=2, verbose=False)
    means = torch.tensor([[0.0, 0, 0], [0.5, 0.5, 0], [0.0, 0, -1.0]])   # third sits 1 m in front (closer to camera)
    flag, n_checked = floater_mask(means, cams, thr=0.5, min_views=2)
    assert flag.tolist() == [False, False, True]
    assert (n_checked >= 2).all()
    m = GaussianModel(means, torch.rand(3, 3), init_scale=0.2)
    m = prune_gaussians(m, ~flag)
    assert m.n == 2 and m.sh.shape[0] == 2 and m.grad_accum.shape[0] == 2

def test_blockwise_metrics_use_full_cloud_neighbours_not_per_block_search():
    from gs.geometry import holdout_mask, blockwise_metrics, block_ids, hidden_blocks, in_blocks
    from scipy.spatial import cKDTree
    # a 20 m x 20 m flat reference on a 5 cm grid; reconstruction = reference shifted 4 cm along x, so every
    # point's true nearest neighbour is 1 cm away, and for points at x = k - 0.01 that neighbour lies across a
    # 1 m block border
    g = np.stack(np.meshgrid(np.arange(0, 20, 0.05), np.arange(0, 20, 0.05), indexing="ij"), -1).reshape(-1, 2)
    ref = np.c_[g, np.zeros(len(g))]
    rec = ref + np.array([0.04, 0.0, 0.0])
    _, d_rec, d_ref, _ = geometry_metrics(rec, ref, taus=(0.02,))
    out = blockwise_metrics(rec, ref, d_rec, d_ref, block=1.0, frac=0.5, seed=3, taus=(0.02,))
    # both subsets see the true 1 cm error
    for k in ("supervised_blocks", "heldout_blocks"):
        assert abs(out[k]["acc_mean"] - 0.01) < 1e-3 and out[k]["fscore@0.02"] > 0.99
    # the reference and reconstruction agree on which blocks are hidden
    assert (holdout_mask(ref, 1.0, 0.5, 3) == holdout_mask(ref.copy(), 1.0, 0.5, 3)).all()
    # the naive per-block search (the old bug) is worse: border points lose their true neighbour
    _, hidden = hidden_blocks(ref, 1.0, 0.5, 3)
    hr, hg = in_blocks(rec, 1.0, hidden), holdout_mask(ref, 1.0, 0.5, 3)
    naive = cKDTree(ref[hg]).query(rec[hr])[0].mean()
    assert naive > out["heldout_blocks"]["acc_mean"] + 5e-4      # 0.0108 vs 0.0101 m here
    # roughly half of the blocks are hidden
    assert 0.3 < hg.mean() < 0.7 and len(np.unique(block_ids(ref, 1.0))) == 400


def test_heldout_split_is_defined_on_the_reference_and_survives_floaters():
    """Regression: the block lottery used to run separately on rec and on ref. A reconstruction never occupies
    exactly the reference's blocks, so the draws fell out of step and rec's 'held-out' half was arbitrary.
    Here the reconstruction is exact in supervised blocks, 10 cm too high in held-out blocks, and has a floater
    cluster in a block the reference does not occupy. A correct split must see 0 vs 0.10 m."""
    from gs.geometry import blockwise_metrics, hidden_blocks, in_blocks, holdout_mask
    g = np.stack(np.meshgrid(np.arange(0, 20, 0.05), np.arange(0, 20, 0.05), indexing="ij"), -1).reshape(-1, 2)
    ref = np.c_[g, np.zeros(len(g))]
    _, hidden = hidden_blocks(ref, 1.0, 0.5, 7)
    rec = ref.copy(); rec[in_blocks(ref, 1.0, hidden), 2] += 0.10
    floater = np.random.default_rng(0).uniform([5, 5, 10], [5.9, 5.9, 10.9], (300, 3))
    rec = np.r_[floater, rec]                          # floaters first: shifts any per-cloud lottery
    _, d_rec, d_ref, _ = geometry_metrics(rec, ref, taus=(0.05,), max_dist=1.0)
    out = blockwise_metrics(rec, ref, d_rec, d_ref, block=1.0, frac=0.5, seed=7, taus=(0.05,), max_dist=1.0)
    assert out["supervised_blocks"]["acc_mean"] < 1e-6
    assert abs(out["heldout_blocks"]["acc_mean"] - 0.10) < 1e-6
    assert abs(out["split"]["rec_frac_unassigned"] - 300 / len(rec)) < 1e-9
    # what the old per-cloud lottery would have done: rec's own draw disagrees with the reference's split
    old = holdout_mask(rec, 1.0, 0.5, 7)[300:]          # lottery re-run on rec's own blocks (incl. the floater's)
    assert (old != in_blocks(rec[300:], 1.0, hidden)).mean() > 0.2
