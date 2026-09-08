"""Unit tests for the Gaussian representation and the reference rasteriser (CPU, seconds)."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gs.model import GaussianModel, quat_to_rotmat
from gs.render import render_reference, eval_sh, project
from gs.data import Camera
from gs.losses import ssim


def _cam(W=48, H=48, f=40.0, dist=5.0):
    w2c = torch.eye(4); w2c[2, 3] = dist                   # camera at z=-dist looking down +z
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return Camera(torch.ones(H, W, 3), w2c, K)


def test_quat_gives_rotation_and_covariance_is_psd():
    q = torch.randn(50, 4)
    R = quat_to_rotmat(q)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(50, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(50), atol=1e-5)
    m = GaussianModel(torch.randn(50, 3), torch.rand(50, 3), init_scale=0.1)
    m.quats.data = q
    ev = torch.linalg.eigvalsh(m.covariances())
    assert (ev >= -1e-7).all()


def test_single_gaussian_renders_centered_blob_with_its_colour():
    m = GaussianModel(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0]]), init_scale=0.3)
    m.opacity_logits.data[:] = 10.0                          # opaque
    cam = _cam()
    img, info = render_reference(m, cam, bg=(0, 0, 0))
    c = img[24, 24]
    assert c[0] > 0.9 and c[1] < 0.05 and c[2] < 0.05         # red at the centre
    assert img[0, 0].sum() < 1e-3                             # background in the corner
    assert info["n_visible"] == 1


def test_depth_ordering_front_gaussian_wins():
    xyz = torch.tensor([[0, 0, 0.0], [0, 0, 1.0]])            # second one is 1 m farther from the camera
    m = GaussianModel(xyz, torch.tensor([[1, 0, 0.0], [0, 0, 1.0]]), init_scale=0.3)
    m.opacity_logits.data[:] = 10.0
    img, _ = render_reference(m, _cam(), bg=(0, 0, 0))
    assert img[24, 24, 0] > 0.9 and img[24, 24, 2] < 0.05


def test_gradients_flow_to_all_parameters():
    m = GaussianModel(torch.randn(20, 3) * 0.3, torch.rand(20, 3), init_scale=0.2)
    img, info = render_reference(m, _cam(), bg=(0, 0, 0))
    (img - 0.5).abs().mean().backward()
    for p in [m.means, m.log_scales, m.quats, m.opacity_logits, m.sh]:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert info["means2d"].grad is not None


def test_sh_degree0_is_constant_colour():
    sh = torch.zeros(5, 16, 3); sh[:, 0] = 1.0
    d = torch.randn(5, 3); d = d / d.norm(dim=-1, keepdim=True)
    c = eval_sh(sh, d, 3)
    assert torch.allclose(c, c[:1].expand(5, 3))


def test_densify_split_and_prune_keep_optimizer_consistent():
    m = GaussianModel(torch.randn(30, 3), torch.rand(30, 3), init_scale=0.5)
    opt = torch.optim.Adam(m.param_groups(1e-3))
    img, info = render_reference(m, _cam(), bg=(0, 0, 0)); img.mean().backward(); opt.step()
    m.grad_accum[:] = 1.0; m.grad_count[:] = 1.0            # everyone is "under-reconstructed"
    m.opacity_logits.data[:5] = -20.0                          # five transparent ones -> pruned
    st = m.densify_and_prune(opt, grad_thresh=0.5, scene_extent=1.0)
    assert st["split"] == 30 and st["pruned"] >= 30 and m.n == 60   # 30 split into 60 children, originals removed
    assert all(g["params"][0].shape[0] == m.n for g in opt.param_groups)
    img, _ = render_reference(m, _cam(), bg=(0, 0, 0)); img.mean().backward(); opt.step()


def test_ssim_identity():
    a = torch.rand(32, 32, 3)
    assert abs(float(ssim(a, a)) - 1.0) < 1e-4
