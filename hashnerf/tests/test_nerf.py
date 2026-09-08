import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nerf.encoding import HashGridEncoding, SHEncoding, FrequencyEncoding
from nerf.render import composite, sample_pdf
from nerf.model import HashNeRF


def test_hash_encoding_shape_and_continuity():
    enc = HashGridEncoding(n_levels=4, log2_T=12, base_res=4, max_res=32)
    x = torch.rand(100, 3) * 3 - 1.5
    y = enc(x)
    assert y.shape == (100, 8)
    # trilinear interpolation: tiny displacement -> tiny change (continuity)
    y2 = enc(x + 1e-4)
    assert (y - y2).abs().max() < 1e-3


def test_sh_basis_is_orthonormal_on_sphere():
    torch.manual_seed(0)
    d = torch.randn(200000, 3); d = d / d.norm(dim=-1, keepdim=True)
    Y = SHEncoding()(d)                       # (N,16)
    G = Y.T @ Y / len(d) * 4 * torch.pi       # Monte-Carlo integral over the sphere
    assert (G - torch.eye(16)).abs().max() < 0.05


def test_frequency_encoding_dim():
    e = FrequencyEncoding(n_freqs=10)
    assert e(torch.zeros(5, 3)).shape == (5, e.out_dim) and e.out_dim == 63


def test_composite_opaque_and_empty():
    R, S = 4, 16
    t = torch.linspace(2, 6, S).expand(R, S)
    rd = torch.tensor([[0, 0, -1.0]]).expand(R, 3)
    rgb = torch.full((R, S, 3), 0.25)
    # empty space -> white background, zero accumulation
    c, depth, acc, w = composite(torch.zeros(R, S), rgb, t, rd)
    assert torch.allclose(c, torch.ones(R, 3)) and acc.max() < 1e-6
    # very dense first sample -> colour of first sample, depth ~ near
    sig = torch.zeros(R, S); sig[:, 0] = 1e4
    c, depth, acc, w = composite(sig, rgb, t, rd)
    assert torch.allclose(c, torch.full((R, 3), 0.25), atol=1e-4) and (depth - 2).abs().max() < 1e-3


def test_sample_pdf_concentrates_where_weights_are():
    bins = torch.linspace(2, 6, 17).expand(2, 17)
    w = torch.zeros(2, 16); w[:, 8] = 1.0                 # all weight in bin 8 -> t in [4, 4.25]
    s = sample_pdf(bins, w, 32, perturb=True)
    assert (s >= 4 - 1e-3).all() and (s <= 4.25 + 1e-3).all()


def test_hashnerf_forward_shapes():
    m = HashNeRF(log2_T=12, n_levels=4, max_res=32)
    x = torch.rand(10, 3) * 3 - 1.5; d = torch.randn(10, 3); d = d / d.norm(dim=-1, keepdim=True)
    sigma, rgb = m(x, d)
    assert sigma.shape == (10,) and rgb.shape == (10, 3) and (sigma >= 0).all() and (rgb >= 0).all() and (rgb <= 1).all()
