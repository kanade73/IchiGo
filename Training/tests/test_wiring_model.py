import hashlib

import numpy as np
import pytest
import torch

from ichigo_train.model import HEAD_TENSOR_NAMES, LogicNet, build_model, head_shapes, postprocess
from ichigo_train.wiring import DEFAULT_SEED, ModelSpec, generate_wiring, validate_wiring

# Computed with the pre-T33 wiring.py (only spec.seed, hardcoded 90/10 bank split) -- any change
# to generate_wiring's default behaviour (ModelSpec.from_profile, i.e. bank1_ratio=0.1,
# wiring_seed=None) must keep reproducing this exact byte sequence.
TINY_DEFAULT_WIRING_SHA256 = "5fafed9075c75af8ffef8ca7cb39360f9ee7b3d92b05a3208ad442c0b3027202"


def test_seed_reproducible_and_structure():
    w1 = generate_wiring(ModelSpec.from_profile("tiny"))
    w2 = generate_wiring(ModelSpec.from_profile("tiny"))
    assert np.array_equal(w1.wiring, w2.wiring) and np.array_equal(w1.theta, w2.theta)
    w3 = generate_wiring(ModelSpec.from_profile("tiny", seed=1))
    assert not np.array_equal(w1.wiring, w3.wiring)
    validate_wiring(w1.wiring, w1.dilations)
    L, C = w1.wiring.shape[:2]
    # layer 0: A = (0, c%32, 0, 0); all bank 0
    for c in range(C):
        assert w1.wiring[0, c, 0].tolist() == [0, c % 32, 0, 0]
    assert (w1.wiring[0, :, :, 0] == 0).all()
    # layer>=1: c%4==0 has identity A; bank1 appears somewhere
    for l in range(1, L):
        for c in range(0, C, 4):
            assert w1.wiring[l, c, 0].tolist() == [0, c, 0, 0]
    assert (w1.wiring[1:, :, :, 0] == 1).any()
    # theta init
    for l in range(L):
        for c in range(C):
            if c % 4 == 0:
                exp = np.zeros(16, dtype=np.float32); exp[12] = 3
                assert np.array_equal(w1.theta[l, c], exp)
            else:
                assert np.abs(w1.theta[l, c]).max() < 0.5


def test_validate_rejects_bad_wiring():
    w = generate_wiring(ModelSpec.from_profile("tiny"))
    bad = w.wiring.copy(); bad[0, 5, 1, 0] = 1
    with pytest.raises(ValueError):
        validate_wiring(bad, w.dilations)
    bad = w.wiring.copy(); bad[2, 5, 1, 2] = 1  # layer 2 has dilation 2
    with pytest.raises(ValueError):
        validate_wiring(bad, w.dilations)
    bad = w.wiring.copy(); bad[1, 1, 1] = bad[1, 1, 0]
    with pytest.raises(ValueError):
        validate_wiring(bad, w.dilations)


def test_head_shapes_and_forward_both_sizes():
    m = build_model("tiny")
    shapes = head_shapes(64)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]
    for S, B in ((9, 3), (19, 2)):
        sp = torch.from_numpy(np.random.default_rng(S).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
        g = torch.zeros(B, 4)
        out = m.forward_hard(sp, g)
        assert out["policy_logits"].shape == (B, S * S + 1)
        assert out["wdl_logits"].shape == (B, 3)
        assert out["score_mean"].shape == (B,)
        assert out["ownership"].shape == (B, S * S)
        soft = m(sp, g, tau=1.0)
        assert soft["policy_logits"].shape == (B, S * S + 1)


def test_out_of_board_reads_zero():
    """A layer-0 gate reading only an offset neighbour (gate 'b' with B off-board) yields 0 at the edge."""
    spec = ModelSpec(channels=16, dilations=[1])
    w = generate_wiring(spec)
    w.wiring[0, 0, 1] = [0, 28, 1, 0]  # B = channel 28 (constant 1) at (x+1, y)
    w.theta[0, 0] = 0; w.theta[0, 0, 10] = 5  # gate 'b'
    m = LogicNet(spec, w)
    sp = torch.zeros(1, 9, 9, 32, dtype=torch.uint8); sp[..., 28] = 1
    out = m.hard_layers(sp)[0][0, :, :, 0].numpy()
    assert (out[:, :8] == 1).all() and (out[:, 8] == 0).all()


def test_soft_with_sharp_theta_matches_hard():
    m = build_model("tiny")
    with torch.no_grad():
        m.theta.mul_(0)
        g = torch.from_numpy(m.hard_gates().astype(np.int64))
        m.theta.scatter_(2, g.unsqueeze(-1), 60.0)
    sp = torch.from_numpy(np.random.default_rng(3).integers(0, 2, size=(2, 9, 9, 32)).astype(np.uint8))
    glob = torch.zeros(2, 4)
    hard = m.forward_hard(sp, glob)
    soft = m(sp, glob, tau=1.0)
    assert torch.allclose(hard["policy_logits"], soft["policy_logits"].detach(), atol=1e-4)


def test_frozen_prefix_uses_hard_gates():
    m = build_model("tiny")
    sp = torch.from_numpy(np.random.default_rng(4).integers(0, 2, size=(1, 9, 9, 32)).astype(np.uint8))
    layers_soft = m.soft_layers(sp, tau=1.0, frozen_prefix=2)
    layers_hard = m.hard_layers(sp)
    assert torch.equal(layers_soft[1], layers_hard[1].to(torch.float32))


def test_postprocess_masks_illegal():
    logits = np.array([[1.0, 2.0, 3.0, 0.0]], dtype=np.float32)
    legal = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    out = postprocess(logits, legal, np.zeros((1, 3), dtype=np.float32))
    assert out["policy"][0, 1] == 0 and abs(out["policy"].sum() - 1) < 1e-6
    assert abs(out["expected_result"][0] - 0.5) < 1e-6
    with pytest.raises(ValueError):
        postprocess(logits, np.zeros((1, 4), dtype=np.uint8), np.zeros((1, 3), dtype=np.float32))


# ---- T33: bank1_ratio / wiring_seed (docs/spec/04-tasks.md T33) ----

def test_default_wiring_reproducibility_is_bit_for_bit_unchanged():
    """ModelSpec.from_profile's defaults (bank1_ratio=0.1, wiring_seed=None) must reproduce the
    exact wiring bytes generated before T33 added those fields."""
    w = generate_wiring(ModelSpec.from_profile("tiny"))
    assert hashlib.sha256(w.wiring.tobytes()).hexdigest() == TINY_DEFAULT_WIRING_SHA256


def test_custom_with_defaults_matches_from_profile():
    p = ModelSpec.from_profile("tiny")
    c = ModelSpec.custom(channels=p.channels, dilations=p.dilations, seed=DEFAULT_SEED, profile="tiny")
    assert c.bank1_ratio == 0.1 and c.wiring_seed is None
    wp, wc = generate_wiring(p), generate_wiring(c)
    assert np.array_equal(wp.wiring, wc.wiring) and np.array_equal(wp.theta, wc.theta)


def test_bank1_ratio_zero_has_no_bank1_references():
    spec = ModelSpec.custom(channels=64, dilations=[1, 1, 2, 1], seed=5, bank1_ratio=0.0)
    w = generate_wiring(spec)
    assert not (w.wiring[1:, :, :, 0] == 1).any()
    validate_wiring(w.wiring, spec.dilations)


def test_bank1_ratio_higher_gives_more_bank1_references():
    dilations = [1, 1, 2, 1, 4, 1, 8, 1]
    low = generate_wiring(ModelSpec.custom(channels=256, dilations=dilations, seed=9, bank1_ratio=0.1))
    high = generate_wiring(ModelSpec.custom(channels=256, dilations=dilations, seed=9, bank1_ratio=0.3))
    frac_low = float((low.wiring[1:, :, :, 0] == 1).mean())
    frac_high = float((high.wiring[1:, :, :, 0] == 1).mean())
    assert frac_high > frac_low
    # roughly on target (not an exact statistical test, just sanity bounds)
    assert 0.03 < frac_low < 0.2
    assert 0.15 < frac_high < 0.45


def test_wiring_seed_decouples_wiring_from_training_seed():
    dilations = [1, 1, 2, 1]
    a = generate_wiring(ModelSpec.custom(channels=64, dilations=dilations, seed=1, wiring_seed=42))
    b = generate_wiring(ModelSpec.custom(channels=64, dilations=dilations, seed=2, wiring_seed=42))
    # same wiring_seed -> identical wiring bits even though the training seed differs
    assert np.array_equal(a.wiring, b.wiring)
    # different wiring_seed -> different wiring
    c = generate_wiring(ModelSpec.custom(channels=64, dilations=dilations, seed=1, wiring_seed=43))
    assert not np.array_equal(a.wiring, c.wiring)


def test_wiring_seed_none_falls_back_to_training_seed():
    dilations = [1, 1, 2, 1]
    a = generate_wiring(ModelSpec.custom(channels=64, dilations=dilations, seed=7, wiring_seed=None))
    b = generate_wiring(ModelSpec.custom(channels=64, dilations=dilations, seed=7, wiring_seed=7))
    assert np.array_equal(a.wiring, b.wiring) and np.array_equal(a.theta, b.theta)
