"""BaselineCNN (docs/spec/04-tasks.md T33): shape/interface parity with LogicNet, and that the
zero-size ``theta`` doesn't crash the code paths train.py exercises on ``model.theta`` (optimizer,
gradient clipping, gate statistics) even though this model has no gates at all."""

import numpy as np
import torch

from ichigo_train import losses as L
from ichigo_train.baseline_cnn import BaselineCNN
from ichigo_train.metrics import gate_statistics
from ichigo_train.model import HEAD_TENSOR_NAMES, global_input_size, head_shapes
from ichigo_train.optim import build_optimizer, clip_gradients


def make_batch(board_size: int, batch: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    S = board_size
    legal = torch.ones(batch, S * S + 1, dtype=torch.uint8)
    policy = torch.zeros(batch, S * S + 1)
    policy[:, 0] = 1.0
    return {
        "legal": legal, "policy": policy,
        "expected_result": torch.full((batch,), 0.5),
        "wdl": torch.tensor(rng.dirichlet([1, 1, 1], size=batch), dtype=torch.float32),
        "score": torch.zeros(batch), "ownership": torch.zeros(batch, S * S),
        "target_mask": torch.ones(batch, 5), "sample_weight": torch.ones(batch),
    }


def test_head_tensor_shapes_match_head_shapes():
    m = BaselineCNN(channels=32, num_blocks=2, seed=1, head_version=2)
    shapes = head_shapes(32, 2)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]
    assert m.head_version == 2
    assert m.channels == 32
    assert m.dilations == []
    assert m.spec.layers == 0
    assert m.spec.channels == 32


def test_forward_and_forward_hard_shapes_9_and_19():
    m = BaselineCNN(channels=16, num_blocks=2, seed=2, head_version=2)
    for S, B in ((9, 3), (19, 2)):
        sp = torch.from_numpy(np.random.default_rng(S).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
        g = torch.zeros(B, 4)
        soft = m.forward(sp, g, tau=1.0, frozen_prefix=0)
        hard = m.forward_hard(sp, g)
        for out in (soft, hard):
            assert out["policy_logits"].shape == (B, S * S + 1)
            assert out["wdl_logits"].shape == (B, 3)
            assert out["score_mean"].shape == (B,)
            assert out["ownership"].shape == (B, S * S)
        # no discrete gates: hard and soft are the exact same computation
        assert torch.equal(soft["policy_logits"], hard["policy_logits"])


def test_head_version_1_matches_v1_shapes():
    m = BaselineCNN(channels=8, num_blocks=1, seed=3, head_version=1)
    shapes = head_shapes(8, 1)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]


def test_baseline_cnn_head_version_2_unaffected_by_headv3_addition():
    """Adding headVersion 3 (docs/spec/01-network.md §4, zreg) must not change BaselineCNN's
    default (headVersion 2) shapes, tensor count, or global_input_size formula."""
    m = BaselineCNN(channels=8, num_blocks=1, seed=7, head_version=2)
    shapes = head_shapes(8, 2)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]
    assert m.head_version == 2
    assert global_input_size(8, 2) == 2 * 8 + 64 + 1 + 4 == 85


def test_baseline_cnn_head_version_3_also_works_via_shared_head_shapes():
    """BaselineCNN has no headVersion-specific code of its own -- it reuses model.head_shapes /
    LogicNet.heads_forward, so headVersion 3 works automatically once SUPPORTED_HEAD_VERSIONS
    includes it. Not required by 04-tasks.md T33 (the CNN baseline is a diagnostic for the
    logic-gate network specifically), but nothing should stop it."""
    m = BaselineCNN(channels=8, num_blocks=1, seed=8, head_version=3)
    shapes = head_shapes(8, 3)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]
    sp = torch.from_numpy(np.random.default_rng(9).integers(0, 2, size=(2, 9, 9, 32)).astype(np.uint8))
    out = m.forward(sp, torch.zeros(2, 4))
    assert out["wdl_logits"].shape == (2, 3)


def test_zero_size_theta_and_empty_gates():
    m = BaselineCNN(channels=8, num_blocks=1, seed=4)
    assert m.theta.numel() == 0
    assert m.wiring_numpy().shape == (0, 0, 2, 4) and m.wiring_numpy().dtype == np.int32
    assert m.hard_gates().shape == (0, 0) and m.hard_gates().dtype == np.uint8
    assert m.hard_layers(None) == []
    heads_np = m.head_numpy()
    for n in HEAD_TENSOR_NAMES:
        assert n in heads_np


def test_optimizer_and_clip_gradients_do_not_crash_on_zero_size_theta():
    m = BaselineCNN(channels=8, num_blocks=1, seed=5, head_version=2)
    opt = build_optimizer(m, gate_lr=0.01, head_lr=0.001, head_wd=0.0001)
    S, B = 9, 4
    sp = torch.from_numpy(np.random.default_rng(1).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
    out = m.forward(sp, torch.zeros(B, 4))
    r = L.compute_losses(out, make_batch(S, B), S)
    opt.zero_grad(set_to_none=True)
    r["total"].backward()
    gnorm = clip_gradients(m, 1.0)
    assert np.isfinite(gnorm)
    opt.step()
    assert m.theta.grad is None  # never enters the forward graph


def test_gate_statistics_does_not_crash():
    m = BaselineCNN(channels=8, num_blocks=1, seed=6)
    sample = torch.from_numpy(np.random.default_rng(2).integers(0, 2, size=(4, 9, 9, 32)).astype(np.uint8))
    stats = gate_statistics(m, sample)
    assert stats["layers"] == []
    assert stats["gateHistogram"] == [0] * 16
    stats_none = gate_statistics(m, None)
    assert stats_none["layers"] == [] and stats_none["gateHistogram"] == [0] * 16
