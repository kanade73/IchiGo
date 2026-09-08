"""model_factory.build_model_from_config (docs/spec/04-tasks.md T33): a config dict selects and
parametrises either LogicNet or BaselineCNN. Mirrors what configs/experiments/*.json (the T33
capacity/wiring/CNN-baseline matrix) is expected to drive once train.py is wired to this factory
(see the report's diff -- not applied to train.py here)."""

import numpy as np
import pytest
import torch

from ichigo_train import losses as L
from ichigo_train.baseline_cnn import BaselineCNN
from ichigo_train.model import HEAD_TENSOR_NAMES, LogicNet, head_shapes
from ichigo_train.model_factory import build_model_from_config, model_spec_from_config
from ichigo_train.optim import build_optimizer, clip_gradients
from ichigo_train.wiring import PROFILES


def base_cfg(**over) -> dict:
    cfg = {"profile": "small", "seed": 20260908}
    cfg.update(over)
    return cfg


def make_batch(board_size: int, batch: int) -> dict:
    S = board_size
    legal = torch.ones(batch, S * S + 1, dtype=torch.uint8)
    policy = torch.zeros(batch, S * S + 1)
    policy[:, 0] = 1.0
    return {
        "legal": legal, "policy": policy, "expected_result": torch.full((batch,), 0.5),
        "wdl": torch.tensor([[0.34, 0.33, 0.33]] * batch, dtype=torch.float32),
        "score": torch.zeros(batch), "ownership": torch.zeros(batch, S * S),
        "target_mask": torch.ones(batch, 5), "sample_weight": torch.ones(batch),
    }


# ---- logic ----

def test_logic_default_matches_profile():
    cfg = base_cfg()
    m = build_model_from_config(cfg)
    assert isinstance(m, LogicNet)
    small = PROFILES["small"]
    assert m.channels == small["channels"]
    assert m.dilations == list(small["dilations"])
    assert m.head_version == 2  # model_factory's default, matches model.HEAD_VERSION
    shapes = head_shapes(small["channels"], 2)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]


def test_logic_channels_and_dilations_overrides():
    cfg = base_cfg(channels=512, dilations=[1, 1, 1, 1, 1, 1, 1, 1], headVersion=1)
    m = build_model_from_config(cfg)
    assert m.channels == 512
    assert m.dilations == [1, 1, 1, 1, 1, 1, 1, 1]
    assert m.head_version == 1
    shapes = head_shapes(512, 1)
    for n in HEAD_TENSOR_NAMES:
        assert tuple(m.heads[n].shape) == shapes[n]


def test_logic_wiring_seed_and_bank1_ratio_reach_the_spec():
    cfg = base_cfg(wiringSeed=7, bank1Ratio=0.3)
    m = build_model_from_config(cfg)
    assert m.spec.wiring_seed == 7
    assert m.spec.bank1_ratio == 0.3
    # same wiringSeed, different training seed -> same wiring
    m2 = build_model_from_config(base_cfg(seed=1, wiringSeed=7, bank1Ratio=0.3))
    assert np.array_equal(m.wiring_numpy(), m2.wiring_numpy())


def test_model_spec_from_config_unknown_profile_raises():
    with pytest.raises(ValueError):
        model_spec_from_config(base_cfg(profile="huge"))


def test_model_spec_from_config_unknown_model_type_raises():
    with pytest.raises(ValueError):
        build_model_from_config(base_cfg(modelType="mystery"))


# ---- cnn-baseline ----

def test_cnn_baseline_forward_shapes_9_and_19():
    cfg = base_cfg(modelType="cnn-baseline", headVersion=2)
    m = build_model_from_config(cfg)
    assert isinstance(m, BaselineCNN)
    for S, B in ((9, 3), (19, 2)):
        sp = torch.from_numpy(np.random.default_rng(S).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
        g = torch.zeros(B, 4)
        out = m.forward(sp, g)
        assert out["policy_logits"].shape == (B, S * S + 1)
        assert out["wdl_logits"].shape == (B, 3)
        assert out["score_mean"].shape == (B,)
        assert out["ownership"].shape == (B, S * S)
        hard = m.forward_hard(sp, g)
        assert torch.equal(out["policy_logits"], hard["policy_logits"])


def test_cnn_baseline_channels_override():
    m = build_model_from_config(base_cfg(modelType="cnn-baseline", channels=32))
    assert m.channels == 32


def test_cnn_baseline_optimizer_builds_and_backward_step_runs():
    cfg = base_cfg(modelType="cnn-baseline", channels=16, headVersion=2)
    m = build_model_from_config(cfg)
    opt = build_optimizer(m, gate_lr=0.01, head_lr=0.001, head_wd=0.0001)
    S, B = 9, 4
    sp = torch.from_numpy(np.random.default_rng(3).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
    out = m.forward(sp, torch.zeros(B, 4))
    r = L.compute_losses(out, make_batch(S, B), S)
    opt.zero_grad(set_to_none=True)
    r["total"].backward()
    gnorm = clip_gradients(m, 1.0)
    assert np.isfinite(gnorm) and gnorm >= 0
    opt.step()


def test_logic_optimizer_builds_and_backward_step_runs():
    cfg = base_cfg(profile="tiny")
    m = build_model_from_config(cfg)
    opt = build_optimizer(m, gate_lr=0.01, head_lr=0.001, head_wd=0.0001)
    S, B = 9, 4
    sp = torch.from_numpy(np.random.default_rng(4).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
    out = m.forward(sp, torch.zeros(B, 4), tau=1.0)
    r = L.compute_losses(out, make_batch(S, B), S)
    opt.zero_grad(set_to_none=True)
    r["total"].backward()
    gnorm = clip_gradients(m, 1.0)
    assert np.isfinite(gnorm) and gnorm >= 0
    opt.step()
