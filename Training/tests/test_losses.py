import math

import numpy as np
import torch

from ichigo_train import losses as L
from ichigo_train.model import build_model
from ichigo_train.optim import build_optimizer, build_scheduler, clip_gradients, lr_multiplier


def _batch(B=2, S=9):
    P = S * S + 1
    legal = torch.ones(B, P, dtype=torch.uint8)
    legal[0, 5] = 0
    policy = torch.zeros(B, P)
    policy[0, 0] = 0.5; policy[0, 1] = 0.5
    policy[1, 3] = 1.0
    return {
        "legal": legal, "policy": policy, "expected_result": torch.tensor([0.8, 0.3]),
        "score": torch.tensor([9.0, -4.5]), "ownership": torch.zeros(B, S, S), "wdl": torch.tensor([[1.0, 0, 0], [0, 0, 1.0]]),
        "target_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.uint8), "sample_weight": torch.tensor([1.0, 3.0]),
    }


def test_hand_computed_values():
    b = _batch()
    B, P = 2, 82
    out = {"policy_logits": torch.zeros(B, P), "wdl_logits": torch.zeros(B, 3), "score_mean": torch.tensor([0.0, 0.0]), "ownership": torch.full((B, 81), 0.5)}
    r = L.compute_losses(out, b, 9)
    # sample 0: 81 legal moves uniform -> log p = -log 81 for both targets
    lp0 = math.log(81); lp1 = math.log(82)
    assert abs(r["policy"].item() - (1 * lp0 + 3 * lp1) / 4) < 1e-5
    # expected: e = 1/3 + 0.5/3 = 0.5
    le = -math.log(0.5)
    assert abs(r["expected_result"].item() - le) < 1e-5
    # wdl only sample 0 valid: -log(1/3)
    assert abs(r["wdl"].item() - math.log(3)) < 1e-5
    # score huber of (0-9)/9=-1 -> 0.5 ; (0+4.5)/9=0.5 -> 0.125 ; weighted (1*0.5 + 3*0.125)/4
    assert abs(r["score"].item() - (0.5 + 0.375) / 4) < 1e-6
    assert abs(r["ownership"].item() - 0.25) < 1e-6
    total = r["policy"] + r["expected_result"] + 0.5 * r["wdl"] + 0.25 * r["score"] + 0.25 * r["ownership"]
    assert abs(r["total"].item() - total.item()) < 1e-6


def test_all_masks_zero_gives_zero_not_nan():
    b = _batch()
    b["target_mask"] = torch.zeros_like(b["target_mask"])
    out = {"policy_logits": torch.randn(2, 82, requires_grad=True), "wdl_logits": torch.randn(2, 3, requires_grad=True),
           "score_mean": torch.randn(2, requires_grad=True), "ownership": torch.randn(2, 81, requires_grad=True)}
    r = L.compute_losses(out, b, 9)
    assert r["total"].item() == 0 and all(torch.isfinite(v).all() for v in r.values())
    r["total"].backward()
    assert torch.isfinite(out["policy_logits"].grad).all()


def test_illegal_logits_do_not_produce_nan():
    b = _batch()
    out = {"policy_logits": torch.full((2, 82), 50.0), "wdl_logits": torch.zeros(2, 3), "score_mean": torch.zeros(2), "ownership": torch.zeros(2, 81)}
    out["policy_logits"][0, 5] = 1e4  # illegal, huge
    r = L.compute_losses(out, b, 9)
    assert torch.isfinite(r["total"])


def test_one_step_updates_theta_and_heads():
    m = build_model("tiny")
    opt = build_optimizer(m, 0.01, 0.001, 1e-4)
    sched = build_scheduler(opt, 100)
    sp = torch.from_numpy(np.random.default_rng(0).integers(0, 2, size=(2, 9, 9, 32)).astype(np.uint8))
    b = _batch()
    out = m(sp, torch.zeros(2, 4), tau=1.0)
    r = L.compute_losses(out, b, 9)
    theta0 = m.theta.detach().clone(); w0 = m.heads["Wlocal"].detach().clone()
    r["total"].backward()
    n = clip_gradients(m, 1.0)
    assert n > 0
    opt.step(); sched.step()
    assert not torch.equal(theta0, m.theta) and not torch.equal(w0, m.heads["Wlocal"])


def test_lr_schedule_shape():
    ms = 1000
    assert lr_multiplier(0, ms) == 1 / 50 and abs(lr_multiplier(49, ms) - 1.0) < 1e-9
    assert abs(lr_multiplier(ms - 1, ms) - 0.1) < 2e-3 and lr_multiplier(500, ms) < 1.0


def _mixed_mask_batch(B, S, valid_score_idx):
    """B samples, every term valid on every sample except "score", which is valid on exactly
    ``valid_score_idx`` -- the classic case a per-microbatch-local-mean normalisation gets wrong."""
    P = S * S + 1
    rng = np.random.default_rng(2)
    legal = torch.ones(B, P, dtype=torch.uint8)
    policy = torch.zeros(B, P)
    for i in range(B):
        policy[i, rng.integers(0, P)] = 1.0
    mask = torch.ones(B, 5, dtype=torch.uint8)
    mask[:, 2] = 0
    mask[valid_score_idx, 2] = 1
    score = torch.zeros(B)
    score[valid_score_idx] = 3.5
    return {
        "legal": legal, "policy": policy, "expected_result": torch.rand(B), "score": score,
        "ownership": torch.zeros(B, S, S), "wdl": torch.zeros(B, 3), "target_mask": mask, "sample_weight": torch.ones(B),
    }


def test_microbatch_accumulation_normalises_over_whole_step():
    """T26 (docs/spec/02-training.md §7): an optimizer step's microbatches must be normalised by
    the valid-weight sum over the WHOLE accumulation window, not each microbatch's own local sum
    (which is wrong whenever the valid mask differs between microbatches -- here only one of 16
    samples has a valid score label). A 2-microbatch step using the global denominator (and no
    /accumulation division on the backward, per losses.py's docstring) must match a single-batch
    step over the same 16 samples within 1e-6; the naive per-microbatch-local-mean approach must
    NOT match (otherwise this test would not be exercising the bug it guards against)."""
    torch.manual_seed(0)
    m = build_model("tiny")
    S = 9
    B = 16
    sp = torch.from_numpy(np.random.default_rng(1).integers(0, 2, size=(B, S, S, 32)).astype(np.uint8))
    glob = torch.zeros(B, 4)
    batch = _mixed_mask_batch(B, S, valid_score_idx=3)  # sample 3 lands in the first microbatch (0:8)

    m.zero_grad(set_to_none=True)
    out = m(sp, glob, tau=1.0)
    r = L.compute_losses(out, batch, S)
    r["total"].backward()
    theta_single = m.theta.grad.clone(); w_single = m.heads["Wlocal"].grad.clone()

    splits = [slice(0, 8), slice(8, 16)]
    microbatches = [{k: v[sl] for k, v in batch.items()} for sl in splits]
    global_sums = {name: torch.zeros(()) for name in L.TERM_NAMES}
    for mb in microbatches:
        for name, v in L.local_weight_sums(mb).items():
            global_sums[name] = global_sums[name] + v

    m.zero_grad(set_to_none=True)
    for sl, mb in zip(splits, microbatches):
        out_mb = m(sp[sl], glob[sl], tau=1.0)
        r_mb = L.compute_losses(out_mb, mb, S, weight_denominators=global_sums, numerator_scale=1.0)
        r_mb["total"].backward()
    theta_fixed = m.theta.grad.clone(); w_fixed = m.heads["Wlocal"].grad.clone()

    assert torch.allclose(theta_single, theta_fixed, atol=1e-6)
    assert torch.allclose(w_single, w_fixed, atol=1e-6)

    # the old (buggy) approach: each microbatch normalises by its own local sum, and the two
    # losses are averaged by dividing by the accumulation count.
    m.zero_grad(set_to_none=True)
    for sl, mb in zip(splits, microbatches):
        out_mb = m(sp[sl], glob[sl], tau=1.0)
        r_mb = L.compute_losses(out_mb, mb, S)  # local (per-microbatch) denominator
        (r_mb["total"] / len(splits)).backward()
    theta_buggy = m.theta.grad.clone()
    assert not torch.allclose(theta_single, theta_buggy, atol=1e-6)
