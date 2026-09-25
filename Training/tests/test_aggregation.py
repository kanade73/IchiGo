import numpy as np
import pytest
import torch

from ichigo_train import aggregation as AGG
from ichigo_train.checkpoint import model_from_checkpoint
from ichigo_train.config import ConfigError, load_config
from ichigo_train.model import LogicNet
from ichigo_train.optim import build_optimizer
from ichigo_train.wiring import ModelSpec


def random_spatial(rng, B, S):
    board = rng.integers(-1, 2, size=(B, S, S))
    sp = np.zeros((B, S, S, 32), dtype=np.uint8)
    sp[..., 0] = board == 1
    sp[..., 1] = board == -1
    sp[..., 16] = board == 0
    sp[..., 28] = 1
    sp[..., 2:16] = rng.integers(0, 2, size=(B, S, S, 14))
    return torch.from_numpy(sp)


def flood_components(state):
    S = state.shape[0]
    label = -np.ones((S, S), dtype=int)
    n = 0
    for y in range(S):
        for x in range(S):
            if label[y, x] >= 0:
                continue
            stack = [(y, x)]
            label[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < S and 0 <= nx < S and label[ny, nx] < 0 and state[ny, nx] == state[cy, cx]:
                        label[ny, nx] = n
                        stack.append((ny, nx))
            n += 1
    return label.reshape(-1)


@pytest.mark.parametrize("S", [5, 9, 19])
def test_component_labels_match_flood_fill(S):
    rng = np.random.default_rng(1)
    sp = random_spatial(rng, 6, S)
    labels, sizes, state = AGG.component_labels(sp)
    for b in range(6):
        st = sp[b, ..., 0].numpy().astype(int) - sp[b, ..., 1].numpy().astype(int)
        lab = flood_components(st)
        same_ref = lab[:, None] == lab[None, :]
        got = labels[b].numpy()
        np.testing.assert_array_equal(got[:, None] == got[None, :], same_ref)
        np.testing.assert_array_equal(sizes[b].numpy(), np.bincount(lab)[lab])
        np.testing.assert_array_equal(state[b].numpy(), st.reshape(-1))


@pytest.mark.parametrize("scope", ["chains", "chains+eyes", "all"])
@pytest.mark.parametrize("relaxation", ["prob", "max"])
def test_component_or_scope_hard_exact_and_soft_agrees_on_bits(scope, relaxation):
    rng = np.random.default_rng(2)
    sp = random_spatial(rng, 4, 9)
    labels, sizes, state = AGG.component_labels(sp)
    member = AGG.pooled_mask(labels, sizes, state, scope, 3)
    x = torch.from_numpy(rng.integers(0, 2, size=(4, 9, 9, 5)).astype(np.float32))
    hard = AGG.component_or(x, labels, member, hard=True).reshape(4, 81, 5).numpy()
    soft = AGG.component_or(x, labels, member, hard=False, relaxation=relaxation).reshape(4, 81, 5).numpy()
    flat = x.reshape(4, 81, 5).numpy()
    for b in range(4):
        lab, sz, stt = labels[b].numpy(), sizes[b].numpy(), state[b].numpy()
        for p in range(81):
            pooled = scope == "all" or stt[p] != 0 or (scope == "chains+eyes" and sz[p] <= 3)
            want = flat[b][lab == lab[p]].max(axis=0) if pooled else flat[b][p]
            np.testing.assert_array_equal(hard[b][p], want)
    np.testing.assert_allclose(soft, hard, atol=1e-5)


def test_component_or_gradients_flow_on_chains():
    sp = random_spatial(np.random.default_rng(3), 2, 9)
    labels, sizes, state = AGG.component_labels(sp)
    member = AGG.pooled_mask(labels, sizes, state, "chains", 8)
    x = torch.full((2, 9, 9, 3), 0.3, requires_grad=True)
    for relaxation in ("prob", "max"):
        AGG.component_or(x, labels, member, hard=False, relaxation=relaxation).sum().backward()
        assert x.grad.abs().sum() > 0
        x.grad = None


def test_threshold_hard_matches_bruteforce():
    torch.manual_seed(0)
    layer = AGG.ThresholdLayer(channels=6, nodes=4, fan_in=5, seed=3)
    with torch.no_grad():
        layer.threshold.copy_(torch.tensor([0.5, 2.0, 2.5, 5.0]))
    y = torch.from_numpy(np.random.default_rng(4).integers(0, 2, size=(2, 5, 5, 6)).astype(np.float32))
    out = layer(y, tau=1.0, hard=True).numpy()
    hp = layer.hard_parameters()
    pad = np.pad(y.numpy(), ((0, 0), (1, 1), (1, 1), (0, 0)))
    for b in range(2):
        for yy in range(5):
            for xx in range(5):
                for j in range(4):
                    total = 0
                    for (ch, dx, dy), neg in zip(hp["refs"][j], hp["negate"][j]):
                        v = pad[b, yy + 1 + dy, xx + 1 + dx, ch]
                        total += (1 - v) if neg else v
                    assert out[b, yy, xx, j] == float(total >= hp["threshold"][j])


def aggregated_model(kind):
    spec = ModelSpec.custom(channels=16, dilations=[1, 1, 2, 1], seed=5, profile="tiny")
    cfg = {"type": kind, "afterLayers": [1, 2]}
    cfg.update({"channels": 8, "scope": "chains+eyes"} if kind == "component-or" else {"nodes": 4, "fanIn": 6})
    return LogicNet(spec, head_version=3, aggregation=cfg)


@pytest.mark.parametrize("kind", ["component-or", "threshold"])
def test_all_frozen_soft_path_equals_hard_path(kind):
    model = aggregated_model(kind)
    sp = random_spatial(np.random.default_rng(6), 3, 9)
    glob = torch.zeros(3, 4)
    soft_all_frozen = model.soft_layers(sp, tau=0.2, frozen_prefix=4)
    hard = model.hard_layers(sp)
    for s, h in zip(soft_all_frozen, hard):
        assert torch.equal(s, h.to(torch.float32))
    out = model(sp, glob, tau=1.0)
    out["policy_logits"].sum().backward()
    names = [n for n, q in model.named_parameters() if n.startswith("agg.") and q.grad is not None]
    assert (len(names) > 0) == (kind == "threshold")
    model.forward_hard(sp, glob)


def test_optimizer_groups_and_checkpoint_roundtrip(tmp_path):
    model = aggregated_model("threshold")
    opt = build_optimizer(model, 0.1, 0.001, 1e-4)
    gate_ids = {id(q) for q in opt.param_groups[0]["params"]}
    assert all(id(q) in gate_ids for n, q in model.named_parameters() if n.startswith("agg."))
    ck = {"checkpointVersion": 2, "spec": {"channels": 16, "dilations": [1, 1, 2, 1], "seed": 5, "profile": "tiny",
                                           "bank1_ratio": 0.1, "wiring_seed": None, "gate_arity": 2},
          "headVersion": 3, "wiring": model.wiring_numpy(), "theta": model.theta.detach().numpy(), "heads": model.head_numpy(),
          "config": {"aggregation": model.aggregation_config}, "stateDict": model.state_dict()}
    again = model_from_checkpoint(ck)
    sp = random_spatial(np.random.default_rng(7), 2, 9)
    a, b = model.forward_hard(sp, torch.zeros(2, 4)), again.forward_hard(sp, torch.zeros(2, 4))
    assert torch.equal(a["policy_logits"], b["policy_logits"])


def test_export_refuses_aggregation(tmp_path):
    from ichigo_train.export import export_model
    with pytest.raises(ValueError, match="aggregation"):
        export_model(aggregated_model("component-or"), str(tmp_path / "m.ichigo"), [9])


def test_config_validation(tmp_path):
    import json
    base = {"data": str(tmp_path), "out": str(tmp_path / "o"), "boardSize": 9, "runId": "r", "profile": "small"}
    good = dict(base, aggregation={"type": "threshold", "afterLayers": [2, 5], "nodes": 32, "fanIn": 16})
    p = tmp_path / "c.json"
    p.write_text(json.dumps(good))
    load_config(str(p))
    for bad in ({"type": "sum", "afterLayers": [1]}, {"type": "component-or", "afterLayers": [8]},
                {"type": "component-or", "afterLayers": [1], "channels": 999}, {"type": "threshold", "afterLayers": [1], "extra": 1},
                {"type": "component-or", "afterLayers": [1], "scope": "board"}, {"type": "threshold", "afterLayers": [1], "weights": "float"}):
        p.write_text(json.dumps(dict(base, aggregation=bad)))
        with pytest.raises(ConfigError):
            load_config(str(p))


@pytest.mark.parametrize("kind", ["component-or", "threshold"])
def test_training_run_and_resume_with_aggregation(tmp_path, kind):
    from test_train import make_fixture, write_config
    from ichigo_train.train import Trainer
    make_fixture(tmp_path / "fx")
    agg = {"type": kind, "afterLayers": [1, 2]}
    agg.update({"channels": 32} if kind == "component-or" else {"nodes": 8, "fanIn": 6})
    cfg = write_config(tmp_path / "a.json", tmp_path / "fx", tmp_path / "run", maxSteps=20, validationInterval=10, checkpointInterval=10,
                       aggregation=agg)
    t = Trainer(cfg, None)
    assert t.run() == 0
    t2 = Trainer(cfg, str(tmp_path / "run" / "checkpoint-latest.pt"))
    assert t2.model.aggregation_config == t.model.aggregation_config
    sp = random_spatial(np.random.default_rng(8), 2, 9)
    a = t.model.forward_hard(sp, torch.zeros(2, 4))["policy_logits"]
    b = t2.model.forward_hard(sp, torch.zeros(2, 4))["policy_logits"]
    assert torch.equal(a, b)


@pytest.mark.parametrize("weights,output_ste", [("tanh", False), ("ste", False), ("ste", True)])
def test_threshold_soft_hard_gap(weights, output_ste):
    """STE weights make the soft forward use sign(w); with outputSte it equals the hard forward."""
    layer = AGG.ThresholdLayer(channels=8, nodes=6, fan_in=7, seed=9, weights=weights, output_ste=output_ste)
    y = torch.from_numpy(np.random.default_rng(10).integers(0, 2, size=(3, 9, 9, 8)).astype(np.float32))
    hard = layer(y, tau=1.0, hard=True)
    soft = layer(y, tau=0.05, hard=False)
    if output_ste:
        assert torch.equal(soft.detach(), hard)
    elif weights == "ste":
        assert torch.equal((soft.detach() > 0.5).float(), hard)
    soft.sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
