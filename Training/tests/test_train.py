import json
import os

import numpy as np
import pytest
import torch

from ichigo_train.config import ConfigError, load_config
from ichigo_train.dataset import ShardBuffer, validate_arrays
from ichigo_train.train import Trainer

S = 9


def make_fixture(path, n=32, seed=0):
    rng = np.random.default_rng(seed)
    buf = ShardBuffer(S)
    for i in range(n):
        spatial = rng.integers(0, 2, size=S * S * 32).tolist()
        legal = rng.integers(0, 2, size=S * S + 1).tolist(); legal[-1] = 1
        pol = np.zeros(S * S + 1); li = [j for j, l in enumerate(legal) if l]
        pol[rng.choice(li)] = 1.0
        buf.add({"spatial": spatial, "global": [0.0, 9 / 19, i / 162, 0], "legal": legal, "policy": pol.tolist(), "expected_result": float(rng.uniform()),
                 "score": float(rng.normal() * 5), "ownership": rng.uniform(-1, 1, size=S * S).tolist(), "wdl": [0, 0, 0], "mask": [1, 1, 1, 1, 0],
                 "positionId": f"{i:064x}", "gameId": f"{i:064x}"})
    a = buf.to_arrays()
    validate_arrays(a, S)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "fixture.npz"), **a)


def write_config(path, data, out, **over):
    cfg = {"schemaVersion": 1, "runId": "t", "data": str(data), "out": str(out), "boardSize": 9, "profile": "tiny", "seed": 1, "device": "cpu",
           "microBatch": 4, "effectiveBatch": 8, "maxSteps": 10, "validationInterval": 5, "checkpointInterval": 5, "fixtureMode": True,
           "augmentation": "d4"}
    cfg.update(over)
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def snapshot(trainer):
    return {"theta": trainer.model.theta.detach().clone(), "W": trainer.model.heads["Wlocal"].detach().clone(),
            "step": trainer.step, "sampler": trainer.sampler.state(), "aug": trainer.aug_rng.bit_generator.state["state"]["state"]}


def test_resume_is_bit_identical(tmp_path):
    make_fixture(tmp_path / "fx")
    cfgA = write_config(tmp_path / "a.json", tmp_path / "fx", tmp_path / "runA")
    ta = Trainer(cfgA, None)
    for _ in range(10):
        ta.train_step()
    a = snapshot(ta)
    cfgB = write_config(tmp_path / "b.json", tmp_path / "fx", tmp_path / "runB")
    tb = Trainer(cfgB, None)
    for _ in range(5):
        tb.train_step()
    tb._save("checkpoint-latest.pt")
    tb.csv.close(); tb.log.close()
    tc = Trainer(cfgB, str(tmp_path / "runB" / "checkpoint-latest.pt"))
    assert tc.step == 5
    for _ in range(5):
        tc.train_step()
    c = snapshot(tc)
    assert c["step"] == 10 and torch.equal(a["theta"], c["theta"]) and torch.equal(a["W"], c["W"])
    assert a["sampler"] == c["sampler"] and a["aug"] == c["aug"]
    for t in (ta, tc):
        t.csv.close(); t.log.close()


def test_resume_rejects_other_dataset_and_wiring(tmp_path):
    make_fixture(tmp_path / "fx")
    make_fixture(tmp_path / "fx2", seed=5)
    cfg = write_config(tmp_path / "a.json", tmp_path / "fx", tmp_path / "runA")
    t = Trainer(cfg, None)
    t.train_step(); t._save("checkpoint-latest.pt"); t.csv.close(); t.log.close()
    cfg2 = write_config(tmp_path / "b.json", tmp_path / "fx2", tmp_path / "runB")
    with pytest.raises(ConfigError, match="dataset"):
        Trainer(cfg2, str(tmp_path / "runA" / "checkpoint-latest.pt"))
    cfg3 = write_config(tmp_path / "c.json", tmp_path / "fx", tmp_path / "runC", seed=2)
    with pytest.raises(ConfigError, match="wiring"):
        Trainer(cfg3, str(tmp_path / "runA" / "checkpoint-latest.pt"))


def test_prefix_freeze_stops_theta_updates_and_exports_hard(tmp_path):
    make_fixture(tmp_path / "fx")
    cfg = write_config(tmp_path / "a.json", tmp_path / "fx", tmp_path / "run", maxSteps=20, validationInterval=20, checkpointInterval=20)
    t = Trainer(cfg, None)
    sched = t.schedule
    fs = sched.freeze_steps()  # 20 steps, 4 layers: s1_end=12, s2_end=18
    assert fs == [14, 15, 16, 18]  # round(12+6*k/4): 13.5->14, 15, 16.5->16 (banker), 18
    thetas = {}
    while t.step < 20:
        st = sched.state_at(t.step)
        before = t.model.theta.detach().clone()
        r = t.train_step()
        after = t.model.theta.detach()
        # frozen prefix never changes, unfrozen layers do change (when not heads-only)
        if st.frozen_prefix:
            assert torch.equal(before[: st.frozen_prefix], after[: st.frozen_prefix])
        if not st.heads_only:
            assert not torch.equal(before[st.frozen_prefix:], after[st.frozen_prefix:])
        else:
            assert torch.equal(before, after)
        assert r["frozenPrefix"] == st.frozen_prefix
    assert sched.state_at(t.step).heads_only
    rc = t.run.__func__  # noqa: just ensure attribute exists
    t._summary([0.1])
    summary = json.load(open(tmp_path / "run" / "run-summary.json"))
    assert summary["finalGatesAllFrozen"] and summary["frozenPrefix"] == 4
    # export uses argmax of theta: all 0/1 gates
    from ichigo_train.export import export_model
    from ichigo_train.model_format import read_model
    export_model(t.model, str(tmp_path / "m.ichigo"), [9])
    m = read_model(str(tmp_path / "m.ichigo"))
    assert m.gates.max() <= 15 and m.gates.shape == (4, 64)
    t.csv.close(); t.log.close()


def test_full_run_writes_artifacts_and_validation(tmp_path):
    make_fixture(tmp_path / "fx")
    cfg = write_config(tmp_path / "a.json", tmp_path / "fx", tmp_path / "run", maxSteps=6, validationInterval=3, checkpointInterval=3)
    t = Trainer(cfg, None)
    assert t.run() == 0
    out = tmp_path / "run"
    for f in ["config.json", "resolved-config.json", "environment.json", "metrics.csv", "training.log", "checkpoint-latest.pt", "checkpoint-best-hard.pt", "wiring.npz", "run-summary.json"]:
        assert (out / f).exists(), f
    vals = sorted(os.listdir(out / "validation"))
    assert any("initial" in v for v in vals) and len(vals) >= 3
    rec = json.load(open(out / "validation" / vals[-1]))
    assert set(rec) >= {"soft", "hard", "gates", "softHardPolicyCEDiff"}
    assert rec["hard"]["policyTop1"] <= 1 and len(rec["gates"]["layers"]) == 4
    summary = json.load(open(out / "run-summary.json"))
    assert summary["completed"] and summary["bestHard"]["step"] in (0, 3, 6)


def test_config_rejects_unknown_and_missing(tmp_path):
    p = tmp_path / "c.json"
    json.dump({"schemaVersion": 1, "runId": "x", "data": "d", "out": "o", "boardSize": 9, "bogus": 1}, open(p, "w"))
    with pytest.raises(ConfigError, match="unknown"):
        load_config(str(p))
    json.dump({"schemaVersion": 1, "runId": "x", "data": "d", "boardSize": 9}, open(p, "w"))
    with pytest.raises(ConfigError, match="missing"):
        load_config(str(p))
    json.dump({"schemaVersion": 1, "runId": "x", "data": "d", "out": "o", "boardSize": 9, "microBatch": 3, "effectiveBatch": 8}, open(p, "w"))
    with pytest.raises(ConfigError, match="divisible"):
        load_config(str(p))
