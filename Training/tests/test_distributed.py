"""T26 acceptance (docs/spec/02-training.md §7/§10 "4GPU検収の具体条件", 04-tasks.md T26):
same-global-batch gradient parity between a 2-process gloo/CPU DDP run and an equivalent
1-process run, and rejection of a non-divisible effectiveBatch. The multiprocess test is skipped
cleanly if torch.multiprocessing.spawn or the gloo backend is unavailable in this environment.
"""

from __future__ import annotations

import json
import os
import socket

import numpy as np
import pytest
import torch

from ichigo_train.config import ConfigError
from ichigo_train.dataset import ShardBuffer, validate_arrays
from ichigo_train.optim import build_scheduler
from ichigo_train.train import resolve_accumulation

S = 9


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_fixture(path, n, seed=0):
    """Like tests/test_train.py's make_fixture, but only an ODD-sized minority of samples have a
    valid "score" label. An odd count can never split evenly across 2 ranks, so a per-rank-local
    (instead of all-reduced-global) denominator would normalise the score term by a different
    amount on each rank -- diverging from a single-process run over the same combined batch. This
    is exactly the bug T26's loss normalisation fix (losses.py, distributed.all_reduce_sum) must
    prevent."""
    rng = np.random.default_rng(seed)
    buf = ShardBuffer(S)
    score_valid = set(range(0, n, 5))
    assert len(score_valid) % 2 == 1, "need an odd count of valid-score samples to force an uneven per-rank split"
    for i in range(n):
        spatial = rng.integers(0, 2, size=S * S * 32).tolist()
        legal = rng.integers(0, 2, size=S * S + 1).tolist(); legal[-1] = 1
        pol = np.zeros(S * S + 1); li = [j for j, l in enumerate(legal) if l]
        pol[rng.choice(li)] = 1.0
        valid = i in score_valid
        buf.add({"spatial": spatial, "global": [0.0, 9 / 19, i / max(n, 1), 0], "legal": legal, "policy": pol.tolist(),
                 "expected_result": float(rng.uniform()), "score": float(rng.normal() * 5) if valid else 0.0,
                 "ownership": rng.uniform(-1, 1, size=S * S).tolist(), "wdl": [0, 0, 0], "mask": [1, 1, 1 if valid else 0, 1, 0],
                 "positionId": f"{i:064x}", "gameId": f"{i:064x}"})
    a = buf.to_arrays()
    validate_arrays(a, S)
    os.makedirs(path, exist_ok=True)
    np.savez(os.path.join(path, "fixture.npz"), **a)
    return score_valid


def write_config(path, data, out, **over):
    cfg = {"schemaVersion": 1, "runId": "ddp-t26", "data": str(data), "out": str(out), "boardSize": S, "profile": "tiny", "seed": 7,
           "device": "cpu", "microBatch": 4, "effectiveBatch": 8, "maxSteps": 100, "validationInterval": 100, "checkpointInterval": 100,
           "fixtureMode": True, "augmentation": "none"}  # augmentation off: independent per-rank aug RNG streams would
    cfg.update(over)                                     # otherwise apply different D4 symmetries than a single-process run.
    with open(path, "w") as f:
        json.dump(cfg, f)
    return str(path)


def _snapshot_params(trainer):
    return {"theta": trainer.model.theta.detach().numpy().copy(),
            "Wlocal": trainer.model.heads["Wlocal"].detach().numpy().copy(),
            "Wpolicy": trainer.model.heads["Wpolicy"].detach().numpy().copy()}


def _use_sgd(trainer):
    """Replaces the Trainer's AdamW with plain SGD (same param groups/LRs), sharing the exact
    same theta/head tensors so train_step()'s forward/backward/DDP path is untouched.

    Why: AdamW divides each parameter's update by (roughly) its own gradient's running RMS. That
    adaptive division amplifies the tiny (~1e-7 relative, inherent and unavoidable) floating-point
    non-associativity between a real 2-rank gloo all-reduce and a single-process sum -- confirmed
    by direct measurement, the raw gradient at step 1 already agrees to ~5e-7 relative, i.e. the
    loss-normalisation fix is correct to floating-point precision -- into parameter differences
    that grow past 1e-5 within a handful of AdamW steps, for parameters whose gradient variance is
    small. This is a documented general property of adaptive optimisers under DDP, not specific to
    this codebase, and would defeat any tight (1e-5) multi-step parameter-comparison test
    regardless of what optimizer.py's production AdamW does correctly. SGD's linear update does
    not have this amplification, so it isolates exactly what this test is meant to check: the loss
    normalisation, not AdamW's numerical sensitivity."""
    trainer.opt = torch.optim.SGD([
        {"params": [trainer.model.theta], "lr": trainer.cfg["gateLearningRate"]},
        {"params": list(trainer.model.heads.values()), "lr": trainer.cfg["headLearningRate"]},
    ])
    trainer.sched = build_scheduler(trainer.opt, trainer.cfg["maxSteps"])


def _ddp_worker(rank, world_size, port, cfg_path, n_steps, out_path):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    from ichigo_train import distributed as D
    from ichigo_train.train import Trainer
    t = Trainer(cfg_path, None)
    assert t.world_size == world_size and t.rank == rank
    _use_sgd(t)
    for _ in range(n_steps):
        t.train_step()
    if t.csv is not None:
        t.csv.close()
    if t.log is not None:
        t.log.close()
    if rank == 0:
        np.savez(out_path, **_snapshot_params(t))
    D.barrier()
    D.cleanup()


def _spawn_available() -> bool:
    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp
        return "spawn" in mp.get_all_start_methods() and dist.is_available() and dist.is_gloo_available()
    except Exception:
        return False


@pytest.mark.skipif(not _spawn_available(), reason="torch.multiprocessing spawn / gloo backend unavailable")
def test_ddp_gradients_match_single_process(tmp_path):
    import torch.multiprocessing as mp

    n = 24
    make_fixture(tmp_path / "fx", n=n, seed=11)
    steps = 3
    port = _free_port()

    # 2 ranks x microBatch 4, accumulation 1 -> effectiveBatch 8 (each optimizer step combines
    # exactly 8 samples across the 2 ranks, disjoint, same permutation as the single-process run).
    # maxSteps stays at write_config's large default (100): it only shapes the tau/freeze schedule
    # and LR warmup, and this test only executes `steps` of them -- using maxSteps==steps would
    # make the prefix-freeze schedule trigger a freeze within these first few steps (freeze timing
    # is proportional to maxSteps), switching some layers to the discontinuous hard-gate path and
    # making the comparison sensitive to argmax ties, which is not what this test is about.
    ddp_cfg = write_config(tmp_path / "ddp.json", tmp_path / "fx", tmp_path / "run-ddp", microBatch=4, effectiveBatch=8)
    ddp_out = tmp_path / "ddp-params.npz"
    mp.spawn(_ddp_worker, args=(2, port, ddp_cfg, steps, str(ddp_out)), nprocs=2, join=True)
    ddp_params = np.load(ddp_out)

    # 1 rank x microBatch 8, accumulation 1 -> same effectiveBatch 8, same 8 samples per step.
    single_cfg = write_config(tmp_path / "single.json", tmp_path / "fx", tmp_path / "run-single", microBatch=8, effectiveBatch=8)
    from ichigo_train.train import Trainer
    ts = Trainer(single_cfg, None)
    assert ts.world_size == 1
    _use_sgd(ts)
    for _ in range(steps):
        ts.train_step()
    single_params = _snapshot_params(ts)
    ts.csv.close(); ts.log.close()

    for name in ("theta", "Wlocal", "Wpolicy"):
        diff = float(np.abs(ddp_params[name] - single_params[name]).max())
        assert diff < 1e-5, f"{name} differs by {diff} between the 2-process DDP run and the 1-process run"


def test_effective_batch_not_divisible_by_microbatch_times_worldsize_is_rejected():
    with pytest.raises(ConfigError, match="divisible"):
        resolve_accumulation(effective_batch=128, micro_batch=8, world_size=3)
    assert resolve_accumulation(128, 8, 4) == 4
    assert resolve_accumulation(128, 32, 4) == 1
    assert resolve_accumulation(128, 8, 1) == 16
