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


# ---- production-ordering regression coverage -----------------------------------------------
#
# The two tests above construct a Trainer (which already runs the full DDP-construction path in
# __init__) but only ever call train_step() directly -- never run(), so they never exercise
# validate()'s or the checkpoint interval's barrier()/broadcast_object() sync points, and never hit
# an interval or a prefix-freeze event. A real torchrun launch always goes through run(): an initial
# validate("initial") before the first optimizer step, then interval-triggered validate()/checkpoint
# calls (and possibly a freeze-triggered validate()) interleaved with train_step(), then the final
# summary/cleanup. test_ddp_full_run_hits_validation_and_checkpoint_intervals below reproduces that
# exact ordering with 2 real (CPU/gloo) processes; test_trainer_wraps_ddp_before_any_validation_collective
# checks the specific invariant documented in train.py's module docstring and
# Trainer._verify_ddp_param_consistency -- DDP wraps the model before any other collective that
# depends on the model matching across ranks -- directly, with a fake (monkeypatched) process group.


def _run_worker(rank, world_size, port, cfg_path, status_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    from ichigo_train.train import Trainer
    t = Trainer(cfg_path, None)
    is_main, out_dir = t.is_main, t.out
    csv_is_none, log_is_none = t.csv is None, t.log is None
    rc = t.run()
    with open(os.path.join(status_dir, f"status-{rank}.json"), "w") as f:
        json.dump({"rank": rank, "returnCode": rc, "isMain": is_main, "out": out_dir,
                   "csvIsNone": csv_is_none, "logIsNone": log_is_none}, f)


@pytest.mark.skipif(not _spawn_available(), reason="torch.multiprocessing spawn / gloo backend unavailable")
def test_ddp_full_run_hits_validation_and_checkpoint_intervals(tmp_path):
    """Real end-to-end reproduction of the production ordering (2 spawned CPU/gloo processes running
    the actual Trainer.run(), not just train_step()): maxSteps/validationInterval/checkpointInterval
    are chosen so a normal interval validate()+checkpoint AND a prefix-freeze-triggered validate()
    both fire within the run (in addition to the initial validate("initial")), and throughputReference
    points at a 1-process run-summary.json produced first, exactly as docs/spec/02-training.md's T26
    acceptance procedure and this module's train_ddp.sh wrapper describe. Both ranks must finish;
    rank 0 alone must write run-summary.json (worldSize 2, plus the throughput scaling fields); every
    other rank's Trainer must never have opened a csv/log file (the is_main gate this module's
    docstring documents) -- the direct, race-free way to check "non-zero ranks write no files", since
    every rank shares the same `out` directory (as a real torchrun launch does: one config, one out
    path) and rank 0 legitimately populates it."""
    import torch.multiprocessing as mp

    n = 24
    make_fixture(tmp_path / "fx", n=n, seed=13)

    # 1-process reference run, produced first, so the DDP run's throughputReference resolves.
    ref_cfg = write_config(tmp_path / "ref.json", tmp_path / "fx", tmp_path / "run-ref",
                            microBatch=8, effectiveBatch=8, maxSteps=2, validationInterval=2, checkpointInterval=2)
    from ichigo_train.train import Trainer
    ref_trainer = Trainer(ref_cfg, None)
    assert ref_trainer.run() == 0
    ref_summary = tmp_path / "run-ref" / "run-summary.json"
    assert ref_summary.exists()

    steps, interval = 4, 2  # both validationInterval and checkpointInterval hit within maxSteps
    ddp_cfg = write_config(tmp_path / "ddp.json", tmp_path / "fx", tmp_path / "run-ddp",
                            microBatch=4, effectiveBatch=8, maxSteps=steps, validationInterval=interval,
                            checkpointInterval=interval, maxValidationPositions=16,
                            throughputReference=str(ref_summary))
    port = _free_port()
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    mp.spawn(_run_worker, args=(2, port, ddp_cfg, str(status_dir)), nprocs=2, join=True)

    statuses = {}
    for r in range(2):
        with open(status_dir / f"status-{r}.json") as f:
            statuses[r] = json.load(f)

    assert statuses[0]["returnCode"] == 0 and statuses[1]["returnCode"] == 0, statuses
    assert statuses[0]["isMain"] and not statuses[1]["isMain"]
    # rank 0 opened csv/log for writing; every other rank never did.
    assert statuses[0]["csvIsNone"] is False and statuses[0]["logIsNone"] is False
    assert statuses[1]["csvIsNone"] is True and statuses[1]["logIsNone"] is True

    out = tmp_path / "run-ddp"
    for f in ("config.json", "resolved-config.json", "environment.json", "metrics.csv", "training.log",
              "checkpoint-latest.pt", "checkpoint-best-hard.pt", "wiring.npz", "run-summary.json"):
        assert (out / f).exists(), f
    vals = sorted(os.listdir(out / "validation"))
    assert any("initial" in v for v in vals) and len(vals) >= 2  # initial + at least one interval/freeze hit

    summary = json.load(open(out / "run-summary.json"))
    assert summary["worldSize"] == 2
    assert summary["completed"] and summary["stepsTrained"] == steps
    th = summary["throughput"]
    assert th["worldSize"] == 2
    assert sorted(d["rank"] for d in th["perRank"]) == [0, 1]
    assert th["throughputReferenceRun"] == str(ref_summary)
    assert "throughputRatio" in th and "scalingEfficiency" in th
    assert th["scalingEfficiency"] == pytest.approx(th["throughputRatio"] / 2)


@pytest.mark.skipif(not _spawn_available(), reason="torch.multiprocessing spawn / gloo backend unavailable")
def test_ddp_full_run_crosses_into_heads_only_stage(tmp_path):
    """Regression test for the "Expected to mark a variable ready only once" DDP crash that hit
    real 2-GPU runs at the first optimizer step of stage 3 (every gate layer frozen -- schedule
    state ``frozen_prefix == layers``, ``heads_only == True``).

    Root cause: ``train_step`` unconditionally added ``gate_entropy_loss(model.theta, ...)`` into
    the per-microbatch loss that gets ``backward()``-ed, but that term depends on ``model.theta``
    directly -- *outside* the ``fmodel(...)`` (DDP-wrapped) forward call whose output
    DistributedDataParallel's ``find_unused_parameters=True`` traversal inspects to decide which
    parameters are unused this iteration. Once every layer is frozen, ``fmodel(...)``'s forward
    never touches theta (``soft_layers`` uses only detached argmax gates), so DDP's traversal marks
    theta "unused"/ready as soon as the forward returns -- but the entropy term then fires theta's
    real gradient-ready hook a second time in the same ``backward()`` call, which is exactly DDP's
    "mark ready only once" invariant. The fix (train.py's ``train_step``) simply skips the entropy
    term while ``heads_only`` is True: theta's gradient is unconditionally discarded a few lines
    later in that stage regardless (``model.theta.grad = None``), so omitting the term changes no
    single-process numerics -- it only removes the out-of-forward theta dependency that confused
    DDP.

    2 real gloo/CPU processes, microBatch 4 x accumulation 2 (effectiveBatch 16, world 2, so
    ``resolve_accumulation`` gives each rank 2 microbatches per optimizer step -- reproduces with
    actual gradient accumulation, matching the real run this was found in) and maxSteps 10, which
    with the default prefix-60-30-10 schedule and the "tiny" profile's 4 layers gives s1_end=6,
    s2_end=9: the run passes through stage 1 (steps 0-5, all soft), stage 2 (steps 6-8, layers
    freezing one at a time), and reaches stage 3 (step 9, heads_only) as its very last optimizer
    step -- exactly the scenario that crashed real training at step 90 of a 100-step schedule.
    Before the fix this reliably crashed both ranks (non-zero exit, no run-summary.json); after the
    fix both ranks must finish cleanly and rank 0 must write a run-summary.json with
    stepsTrained == maxSteps."""
    import torch.multiprocessing as mp

    n = 24
    make_fixture(tmp_path / "fx", n=n, seed=17)

    steps = 10
    assert resolve_accumulation(effective_batch=16, micro_batch=4, world_size=2) == 2
    ddp_cfg = write_config(tmp_path / "ddp.json", tmp_path / "fx", tmp_path / "run-ddp",
                            microBatch=4, effectiveBatch=16, maxSteps=steps, validationInterval=5,
                            checkpointInterval=5, maxValidationPositions=16)
    port = _free_port()
    status_dir = tmp_path / "status"
    status_dir.mkdir()
    mp.spawn(_run_worker, args=(2, port, ddp_cfg, str(status_dir)), nprocs=2, join=True)

    statuses = {}
    for r in range(2):
        with open(status_dir / f"status-{r}.json") as f:
            statuses[r] = json.load(f)
    assert statuses[0]["returnCode"] == 0 and statuses[1]["returnCode"] == 0, statuses

    out = tmp_path / "run-ddp"
    assert (out / "run-summary.json").exists()
    summary = json.load(open(out / "run-summary.json"))
    assert summary["stepsTrained"] == steps
    assert summary["completed"] is True
    assert summary["worldSize"] == 2
    assert summary["finalGatesAllFrozen"] is True
    assert summary["frozenPrefix"] == 4  # "tiny" profile: 4 logic-gate layers, all frozen by stage 3


def test_trainer_wraps_ddp_before_any_validation_collective(monkeypatch, tmp_path):
    """Unit-level regression test for the collective-ordering invariant this module's docstring and
    Trainer._verify_ddp_param_consistency document: DistributedDataParallel must wrap the model
    before the Trainer issues any collective that depends on the model matching across ranks --
    concretely, before validate()'s barrier()/broadcast_object() sync points. Uses a fake process
    group: torch.distributed's collectives and DistributedDataParallel itself are monkeypatched to
    recording no-ops, so this runs as a single real process pretending to be rank 0 of world_size 2
    (no real peer, no real gloo/NCCL backend) while still exercising the genuine Trainer.__init__ (DDP
    construction) followed by the same validate("initial") call run() makes first."""
    import ichigo_train.distributed as D
    from ichigo_train import train as T

    events: list[str] = []

    def fake_init_process_group(*a, **k):
        events.append("init_process_group")

    def fake_barrier(*a, **k):
        events.append("barrier")

    def fake_broadcast_object_list(box, src=0):
        events.append("broadcast_object_list")
        # rank (0) == src in this single-simulated-process test: box[0] already holds the real
        # rank-0 object, so there is nothing to fill in for a fake "receive".

    class FakeDDP:
        def __new__(cls, module, **kwargs):
            events.append("ddp_construct")
            return module  # identity wrap: sufficient to check ordering, no real replication needed

    saved_state = dict(D._STATE)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(D.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(D.dist, "barrier", fake_barrier)
    monkeypatch.setattr(D.dist, "broadcast_object_list", fake_broadcast_object_list)
    monkeypatch.setattr(T, "DistributedDataParallel", FakeDDP)
    try:
        make_fixture(tmp_path / "fx", n=11, seed=3)
        cfg = write_config(tmp_path / "cfg.json", tmp_path / "fx", tmp_path / "run", maxSteps=4, validationInterval=2, checkpointInterval=2)
        t = T.Trainer(cfg, None)
        assert t.world_size == 2 and t.rank == 0
        t.validate("initial")
        t.csv.close(); t.log.close()
    finally:
        D._STATE.clear()
        D._STATE.update(saved_state)

    assert events[0] == "init_process_group"
    assert "ddp_construct" in events
    ddp_index = events.index("ddp_construct")
    pre_ddp, post_ddp = events[:ddp_index], events[ddp_index + 1:]
    # The only collective allowed before the DDP wrap is _verify_ddp_param_consistency's own
    # single pre-flight broadcast; in particular no barrier() (validate()'s sync points) may occur.
    assert pre_ddp.count("barrier") == 0, f"a barrier() happened before DDP construction: {events}"
    assert pre_ddp.count("broadcast_object_list") == 1, f"expected exactly the pre-flight consistency broadcast before DDP construction: {events}"
    # validate("initial")'s own two barriers and one broadcast must all come after the DDP wrap.
    assert post_ddp.count("barrier") == 2, f"validate()'s barriers must follow DDP construction: {events}"
    assert post_ddp.count("broadcast_object_list") == 1, f"validate()'s broadcast must follow DDP construction: {events}"


def test_verify_ddp_param_consistency_raises_on_mismatch(monkeypatch, tmp_path):
    """Direct unit test of Trainer._verify_ddp_param_consistency: if the (broadcasted) reference
    signature from rank 0 does not match this rank's own model, it must raise a clear, local
    RuntimeError -- not hang, and not silently proceed into DistributedDataParallel's own harsher
    C++-level check."""
    import ichigo_train.distributed as D
    from ichigo_train import train as T

    make_fixture(tmp_path / "fx", n=11, seed=4)
    cfg = write_config(tmp_path / "cfg.json", tmp_path / "fx", tmp_path / "run", maxSteps=1, validationInterval=1, checkpointInterval=1)

    saved_state = dict(D._STATE)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(D.dist, "init_process_group", lambda *a, **k: None)
    # Simulate a peer whose rank-0 broadcast reports one extra parameter this rank does not have.
    monkeypatch.setattr(D.dist, "broadcast_object_list",
                         lambda box, src=0: box.__setitem__(0, box[0] + [("phantom", (1,), "torch.float32", True)]))
    try:
        with pytest.raises(RuntimeError, match="phantom|parameters"):
            T.Trainer(cfg, None)
    finally:
        D._STATE.clear()
        D._STATE.update(saved_state)
