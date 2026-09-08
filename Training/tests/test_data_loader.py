"""Tests for data_loader.py: the two-level (shard, within-shard) shuffle sampler, its resumable
state, its per-shard sha256 verification, and load_split_arrays' verification of the same."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ichigo_train import data_loader as DL
from ichigo_train import dataset as D

S = 9


def _rows(n, seed, id_offset=0):
    rng = np.random.default_rng(seed)
    buf = D.ShardBuffer(S)
    for i in range(n):
        spatial = rng.integers(0, 2, size=S * S * 32).tolist()
        legal = rng.integers(0, 2, size=S * S + 1).tolist(); legal[-1] = 1
        pol = np.zeros(S * S + 1); li = [j for j, l in enumerate(legal) if l]
        pol[rng.choice(li)] = 1.0
        # global[2] doubles as a globally-unique per-position marker for the disjointness test below.
        buf.add({"spatial": spatial, "global": [0.0, 9 / 19, (id_offset + i) / 1e6, 0], "legal": legal, "policy": pol.tolist(),
                 "expected_result": float(rng.uniform()), "score": float(rng.normal() * 5),
                 "ownership": rng.uniform(-1, 1, size=S * S).tolist(), "wdl": [0, 0, 0], "mask": [1, 1, 1, 1, 0],
                 "positionId": f"{id_offset + i:064x}", "gameId": f"{id_offset + i:064x}"})
    return buf.to_arrays()


def make_shard_dataset(out_dir, shard_sizes=(6, 5, 4), val_size=6, seed=0):
    """Writes manifest.json + train/validation shards by hand (bypassing build_dataset, which
    needs full SGF/game-family machinery). Returns the manifest dict."""
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    shards = []
    offset = 0
    for i, n in enumerate(shard_sizes):
        arrays = _rows(n, seed=seed + 100 + i, id_offset=offset)
        offset += n
        D.validate_arrays(arrays, S)
        path = os.path.join(out_dir, "train", f"shard-{i:05d}.npz")
        sha, npos = D.write_shard(path, arrays)
        shards.append({"split": "train", "file": f"train/shard-{i:05d}.npz", "sha256": sha, "positions": npos, "boardSize": S})
    val_arrays = _rows(val_size, seed=seed + 999)
    D.validate_arrays(val_arrays, S)
    vpath = os.path.join(out_dir, "validation", "shard-00000.npz")
    vsha, vn = D.write_shard(vpath, val_arrays)
    shards.append({"split": "validation", "file": "validation/shard-00000.npz", "sha256": vsha, "positions": vn, "boardSize": S})
    manifest = {"schemaVersion": D.SCHEMA_VERSION, "featureVersion": D.FEATURE_VERSION, "rulesId": D.RULES_ID,
                "boardSize": S, "shardSize": 4096, "counts": {"train": sum(shard_sizes), "validation": vn, "test": 0},
                "shards": shards, "provenance": {}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def test_shard_sampler_covers_every_position_each_epoch_and_resumes(tmp_path):
    make_shard_dataset(tmp_path, shard_sizes=(6, 5, 4))  # 15 positions, uneven shard sizes
    total = 15
    s1 = DL.ShardSampler(str(tmp_path), seed=1)
    seen_markers = set()
    for _ in range(total):
        b = s1.next_batch(1)
        seen_markers.add(round(float(b["global"][0, 2]) * 1e6))
    assert len(seen_markers) == total  # every position seen exactly once in the first epoch
    assert s1.epoch == 0  # the epoch/shard rollover is only detected lazily, on the *next* call

    # resume: continue for a few more draws from a snapshot, compare against a from-scratch
    # sampler driven the same total number of steps.
    snap = s1.state()
    for _ in range(4):
        s1.next_batch(1)
    assert s1.epoch == 1  # crossing the last shard's boundary rolls the epoch over correctly
    s2 = DL.ShardSampler(str(tmp_path), seed=1)
    s2.load_state(snap)
    got = []
    for _ in range(4):
        b = s2.next_batch(1)
        got.append(round(float(b["global"][0, 2]) * 1e6))
    s3 = DL.ShardSampler(str(tmp_path), seed=1)
    s3.load_state(snap)
    got2 = []
    for _ in range(4):
        b = s3.next_batch(1)
        got2.append(round(float(b["global"][0, 2]) * 1e6))
    assert got == got2  # resuming from the same state is deterministic
    assert s1.state() == s2.state() == s3.state()


def test_shard_sampler_holds_at_most_one_shard_resident(tmp_path, monkeypatch):
    """Regression test for the whole-corpus-permutation cache-thrashing bug: loading shards must
    scale with the number of DISTINCT shards touched, not with the number of positions drawn."""
    make_shard_dataset(tmp_path, shard_sizes=(4, 4, 4, 4, 4))  # 5 shards, 20 positions
    calls = {"n": 0}
    real_load_shard = DL.load_shard

    def counting_load_shard(path):
        calls["n"] += 1
        return real_load_shard(path)
    monkeypatch.setattr(DL, "load_shard", counting_load_shard)

    s = DL.ShardSampler(str(tmp_path), seed=2)
    for _ in range(25):  # a bit more than one full epoch (20 positions), forces >=2 shard visits per shard
        s.next_batch(1)
    # 5 shards for the first epoch pass, plus at most 5 more once epoch 2 starts reusing them
    # (a fresh permutation, but still one load per distinct shard) -- nowhere near 25 or more.
    assert calls["n"] <= 10, f"expected at most ~10 shard loads (one/shard/epoch), got {calls['n']}"


def test_shard_sampler_verifies_sha256(tmp_path):
    make_shard_dataset(tmp_path, shard_sizes=(6, 5, 4))
    m = json.load(open(tmp_path / "manifest.json"))
    bad = os.path.join(tmp_path, [s["file"] for s in m["shards"] if s["split"] == "train"][0])
    data = open(bad, "rb").read()
    open(bad, "wb").write(data[:-1] + bytes([data[-1] ^ 1]))
    s = DL.ShardSampler(str(tmp_path), seed=1)
    with pytest.raises(ValueError, match="sha256"):
        s.next_batch(1)


def test_load_split_arrays_verifies_sha256(tmp_path):
    make_shard_dataset(tmp_path, shard_sizes=(6, 5, 4))
    m = json.load(open(tmp_path / "manifest.json"))
    bad = os.path.join(tmp_path, [s["file"] for s in m["shards"] if s["split"] == "validation"][0])
    data = open(bad, "rb").read()
    open(bad, "wb").write(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError):
        DL.load_split_arrays(str(tmp_path), "validation", fixture_mode=False)


def test_rank_round_robin_lockstep_and_disjoint(tmp_path):
    make_shard_dataset(tmp_path, shard_sizes=(7, 6, 5))  # 18 positions, deliberately not divisible by 2 within every shard
    r0 = DL.ShardSampler(str(tmp_path), seed=3, rank=0, world_size=2)
    r1 = DL.ShardSampler(str(tmp_path), seed=3, rank=1, world_size=2)
    seen0, seen1 = set(), set()
    for _ in range(9):
        b0 = r0.next_batch(1)
        b1 = r1.next_batch(1)
        assert r0.state() == r1.state()  # both ranks stay in lockstep on (epoch, shardPos, offset)
        m0 = round(float(b0["global"][0, 2]) * 1e6)
        m1 = round(float(b1["global"][0, 2]) * 1e6)
        assert m0 not in seen1 and m1 not in seen0 and m0 != m1
        seen0.add(m0); seen1.add(m1)
    assert len(seen0 & seen1) == 0
    assert len(seen0) + len(seen1) == 18  # together the two ranks cover every position exactly once
