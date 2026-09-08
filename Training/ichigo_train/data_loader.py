"""Shard streaming, sampler state and D4 augmentation for training (T15)."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from .dataset import iter_split, load_shard, read_manifest
from .symmetry import forward_permutation

ARRAY_KEYS = ["spatial", "global", "legal", "policy", "expected_result", "score", "ownership", "wdl", "target_mask", "sample_weight"]


def load_split_arrays(data_dir: str, split: str, fixture_mode: bool = False, limit: int | None = None) -> dict[str, np.ndarray]:
    """Concatenates every shard of a split into memory (fine for <=1e5 9x9 positions; large
    corpora should be streamed with ShardSampler instead). Verifies each shard's sha256 against
    manifest.json (via dataset.iter_split) before use; a corrupted shard raises ValueError."""
    if fixture_mode:
        arrays = load_shard(os.path.join(data_dir, "fixture.npz"))
        return {k: arrays[k] for k in ARRAY_KEYS}
    parts = [arrays for _, arrays in iter_split(data_dir, split, verify=True)]
    if not parts:
        raise ValueError(f"split {split} is empty")
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in ARRAY_KEYS}
    if limit is not None:
        out = {k: v[:limit] for k, v in out.items()}
    return out


def _sha256_file(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class ShardSampler:
    """Two-level shuffle over the training shards: each epoch permutes the shard order, then
    permutes the positions within each shard as it is streamed. At most one shard's arrays are
    held in memory at a time (versus reloading shards under a single flattened whole-corpus
    permutation, which thrashes the cache once the corpus has more than a handful of shards).

    State = (epoch, shardPos, offset): shardPos indexes the epoch's shard permutation, offset
    indexes the current shard's within-shard permutation. Both permutations are derived purely
    from (seed, epoch[, shard index]), so this triple alone is enough to resume deterministically.

    Under DDP each rank runs this same deterministic sequence of (shard, within-shard-offset)
    candidates and keeps its own identical copy of the state (every rank's next_batch() call
    examines exactly the same candidates, in the same order); a candidate is kept by rank
    ``offset % world_size == rank`` so ranks see disjoint positions of the same permutation. The
    round-robin phase resets at each shard boundary rather than running continuously over the
    whole epoch, which is fine: what matters for correctness is that every rank advances through
    an identical state sequence regardless of which candidates *it* happens to keep, not that a
    single shard is split perfectly evenly (docs/spec/02-training.md §7: "DDPがデータを自動分割すると
    思い込まない" -- sharding is the sampler's job, and every rank must derive the same partition).
    """

    def __init__(self, data_dir: str, seed: int, fixture_mode: bool = False, rank: int = 0, world_size: int = 1):
        self.data_dir = data_dir
        self.seed = seed
        self.fixture_mode = fixture_mode
        self.rank = rank
        self.world_size = world_size
        if fixture_mode:
            self.shard_files = [os.path.join(data_dir, "fixture.npz")]
            self._sizes = [load_shard(self.shard_files[0])["spatial"].shape[0]]
            self._sha256: dict[str, str] = {}
        else:
            m = read_manifest(data_dir)
            shards = [s for s in m["shards"] if s["split"] == "train"]
            self.shard_files = [os.path.join(data_dir, s["file"]) for s in shards]
            self._sizes = [s["positions"] for s in shards]
            self._sha256 = {os.path.join(data_dir, s["file"]): s["sha256"] for s in shards}
        if not self.shard_files:
            raise ValueError("no training shards found")
        self.n_shards = len(self.shard_files)
        self.total = sum(self._sizes)
        self.epoch = 0
        self.shard_pos = 0
        self.offset = 0
        self._shard_order = None
        self._within_order = None
        self._current_file = None
        self._current_arrays: dict | None = None

    def state(self) -> dict:
        return {"epoch": self.epoch, "shardPos": self.shard_pos, "offset": self.offset}

    def load_state(self, st: dict):
        self.epoch = st["epoch"]
        self.shard_pos = st["shardPos"]
        self.offset = st["offset"]
        self._shard_order = None
        self._within_order = None
        self._current_file = None
        self._current_arrays = None

    def _ensure_shard_order(self):
        if self._shard_order is None:
            rng = np.random.Generator(np.random.PCG64([self.seed, self.epoch]))
            self._shard_order = rng.permutation(self.n_shards)

    def _load_verified(self, path: str) -> dict:
        arrays = load_shard(path)
        expected = self._sha256.get(path)
        if expected is not None and _sha256_file(path) != expected:
            raise ValueError(f"shard {path} sha256 mismatch against manifest.json (corrupted?)")
        return arrays

    def _ensure_current_shard(self):
        self._ensure_shard_order()
        idx = int(self._shard_order[self.shard_pos])
        f = self.shard_files[idx]
        if self._current_file != f:
            self._current_arrays = self._load_verified(f)
            self._current_file = f
            self._within_order = None
        if self._within_order is None:
            rng = np.random.Generator(np.random.PCG64([self.seed, self.epoch, idx]))
            self._within_order = rng.permutation(self._sizes[idx])

    def _advance_shard(self):
        self.shard_pos += 1
        self.offset = 0
        self._within_order = None
        if self.shard_pos >= self.n_shards:
            self.epoch += 1
            self.shard_pos = 0
            self._shard_order = None

    def next_batch(self, batch: int) -> dict[str, np.ndarray]:
        """Positions are sharded round-robin (per shard) across ranks so every rank sees disjoint
        data while all ranks stay in lockstep on (epoch, shardPos, offset).

        Every rank scans the same fixed-size window of ``batch * world_size`` consecutive
        candidates per call -- not "keep scanning until I personally have `batch` accepted ones",
        which would make each rank advance its own state by a different amount per call (a rank
        whose residue is hit early in the round-robin cycle finishes scanning sooner than one
        whose residue is hit late), silently desyncing (epoch, shardPos, offset) across ranks after
        the very first call. Since a full window of ``batch * world_size`` consecutive candidates
        contains each residue ``0..world_size-1`` exactly ``batch`` times regardless of the
        window's starting phase, every rank ends the call with exactly ``batch`` rows *and* with
        identical (epoch, shardPos, offset) -- required for the saved (rank0-only) checkpoint's
        sampler state to be valid for every rank after a resume.
        """
        rows = []
        scanned = 0
        need = batch * self.world_size
        while scanned < need:
            self._ensure_current_shard()
            idx = int(self._shard_order[self.shard_pos])
            size = self._sizes[idx]
            if self.offset >= size:
                self._advance_shard()
                continue
            take = (scanned % self.world_size == self.rank)
            pos = int(self._within_order[self.offset])
            self.offset += 1
            scanned += 1
            if not take:
                continue
            arrays = self._current_arrays
            rows.append({k: arrays[k][pos] for k in ARRAY_KEYS})
        return {k: np.stack([r[k] for r in rows]) for k in ARRAY_KEYS}


def augment_d4(batch: dict[str, np.ndarray], size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Applies an independent random symmetry to each sample (spatial, legal, policy, ownership)."""
    out = {k: v.copy() for k, v in batch.items()}
    syms = rng.integers(0, 8, size=batch["spatial"].shape[0])
    perms = [forward_permutation(size, s) for s in range(8)]
    P = size * size
    for i, s in enumerate(syms):
        if s == 0:
            continue
        perm = perms[s]
        sp = batch["spatial"][i].reshape(P, 32)
        t = np.empty_like(sp); t[perm] = sp
        out["spatial"][i] = t.reshape(size, size, 32)
        for key in ("legal", "policy"):
            src = batch[key][i]
            dst = out[key][i]
            dst[perm] = src[:P]
            dst[P] = src[P]
        own = batch["ownership"][i].reshape(P)
        t2 = np.empty_like(own); t2[perm] = own
        out["ownership"][i] = t2.reshape(size, size)
    return out


def to_tensors(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    t = {}
    for k, v in batch.items():
        if k in ("spatial", "legal", "target_mask"):
            t[k] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
        else:
            t[k] = torch.from_numpy(np.ascontiguousarray(v.astype(np.float32))).to(device)
    return t
