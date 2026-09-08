"""Shard v1 dataset (docs/spec/02-training.md §4, T13).

``dataset/manifest.json``, ``train|validation|test/*.npz`` (pickle-free), ``positions-index.jsonl``.
One shard holds at most ``SHARD_SIZE`` positions of a single board size. Arrays:

  spatial uint8[N,S,S,32], global f32[N,4], legal uint8[N,S*S+1], policy f32[N,S*S+1],
  expected_result f32[N], score f32[N], ownership f32[N,S,S], wdl f32[N,3],
  target_mask uint8[N,5] (policy, expected_result, score, ownership, wdl), sample_weight f32[N],
  position_id uint8[N,32], game_id uint8[N,32]

Split: ``splitKey`` = min over the 8 D4 images of the game's canonical gameId, first 8 bytes as a
big-endian integer mod 100 → 0..89 train, 90..94 validation, 95..99 test. Duplicate positionIds
are kept in one split with priority train → validation → test.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .symmetry import map_point

SCHEMA_VERSION = 1
FEATURE_VERSION = 1
RULES_ID = "cgos-area-psk-v1"
SHARD_SIZE = 4096
TARGET_NAMES = ["policy", "expected_result", "score", "ownership", "wdl"]
GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(board_size: int, komi, initial_stones, initial_player: str, moves) -> str:
    komi_v = int(komi) if float(komi) == int(komi) else float(komi)
    obj = {"boardSize": board_size, "komi": komi_v, "rulesId": RULES_ID, "initialStones": initial_stones,
           "initialPlayer": initial_player, "moves": moves}
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _transform_gtp(move: str, size: int, sym: int) -> str:
    if move.lower() == "pass":
        return "pass"
    x = GTP_LETTERS.index(move[0].upper())
    y = size - int(move[1:])
    nx, ny = map_point(x, y, size, sym)
    return f"{GTP_LETTERS[nx]}{size - ny}"


def _transform_sgf(pt: str, size: int, sym: int) -> str:
    x, y = ord(pt[0]) - 97, ord(pt[1]) - 97
    nx, ny = map_point(x, y, size, sym)
    return chr(97 + nx) + chr(97 + ny)


def split_key(board_size: int, komi, initial_stones, initial_player: str, moves) -> str:
    """Minimum gameId over the 8 D4 images (so symmetric games share a split)."""
    best = None
    for sym in range(8):
        stones = [[c, _transform_sgf(p, board_size, sym)] for c, p in initial_stones]
        stones.sort(key=lambda s: (s[0] != "B", ord(s[1][1]), ord(s[1][0])))
        mv = [[c, _transform_gtp(m, board_size, sym)] for c, m in moves]
        h = canonical_hash(board_size, komi, stones, initial_player, mv)
        if best is None or h < best:
            best = h
    return best


def split_of(key_hex: str) -> str:
    v = int(key_hex[:16], 16) % 100
    return "train" if v < 90 else ("validation" if v < 95 else "test")


@dataclass
class ShardBuffer:
    size: int
    rows: list = field(default_factory=list)

    def add(self, row: dict):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def to_arrays(self) -> dict[str, np.ndarray]:
        S = self.size
        N = len(self.rows)
        out = {
            "spatial": np.zeros((N, S, S, 32), np.uint8), "global": np.zeros((N, 4), np.float32),
            "legal": np.zeros((N, S * S + 1), np.uint8), "policy": np.zeros((N, S * S + 1), np.float32),
            "expected_result": np.zeros(N, np.float32), "score": np.zeros(N, np.float32),
            "ownership": np.zeros((N, S, S), np.float32), "wdl": np.zeros((N, 3), np.float32),
            "target_mask": np.zeros((N, 5), np.uint8), "sample_weight": np.ones(N, np.float32),
            "position_id": np.zeros((N, 32), np.uint8), "game_id": np.zeros((N, 32), np.uint8),
        }
        for i, r in enumerate(self.rows):
            out["spatial"][i] = np.asarray(r["spatial"], np.uint8).reshape(S, S, 32)
            out["global"][i] = r["global"]
            out["legal"][i] = r["legal"]
            m = r["mask"]
            out["target_mask"][i] = m
            if m[0]:
                out["policy"][i] = r["policy"]
            if m[1]:
                out["expected_result"][i] = r["expected_result"]
            if m[2]:
                out["score"][i] = r["score"]
            if m[3]:
                out["ownership"][i] = np.asarray(r["ownership"], np.float32).reshape(S, S)
            if m[4]:
                out["wdl"][i] = r["wdl"]
            out["sample_weight"][i] = r.get("sample_weight", 1.0)
            out["position_id"][i] = np.frombuffer(bytes.fromhex(r["positionId"]), np.uint8)
            out["game_id"][i] = np.frombuffer(bytes.fromhex(r["gameId"]), np.uint8)
        return out


def validate_arrays(a: dict[str, np.ndarray], size: int) -> None:
    N = a["spatial"].shape[0]
    S = size
    assert a["spatial"].shape == (N, S, S, 32) and a["spatial"].dtype == np.uint8
    assert np.all((a["spatial"] == 0) | (a["spatial"] == 1))
    assert a["legal"].shape == (N, S * S + 1) and np.all(a["legal"] <= 1)
    assert a["policy"].shape == (N, S * S + 1)
    pm = a["target_mask"][:, 0] == 1
    if pm.any():
        assert np.allclose(a["policy"][pm].sum(1), 1, atol=1e-5), "policy must sum to 1"
        assert np.all(a["policy"][pm][a["legal"][pm] == 0] == 0), "illegal moves must have 0 policy"
    assert np.all(np.isfinite(a["global"])) and np.all(np.isfinite(a["policy"]))
    assert np.all(np.isfinite(a["score"])) and np.all(np.isfinite(a["ownership"])) and np.all(np.isfinite(a["wdl"]))
    assert np.all(a["sample_weight"] > 0) and np.all(np.isfinite(a["sample_weight"]))
    em = a["target_mask"][:, 1] == 1
    assert np.all((a["expected_result"][em] >= 0) & (a["expected_result"][em] <= 1))
    om = a["target_mask"][:, 3] == 1
    assert np.all(np.abs(a["ownership"][om]) <= 1)
    wm = a["target_mask"][:, 4] == 1
    if wm.any():
        assert np.allclose(a["wdl"][wm].sum(1), 1, atol=1e-5)
    # masked-out targets are zero
    for k, col in zip(["policy", "expected_result", "score", "ownership", "wdl"], range(5)):
        off = a["target_mask"][:, col] == 0
        assert np.all(a[k][off] == 0)


def write_shard(path: str, arrays: dict[str, np.ndarray]) -> tuple[str, int]:
    np.savez(path, **arrays)
    data = open(path, "rb").read()
    return hashlib.sha256(data).hexdigest(), arrays["spatial"].shape[0]


def load_shard(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class DatasetWriter:
    def __init__(self, out_dir: str, board_size: int, shard_size: int = SHARD_SIZE):
        self.out_dir = out_dir
        self.size = board_size
        self.shard_size = shard_size
        self.buffers = {s: ShardBuffer(board_size) for s in ("train", "validation", "test")}
        self.shards: list[dict] = []
        self.counts = {s: 0 for s in self.buffers}
        for s in self.buffers:
            os.makedirs(os.path.join(out_dir, s), exist_ok=True)
        self.index = open(os.path.join(out_dir, "positions-index.jsonl"), "w")

    def add(self, split: str, row: dict, index_entry: dict):
        self.buffers[split].add(row)
        self.index.write(json.dumps(index_entry) + "\n")
        if len(self.buffers[split]) >= self.shard_size:
            self._flush(split)

    def _flush(self, split: str):
        buf = self.buffers[split]
        if not len(buf):
            return
        arrays = buf.to_arrays()
        validate_arrays(arrays, self.size)
        name = f"shard-{len([s for s in self.shards if s['split'] == split]):05d}.npz"
        path = os.path.join(self.out_dir, split, name)
        sha, n = write_shard(path, arrays)
        self.shards.append({"split": split, "file": f"{split}/{name}", "sha256": sha, "positions": n, "boardSize": self.size})
        self.counts[split] += n
        self.buffers[split] = ShardBuffer(self.size)

    def finish(self, provenance: dict) -> dict:
        for s in list(self.buffers):
            self._flush(s)
        self.index.close()
        if self.counts["validation"] == 0 or self.counts["test"] == 0:
            raise ValueError(f"holdout split is empty ({self.counts}); add more games")
        manifest = {
            "schemaVersion": SCHEMA_VERSION, "featureVersion": FEATURE_VERSION, "rulesId": RULES_ID,
            "boardSize": self.size, "shardSize": self.shard_size, "counts": self.counts, "shards": self.shards,
            "provenance": provenance,
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest


def read_manifest(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "manifest.json")) as f:
        m = json.load(f)
    if m.get("schemaVersion") != SCHEMA_VERSION or m.get("featureVersion") != FEATURE_VERSION or m.get("rulesId") != RULES_ID:
        raise ValueError("dataset schema/feature/rules version mismatch")
    return m


def manifest_hash(data_dir: str) -> str:
    """Stable hash of the shard list (files + sha256) used to bind checkpoints to a dataset."""
    m = read_manifest(data_dir)
    return hashlib.sha256(canonical_json([[s["file"], s["sha256"]] for s in m["shards"]]).encode()).hexdigest()


def iter_split(data_dir: str, split: str, verify: bool = True):
    """Yields (shard_path, arrays) for each shard of a split, verifying sha256 when requested."""
    m = read_manifest(data_dir)
    for s in m["shards"]:
        if s["split"] != split:
            continue
        p = os.path.join(data_dir, s["file"])
        if verify:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if h != s["sha256"]:
                raise ValueError(f"shard {s['file']} sha256 mismatch")
        yield p, load_shard(p)


def build_dataset(positions_path: str, labels_path: str, out_dir: str, provenance: dict, shard_size: int = SHARD_SIZE,
                  wdl_from_results: dict | None = None) -> dict:
    """Join positions and labels on positionId, split by game family, dedupe positions, write shards.

    Streams the positions file; labels are indexed by positionId (a dict of small rows).
    ``wdl_from_results`` (optional) maps gameId -> ("B"|"W"|"draw") for games that ended on the
    board with a trustworthy scored result; those positions also get a wdl target (to-move view).
    """
    labels: dict[str, dict] = {}
    with open(labels_path) as f:
        for line in f:
            l = json.loads(line)
            labels.setdefault(l["positionId"], l)
    report = {"positions": 0, "labelled": 0, "unlabelled": 0, "duplicates": 0, "split": {}, "wdl": 0}
    # Pass 1: split key per game (needs the full move list of the game -> max turn row) and dedupe map
    game_moves: dict[str, dict] = {}
    with open(positions_path) as f:
        for line in f:
            r = json.loads(line)
            g = game_moves.get(r["gameId"])
            if g is None or len(r["moves"]) > len(g["moves"]):
                game_moves[r["gameId"]] = {"moves": r["moves"], "initialStones": r["initialStones"], "initialPlayer": r["initialPlayer"], "komi": r["komi"], "boardSize": r["boardSize"]}
    game_split = {gid: split_of(split_key(g["boardSize"], g["komi"], g["initialStones"], g["initialPlayer"], g["moves"])) for gid, g in game_moves.items()}
    # duplicate positions: keep in the highest-priority split (train > validation > test)
    prio = {"train": 0, "validation": 1, "test": 2}
    pos_best: dict[str, str] = {}
    with open(positions_path) as f:
        for line in f:
            r = json.loads(line)
            if r["positionId"] not in labels:
                continue
            s = game_split[r["gameId"]]
            if r["positionId"] not in pos_best or prio[s] < prio[pos_best[r["positionId"]]]:
                pos_best[r["positionId"]] = s
    size = None
    writer = None
    seen: set[str] = set()
    with open(positions_path) as f:
        for line in f:
            r = json.loads(line)
            report["positions"] += 1
            lab = labels.get(r["positionId"])
            if lab is None:
                report["unlabelled"] += 1
                continue
            if r["positionId"] in seen:
                report["duplicates"] += 1
                continue
            split = game_split[r["gameId"]]
            if split != pos_best[r["positionId"]]:
                report["duplicates"] += 1
                continue
            seen.add(r["positionId"])
            if writer is None:
                size = r["boardSize"]
                writer = DatasetWriter(out_dir, size, shard_size)
            assert r["boardSize"] == size, "mixed board sizes in one dataset"
            mask = list(lab.get("mask", [1, 1, 1, 1, 0]))
            wdl = [0.0, 0.0, 0.0]
            if wdl_from_results and r["gameId"] in wdl_from_results:
                res = wdl_from_results[r["gameId"]]
                if res == "draw":
                    wdl = [0.0, 1.0, 0.0]
                else:
                    wdl = [1.0, 0.0, 0.0] if res == r["toMove"] else [0.0, 0.0, 1.0]
                mask[4] = 1
                report["wdl"] += 1
            row = {"spatial": r["spatial"], "global": r["global"], "legal": r["legal"], "policy": lab["policy"],
                   "expected_result": lab["expectedResult"], "score": lab["score"], "ownership": lab["ownership"],
                   "wdl": wdl, "mask": mask, "positionId": r["positionId"], "gameId": r["gameId"]}
            writer.add(split, row, {"positionId": r["positionId"], "gameId": r["gameId"], "turnNumber": r["turnNumber"],
                                     "split": split, "sourceFile": r.get("sourceFile"), "teacherId": lab.get("teacherId"),
                                     "sourceType": lab.get("sourceType", "teacher")})
            report["labelled"] += 1
    if writer is None:
        raise ValueError("no labelled positions")
    manifest = writer.finish(dict(provenance, buildReport=report))
    report["split"] = manifest["counts"]
    return manifest
