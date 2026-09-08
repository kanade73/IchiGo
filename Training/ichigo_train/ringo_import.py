"""RinGo ``.nngd`` v2 reader and target importer (docs/spec/02-training.md §11.2, T13).

The reader streams one sample at a time and validates magic/version/22 spatial/19 global/length/
finite values; v1 files are rejected. The importer requires an explicit mapping file
(``shardSha256, sampleIndex, positionId, symmetryId, sourceRunId`` per line) that ties every
reused sample to an IchiGo position; it never guesses a mapping. Without a mapping the
``inventory`` command only reports what exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass

import numpy as np

from .symmetry import inverse_permutation

MAGIC = b"NNGD"
VERSION = 2
NUM_SPATIAL = 22
NUM_GLOBAL = 19
HEADER = struct.Struct("<4sIIIII")


@dataclass
class RinGoSample:
    index: int
    spatial: np.ndarray  # f32 [nnLen*nnLen*22] NHWC
    global_: np.ndarray  # f32 [19]
    policy: np.ndarray   # f32 [nnLen*nnLen+1], side-to-move, pass last
    value: np.ndarray    # f32 [3] win, loss, noResult (side-to-move)
    score: float
    score_valid: bool
    ownership: np.ndarray  # int8 [nnLen*nnLen] side-to-move
    ownership_valid: bool


class RinGoFormatError(ValueError):
    pass


def sample_size(nn_len: int) -> int:
    area = nn_len * nn_len
    return (NUM_SPATIAL * area + NUM_GLOBAL + area + 1 + 3 + 1) * 4 + 1 + area + 1


def read_header(f) -> tuple[int, int]:
    head = f.read(HEADER.size)
    if len(head) != HEADER.size:
        raise RinGoFormatError("file shorter than header")
    magic, version, nn_len, ns, ng, count = HEADER.unpack(head)
    if magic != MAGIC:
        raise RinGoFormatError("bad magic")
    if version != VERSION:
        raise RinGoFormatError(f"unsupported RinGoData version {version} (v2 only)")
    if ns != NUM_SPATIAL or ng != NUM_GLOBAL:
        raise RinGoFormatError("unexpected feature counts")
    if nn_len not in (9, 19) and not (1 <= nn_len <= 19):
        raise RinGoFormatError("bad nnLen")
    return nn_len, count


def iter_samples(path: str):
    """Streams samples; validates total length first."""
    with open(path, "rb") as f:
        nn_len, count = read_header(f)
        size = os.path.getsize(path)
        expected = HEADER.size + count * sample_size(nn_len)
        if size != expected:
            raise RinGoFormatError(f"file length {size} != expected {expected}")
        area = nn_len * nn_len
        for i in range(count):
            raw = f.read(sample_size(nn_len))
            off = 0
            spatial = np.frombuffer(raw, "<f4", NUM_SPATIAL * area, off); off += 4 * NUM_SPATIAL * area
            glob = np.frombuffer(raw, "<f4", NUM_GLOBAL, off); off += 4 * NUM_GLOBAL
            policy = np.frombuffer(raw, "<f4", area + 1, off); off += 4 * (area + 1)
            value = np.frombuffer(raw, "<f4", 3, off); off += 12
            score = struct.unpack_from("<f", raw, off)[0]; off += 4
            score_valid = raw[off]; off += 1
            own = np.frombuffer(raw, np.int8, area, off); off += area
            own_valid = raw[off]; off += 1
            for arr, name in ((spatial, "spatial"), (glob, "global"), (policy, "policy"), (value, "value")):
                if not np.all(np.isfinite(arr)):
                    raise RinGoFormatError(f"sample {i}: non-finite {name}")
            if not np.isfinite(score):
                raise RinGoFormatError(f"sample {i}: non-finite score")
            yield nn_len, RinGoSample(i, spatial, glob, policy, value, score, bool(score_valid), own, bool(own_valid))


def inventory(paths: list[str]) -> dict:
    """Read-only inventory of RinGo shards (counts, validity, version)."""
    report = {"files": []}
    for p in paths:
        entry = {"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
        try:
            n = score_valid = own_valid = soft_policy = no_result = 0
            nn_len = None
            for nn_len, s in iter_samples(p):
                n += 1
                score_valid += s.score_valid
                own_valid += s.ownership_valid
                soft_policy += int(np.count_nonzero(s.policy) > 1)
                no_result += int(s.value[2] > 1e-6)
            entry.update({"version": VERSION, "nnLen": nn_len, "samples": n, "scoreValid": score_valid,
                          "ownershipValid": own_valid, "softPolicy": soft_policy, "noResult": no_result,
                          "hasGameId": False, "hasPositionIndex": False})
        except RinGoFormatError as e:
            entry["error"] = str(e)
        report["files"].append(entry)
    return report


def import_targets(shards: list[str], positions_path: str, mapping_path: str, out_labels: str, source_run_id: str | None = None) -> dict:
    """Converts mapped v2 samples into IchiGo label rows (same schema as teacher labels).

    Every mapping line must name a shard by sha256 and a sample index; the sample's policy is
    inverse-transformed by ``symmetryId`` (RinGo transform number == IchiGo D4 id is NOT assumed:
    the mapping carries the id already expressed in IchiGo numbering) and checked against the
    position's legal mask. noResult > 1e-6, invalid score/ownership, or legality mismatch → reject.
    """
    by_sha = {hashlib.sha256(open(p, "rb").read()).hexdigest(): p for p in shards}
    positions: dict[str, dict] = {}
    with open(positions_path) as f:
        for line in f:
            r = json.loads(line)
            positions[r["positionId"]] = {k: r[k] for k in ("positionId", "gameId", "turnNumber", "toMove", "boardSize", "legal")}
    mapping: dict[str, list[dict]] = {}
    with open(mapping_path) as f:
        for line in f:
            m = json.loads(line)
            mapping.setdefault(m["shardSha256"], []).append(m)
    report = {"mapped": 0, "reused": 0, "rejected": {}, "byLabelType": {"soft": 0, "onehot": 0}, "savedTeacherCalls": 0, "unknownShard": 0}

    def rej(reason):
        report["rejected"][reason] = report["rejected"].get(reason, 0) + 1

    with open(out_labels, "w") as out:
        for sha, entries in mapping.items():
            path = by_sha.get(sha)
            if path is None:
                report["unknownShard"] += len(entries)
                continue
            wanted = {e["sampleIndex"]: e for e in entries}
            for nn_len, s in iter_samples(path):
                e = wanted.get(s.index)
                if e is None:
                    continue
                report["mapped"] += 1
                pos = positions.get(e["positionId"])
                if pos is None:
                    rej("position not found"); continue
                if source_run_id and e.get("sourceRunId") != source_run_id:
                    rej("sourceRunId mismatch"); continue
                S = pos["boardSize"]
                if nn_len != S:
                    rej("nnLen mismatch"); continue
                if s.value[2] > 1e-6:
                    rej("noResult > 1e-6"); continue
                sym = int(e["symmetryId"])
                inv = inverse_permutation(S, sym)
                policy = np.zeros(S * S + 1, np.float32)
                policy[inv] = s.policy[: S * S]
                policy[S * S] = s.policy[S * S]
                legal = np.asarray(pos["legal"], np.uint8)
                if abs(float(policy.sum()) - 1) > 1e-4 or np.any(policy[legal == 0] > 0):
                    rej("policy illegal or not normalised"); continue
                policy = policy / policy.sum()
                wl = float(s.value[0] + s.value[1])
                if wl <= 0:
                    rej("win+loss zero"); continue
                expected = float(s.value[0]) / wl
                label = {"positionId": pos["positionId"], "gameId": pos["gameId"], "turnNumber": pos["turnNumber"], "toMove": pos["toMove"],
                         "boardSize": S, "policy": policy.tolist(), "expectedResult": expected,
                         "sourceType": "ringo-import", "teacherId": e.get("sourceRunId", "ringo"),
                         "labelType": "soft" if np.count_nonzero(s.policy) > 1 else "onehot"}
                report["byLabelType"][label["labelType"]] += 1
                mask = [1, 1, int(s.score_valid), int(s.ownership_valid), 0]
                label["score"] = float(s.score) if s.score_valid else 0.0
                own = np.zeros(S * S, np.float32)
                if s.ownership_valid:
                    own[inv] = s.ownership.astype(np.float32)
                label["ownership"] = own.tolist()
                label["mask"] = mask
                out.write(json.dumps(label) + "\n")
                report["reused"] += 1
    report["savedTeacherCalls"] = report["reused"]
    return report
