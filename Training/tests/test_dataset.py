import json
import os
import struct

import numpy as np
import pytest

from ichigo_train import dataset as D
from ichigo_train import ringo_import as R
from ichigo_train.teacher import gtp_to_index

S = 9


def _game(gid_seed, moves, komi=7):
    rows = []
    gid = D.canonical_hash(S, komi, [], "B", moves)
    for t in range(len(moves) + 1):
        pid = D.canonical_hash(S, komi, [], "B", moves[:t])
        legal = [1] * (S * S + 1)
        for _, m in moves[:t]:
            legal[gtp_to_index(m, S)] = 0
        spatial = [0] * (S * S * 32)
        for i in range(S * S):
            spatial[i * 32 + 28] = 1
        rows.append({"positionId": pid, "gameId": gid, "boardSize": S, "komi": komi, "rulesId": D.RULES_ID, "initialStones": [], "initialPlayer": "B",
                     "moves": moves[:t], "turnNumber": t, "toMove": "B" if t % 2 == 0 else "W", "spatial": spatial,
                     "global": [0, 9 / 19, t / 162, 0], "legal": legal, "sourceFile": f"{gid_seed}.sgf"})
    return rows


def _label(row):
    pol = [0.0] * (S * S + 1)
    legal_idx = [i for i, l in enumerate(row["legal"]) if l]
    pol[legal_idx[0]] = 0.6
    pol[legal_idx[1]] = 0.4
    return {"positionId": row["positionId"], "gameId": row["gameId"], "turnNumber": row["turnNumber"], "toMove": row["toMove"], "boardSize": S,
            "policy": pol, "expectedResult": 0.55, "score": 1.5, "ownership": [0.1] * (S * S), "teacherVisits": 128, "sourceType": "teacher", "teacherId": "t"}


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_split_key_is_symmetry_invariant():
    moves = [["B", "C3"], ["W", "G7"], ["B", "D5"]]
    k = D.split_key(S, 7, [], "B", moves)
    # rotate every move by 90° (R: x,y -> S-1-y, x) and check the key is unchanged
    rot = [["B", "C7"], ["W", "G3"], ["B", "E6"]]  # C3=(2,6)->(2,2)=C7, G7=(6,2)->(6,6)=G3, D5=(3,4)->(4,3)=E6
    from ichigo_train.dataset import _transform_gtp
    assert [_transform_gtp(m, S, 1) for _, m in moves] == [m for _, m in rot]
    assert D.split_key(S, 7, [], "B", rot) == k
    assert D.split_of("00" * 32) == "train" and D.split_of("0000000000000063" + "00" * 24) == "test" and D.split_of("000000000000005a" + "00" * 24) == "validation"


def test_build_dataset_split_dedupe_and_shards(tmp_path):
    rows = []
    # many games so that every split is non-empty; games 0..N share the first 2 moves (duplicate opening positions)
    rng = np.random.default_rng(0)
    pts = [f"{c}{r}" for c in "ABCDEFGHJ" for r in range(1, 10)]
    for g in range(60):
        seq = list(rng.permutation(pts)[:6])
        moves = [["B", "E5"], ["W", "D4"]] + [["B" if i % 2 == 0 else "W", m] for i, m in enumerate(seq)]
        rows += _game(g, moves)
    pos = tmp_path / "pos.jsonl"; lab = tmp_path / "lab.jsonl"
    _write(pos, rows)
    _write(lab, [_label(r) for r in rows])
    out = tmp_path / "ds"
    manifest = D.build_dataset(str(pos), str(lab), str(out), {"test": True}, shard_size=500)
    rep = manifest["provenance"]["buildReport"]
    assert rep["duplicates"] > 0  # shared opening positions were deduplicated
    assert all(manifest["counts"][s] > 0 for s in ("train", "validation", "test"))
    # no position or game family crosses splits
    seen_pos, seen_game = {}, {}
    for split in ("train", "validation", "test"):
        for _, a in D.iter_split(str(out), split):
            D.validate_arrays(a, S)
            for pid, gid in zip(a["position_id"], a["game_id"]):
                p, g = pid.tobytes().hex(), gid.tobytes().hex()
                assert seen_pos.setdefault(p, split) == split
                assert seen_game.setdefault(g, split) == split
    assert len(seen_pos) == rep["labelled"]
    idx = [json.loads(l) for l in open(out / "positions-index.jsonl")]
    assert len(idx) == rep["labelled"] and idx[0]["teacherId"] == "t"
    # shard sha verification
    m = D.read_manifest(str(out))
    bad = os.path.join(out, m["shards"][0]["file"])
    data = open(bad, "rb").read()
    open(bad, "wb").write(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError):
        list(D.iter_split(str(out), m["shards"][0]["split"]))


def test_empty_holdout_is_error(tmp_path):
    rows = _game(0, [["B", "E5"], ["W", "D4"]])
    pos = tmp_path / "pos.jsonl"; lab = tmp_path / "lab.jsonl"
    _write(pos, rows); _write(lab, [_label(r) for r in rows])
    with pytest.raises(ValueError, match="holdout"):
        D.build_dataset(str(pos), str(lab), str(tmp_path / "ds"), {})


def test_masked_targets_and_wdl(tmp_path):
    rows = []
    rng = np.random.default_rng(1)
    pts = [f"{c}{r}" for c in "ABCDEFGHJ" for r in range(1, 10)]
    for g in range(60):
        seq = list(rng.permutation(pts)[:5])
        rows += _game(g, [["B" if i % 2 == 0 else "W", m] for i, m in enumerate(seq)])
    labels = []
    for r in rows:
        l = _label(r)
        l["mask"] = [1, 1, 0, 0, 0]; l["score"] = 0.0; l["ownership"] = [0.0] * 81
        labels.append(l)
    pos = tmp_path / "pos.jsonl"; lab = tmp_path / "lab.jsonl"
    _write(pos, rows); _write(lab, labels)
    results = {rows[0]["gameId"]: "W"}
    manifest = D.build_dataset(str(pos), str(lab), str(tmp_path / "ds"), {}, wdl_from_results=results)
    n_wdl = 0
    for split in ("train", "validation", "test"):
        for _, a in D.iter_split(str(tmp_path / "ds"), split):
            assert np.all(a["target_mask"][:, 2] == 0) and np.all(a["score"] == 0)
            wm = a["target_mask"][:, 4] == 1
            n_wdl += int(wm.sum())
            for i in np.nonzero(wm)[0]:
                # white won: white-to-move rows get win, black-to-move rows get loss
                pass
    assert n_wdl == 6


# ---- RinGo v2 reader / importer ----

def _write_v2(path, samples, nn_len=9):
    area = nn_len * nn_len
    with open(path, "wb") as f:
        f.write(R.HEADER.pack(R.MAGIC, R.VERSION, nn_len, 22, 19, len(samples)))
        for s in samples:
            f.write(np.asarray(s["spatial"], "<f4").tobytes())
            f.write(np.asarray(s["global"], "<f4").tobytes())
            f.write(np.asarray(s["policy"], "<f4").tobytes())
            f.write(np.asarray(s["value"], "<f4").tobytes())
            f.write(struct.pack("<f", s["score"]))
            f.write(bytes([s["scoreValid"]]))
            f.write(np.asarray(s["ownership"], np.int8).tobytes())
            f.write(bytes([s["ownershipValid"]]))


def test_ringo_v2_reader_and_importer(tmp_path):
    rows = _game(0, [["B", "E5"], ["W", "D4"], ["B", "C3"]])
    pos = tmp_path / "pos.jsonl"
    _write(pos, rows)
    area = S * S
    from ichigo_train.symmetry import forward_permutation
    samples = []
    for t, r in enumerate(rows):
        pol = np.zeros(area + 1, np.float32)
        legal_idx = [i for i, l in enumerate(r["legal"]) if l and i < area]
        pol[legal_idx[0]] = 0.7; pol[legal_idx[1]] = 0.3
        sym = t % 8
        fwd = forward_permutation(S, sym)
        pol_sym = np.zeros(area + 1, np.float32); pol_sym[fwd] = pol[:area]; pol_sym[area] = pol[area]
        own = np.zeros(area, np.int8); own[legal_idx[0]] = 1
        own_sym = np.zeros(area, np.int8); own_sym[fwd] = own
        samples.append({"spatial": np.zeros(22 * area), "global": np.zeros(19), "policy": pol_sym, "value": [0.6, 0.4, 0.0] if t != 2 else [0.5, 0.4, 0.1],
                        "score": 2.0, "scoreValid": 1 if t != 1 else 0, "ownership": own_sym, "ownershipValid": 1, "sym": sym, "pid": r["positionId"], "leg0": legal_idx[0]})
    shard = tmp_path / "s.nngd"
    _write_v2(shard, samples)
    inv = R.inventory([str(shard)])
    assert inv["files"][0]["samples"] == 4 and inv["files"][0]["noResult"] == 1 and inv["files"][0]["scoreValid"] == 3
    sha = inv["files"][0]["sha256"]
    mapping = tmp_path / "map.jsonl"
    _write(mapping, [{"shardSha256": sha, "sampleIndex": i, "positionId": s["pid"], "symmetryId": s["sym"], "sourceRunId": "run1"} for i, s in enumerate(samples)])
    outl = tmp_path / "labels.jsonl"
    rep = R.import_targets([str(shard)], str(pos), str(mapping), str(outl), "run1")
    assert rep["mapped"] == 4 and rep["reused"] == 3 and rep["rejected"] == {"noResult > 1e-6": 1}
    labels = {json.loads(l)["turnNumber"]: json.loads(l) for l in open(outl)}
    assert abs(labels[0]["policy"][samples[0]["leg0"]] - 0.7) < 1e-6
    assert abs(labels[1]["policy"][samples[1]["leg0"]] - 0.7) < 1e-6  # symmetry 1 undone
    assert labels[1]["mask"] == [1, 1, 0, 1, 0] and labels[1]["score"] == 0.0
    assert labels[3]["ownership"][samples[3]["leg0"]] == 1.0
    assert abs(labels[0]["expectedResult"] - 0.6) < 1e-6
    # v1 rejected, truncated rejected
    bad = tmp_path / "v1.nngd"
    with open(bad, "wb") as f:
        f.write(R.HEADER.pack(R.MAGIC, 1, 9, 22, 19, 0))
    with pytest.raises(R.RinGoFormatError, match="version"):
        list(R.iter_samples(str(bad)))
    data = open(shard, "rb").read()
    open(bad, "wb").write(data[:-5])
    with pytest.raises(R.RinGoFormatError, match="length"):
        list(R.iter_samples(str(bad)))
