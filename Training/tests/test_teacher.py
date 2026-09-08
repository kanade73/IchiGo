import json
import os
import sys

import pytest

from ichigo_train.teacher import TeacherConfig, gtp_to_index, label_positions, response_to_label

FAKE = os.path.join(os.path.dirname(__file__), "fake_teacher.py")
S = 9


def make_positions(path, games=2, turns=5):
    rows = []
    for g in range(games):
        moves = [["B", "E5"], ["W", "D4"], ["B", "F6"], ["W", "C3"]][: turns - 1]
        gid = f"game{g}" + "0" * 40
        for t in range(turns):
            legal = [1] * (S * S + 1)
            for c, m in moves[:t]:
                legal[gtp_to_index(m, S)] = 0
            rows.append({"schemaVersion": 1, "positionId": f"{gid}-{t}", "gameId": gid, "boardSize": S, "komi": 7,
                         "rulesId": "cgos-area-psk-v1", "initialStones": [], "initialPlayer": "B", "moves": moves[:t],
                         "turnNumber": t, "toMove": "B" if t % 2 == 0 else "W", "legal": legal})
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def run(tmp_path, env=None, perspective="sidetomove", timeout=5.0, retries=2, **kw):
    pos = tmp_path / "pos.jsonl"
    make_positions(str(pos), **kw)
    out = tmp_path / "labels.jsonl"
    old = dict(os.environ)
    os.environ.update(env or {})
    try:
        cfg = TeacherConfig(command=[sys.executable, FAKE], teacher_id="fake", perspective=perspective, timeout_seconds=timeout, retries=retries, queue_depth=2)
        stats = label_positions(str(pos), str(out), cfg)
    finally:
        os.environ.clear(); os.environ.update(old)
    labels = [json.loads(l) for l in open(out)]
    rejects = [json.loads(l) for l in open(str(out) + ".rejects.jsonl")]
    return stats, labels, rejects


def test_basic_labels_and_normalisation(tmp_path):
    stats, labels, rejects = run(tmp_path)
    assert stats.labels == 10 and stats.rejected == 0 and not rejects
    by = {(l["gameId"], l["turnNumber"]): l for l in labels}
    l = by[("game0" + "0" * 40, 3)]
    assert abs(sum(l["policy"]) - 1) < 1e-9
    assert abs(l["expectedResult"] - 0.53) < 1e-9 and abs(l["score"] - 0.3) < 1e-9
    assert l["ownership"][0] == 0.75 and l["toMove"] == "W"
    # illegal (occupied) points have 0 policy
    assert l["policy"][gtp_to_index("E5", S)] == 0


def test_out_of_order_and_duplicates(tmp_path):
    stats, labels, _ = run(tmp_path, env={"FAKE_TEACHER_SHUFFLE": "1", "FAKE_TEACHER_DUPLICATE": "1"})
    assert stats.labels == 10 and stats.duplicates >= 8 and len(labels) == 10  # late duplicates after a query completes are dropped as stale
    assert sorted(l["turnNumber"] for l in labels if l["gameId"].startswith("game0")) == [0, 1, 2, 3, 4]


def test_missing_turn_retry_then_reject(tmp_path):
    stats, labels, rejects = run(tmp_path, env={"FAKE_TEACHER_DROP_TURN": "2"}, timeout=1.0, retries=1)
    assert stats.retries == 2 and stats.labels == 8 and stats.rejected == 2
    assert all(r["reason"] == "timeout" for r in rejects)


def test_missing_turn_recovered_by_retry(tmp_path):
    # the fake drops turn 2 only for ids ending in "-1" (first attempt); the resubmission answers it
    stats, labels, rejects = run(tmp_path, env={"FAKE_TEACHER_DROP_TURN": "2", "FAKE_TEACHER_DROP_ONCE": "1"}, timeout=1.0, retries=2, games=1)
    assert stats.labels == 5 and stats.rejected == 0 and stats.retries == 1


def test_black_perspective_conversion(tmp_path):
    stats, labels, _ = run(tmp_path, env={"FAKE_TEACHER_PERSPECTIVE": "black"}, perspective="black", games=1)
    by = {l["turnNumber"]: l for l in labels}
    # values are identical to the side-to-move run because the fake generated black-perspective values
    assert abs(by[1]["expectedResult"] - 0.51) < 1e-9 and abs(by[1]["score"] - 0.1) < 1e-9 and by[1]["ownership"][0] == 0.75
    assert abs(by[2]["expectedResult"] - 0.52) < 1e-9


def test_wrong_perspective_flag_detected_by_values(tmp_path):
    """If the adapter is told 'black' but the teacher reports side-to-move, white turns flip."""
    stats, labels, _ = run(tmp_path, env={"FAKE_TEACHER_PERSPECTIVE": "sidetomove"}, perspective="black", games=1)
    by = {l["turnNumber"]: l for l in labels}
    assert abs(by[1]["expectedResult"] - 0.49) < 1e-9  # flipped: 1 - 0.51


def test_illegal_visit_and_error_rejected(tmp_path):
    stats, labels, rejects = run(tmp_path, env={"FAKE_TEACHER_ILLEGAL": "1"}, games=1)
    assert stats.labels == 1 and stats.rejected == 4  # turn 0 has no last move
    assert all(r["reason"] == "teacher visited illegal move" for r in rejects)
    stats, labels, rejects = run(tmp_path, env={"FAKE_TEACHER_ERROR_ID": "game1"})
    assert stats.labels == 5 and stats.rejected == 5


def test_timeout_when_teacher_hangs(tmp_path):
    stats, labels, rejects = run(tmp_path, env={"FAKE_TEACHER_SLEEP": "3"}, timeout=0.5, retries=0, games=1)
    assert stats.labels == 0 and stats.rejected == 5


def test_response_to_label_rejects_nonfinite_and_sum_zero():
    pos = {"positionId": "p", "gameId": "g", "turnNumber": 0, "toMove": "B", "boardSize": 9, "legal": [1] * 82}
    cfg = TeacherConfig(command=[], teacher_id="t")
    base = {"rootInfo": {"currentPlayer": "B", "winrate": 0.5, "scoreLead": 1.0}, "moveInfos": [{"move": "E5", "visits": 1}], "ownership": [0.0] * 81}
    assert response_to_label(base, pos, cfg)[0] is not None
    bad = dict(base); bad["moveInfos"] = [{"move": "E5", "visits": 0}]
    assert response_to_label(bad, pos, cfg)[1] == "policy sum zero"
    bad = dict(base); bad["rootInfo"] = {"currentPlayer": "B", "winrate": float("nan"), "scoreLead": 1.0}
    assert response_to_label(bad, pos, cfg)[1] == "non-finite value"
    bad = dict(base); bad["ownership"] = [0.0] * 80
    assert response_to_label(bad, pos, cfg)[1] == "ownership length mismatch"
    assert gtp_to_index("pass", 9) == 81 and gtp_to_index("A9", 9) == 0 and gtp_to_index("J1", 9) == 80 and gtp_to_index("I5", 9) is None
