"""Real-teacher fixture (T12): 20 positions of one corpus game + a perspective probe, answered by
KataGo 1.18.2 with g170e-b20c256x2 (reportAnalysisWinratesAs = SIDETOMOVE, 128 visits).
Tests re-run the adapter conversion on the stored raw responses."""
import json
import os

from ichigo_train.teacher import TeacherConfig, response_to_label

FIX = os.path.join(os.path.dirname(__file__), "..", "..", "Tests", "Fixtures", "teacher")


def _load():
    q = json.load(open(os.path.join(FIX, "queries.json")))
    rs = [json.loads(l) for l in open(os.path.join(FIX, "responses.jsonl"))]
    return q, rs


def test_real_teacher_20_positions_convert():
    q, rs = _load()
    cfg = TeacherConfig(command=[], teacher_id="fixture", perspective="sidetomove")
    by_turn = {r["turnNumber"]: r for r in rs if r["id"] == "real20"}
    assert len(by_turn) == 20
    for pos in q["positions"]:
        label, reason = response_to_label(by_turn[pos["turnNumber"]], pos, cfg)
        assert label is not None, reason
        assert abs(sum(label["policy"]) - 1) < 1e-9
        assert all(p == 0 for p, l in zip(label["policy"], pos["legal"]) if l == 0)
        assert 0 <= label["expectedResult"] <= 1 and len(label["ownership"]) == 81
        assert by_turn[pos["turnNumber"]]["rootInfo"]["currentPlayer"] == pos["toMove"]


def test_perspective_probe_is_side_to_move():
    """Black has 9 stones in the centre, white to move: with SIDETOMOVE the reported winrate is
    near 0, scoreLead is strongly negative and the centre ownership is negative (opponent-owned)."""
    q, rs = _load()
    p = [r for r in rs if r["id"] == "probe-white-to-move"][0]
    assert p["rootInfo"]["currentPlayer"] == "W"
    assert p["rootInfo"]["winrate"] < 0.05 and p["rootInfo"]["scoreLead"] < -20
    assert p["ownership"][4 * 9 + 4] < -0.9
    pos = {"positionId": "probe", "gameId": "probe", "turnNumber": 17, "toMove": "W", "boardSize": 9, "legal": [1] * 82}
    for m in q["queries"][1]["moves"]:
        if m[1] != "pass":
            x = "ABCDEFGHJ".index(m[1][0]); y = 9 - int(m[1][1:]); pos["legal"][y * 9 + x] = 0
    label, _ = response_to_label(p, pos, TeacherConfig(command=[], teacher_id="f", perspective="sidetomove"))
    assert label["expectedResult"] < 0.05 and label["score"] < -20 and label["ownership"][40] < -0.9
    # a wrongly-declared 'black' perspective would flip everything for this white-to-move position
    wrong, _ = response_to_label(p, pos, TeacherConfig(command=[], teacher_id="f", perspective="black"))
    assert wrong["expectedResult"] > 0.95 and wrong["ownership"][40] > 0.9
