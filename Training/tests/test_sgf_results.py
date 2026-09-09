"""docs/spec/05-validation.md §5 / T29: SGF RE-tag -> real game-result JSONL."""

import json

import numpy as np
import pytest

from ichigo_train import dataset as D
from ichigo_train import sgf_results as SR
from ichigo_train.teacher import gtp_to_index

S = 9


# ---- parse_re / extract_re: every RE form the spec lists, plus common non-target forms ----

@pytest.mark.parametrize("value,expected", [
    ("B+R", ("B", "resign")),
    ("w+r", ("W", "resign")),
    ("B+Resign", ("B", "resign")),
    ("W+Resign", ("W", "resign")),
    ("B+3.5", ("B", "score")),
    ("W+7", ("W", "score")),
    ("b+0.5", ("B", "score")),
    ("0", ("draw", "draw")),
    ("Draw", ("draw", "draw")),
    ("Jigo", ("draw", "draw")),
    (" B+R ", ("B", "resign")),
])
def test_parse_re_recognised_forms(value, expected):
    assert SR.parse_re(value) == expected


@pytest.mark.parametrize("value", [None, "", "?", "Void", "B+T", "W+F", "B+", "+5.5", "B-3.5", "garbage", "B+R+5.5"])
def test_parse_re_unrecognised_forms_return_none(value):
    assert SR.parse_re(value) is None


def test_extract_re():
    text = "(;GM[1]SZ[9]RE[B+3.5]KM[7];B[ef];W[dd])"
    assert SR.extract_re(text) == "B+3.5"
    assert SR.extract_re("(;GM[1]SZ[9])") is None


# ---- build_results: SGF dir + positions file -> results JSONL ----

def _write_sgf(path, re_tag):
    with open(path, "w") as f:
        f.write(f"(;GM[1]SZ[9]RE[{re_tag}]KM[7];B[ef];W[dd])")


def _positions_row(game_id, source_file, turn=0):
    return {"schemaVersion": 1, "positionId": f"{game_id}-{turn}", "gameId": game_id, "boardSize": 9,
            "komi": 7, "rulesId": "cgos-area-psk-v1", "initialStones": [], "initialPlayer": "B",
            "moves": [], "turnNumber": turn, "toMove": "B" if turn % 2 == 0 else "W",
            "spatial": [0] * (9 * 9 * 32), "global": [0, 9 / 19, 0, 0], "legal": [1] * 82,
            "sourceFile": source_file}


def test_build_results_all_re_forms_and_no_duplicates(tmp_path):
    sgf_dir = tmp_path / "kifu"
    sgf_dir.mkdir()
    _write_sgf(sgf_dir / "resign.sgf", "B+R")
    _write_sgf(sgf_dir / "score.sgf", "W+3.5")
    _write_sgf(sgf_dir / "draw.sgf", "0")
    _write_sgf(sgf_dir / "unscored.sgf", "?")  # unrecognised: skipped, not guessed at
    # "timeout.sgf" is referenced by a position row but absent from disk: reported, not written.

    rows = [
        _positions_row("game-resign", "resign.sgf", turn=0),
        _positions_row("game-resign", "resign.sgf", turn=1),  # same game, second position: no duplicate output row
        _positions_row("game-score", "score.sgf"),
        _positions_row("game-draw", "draw.sgf"),
        _positions_row("game-unscored", "unscored.sgf"),
        _positions_row("game-missing", "timeout.sgf"),
    ]
    positions = tmp_path / "positions.jsonl"
    with open(positions, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "results.jsonl"
    report = SR.build_results(str(sgf_dir), str(positions), str(out))

    assert report["games"] == 5
    assert report["gamesWritten"] == 3
    assert report["skippedUnparsed"] == 1
    assert report["missingFile"] == 1
    assert report["byKind"] == {"resign": 1, "score": 1, "draw": 1}
    assert report["byResult"] == {"B": 1, "W": 1, "draw": 1}

    lines = [json.loads(line) for line in open(out)]
    assert len(lines) == 3
    assert len(lines) == len({l["gameId"] for l in lines})  # one line per game, no duplicates
    by_id = {l["gameId"]: l for l in lines}
    assert by_id["game-resign"] == {"gameId": "game-resign", "result": "B", "kind": "resign"}
    assert by_id["game-score"] == {"gameId": "game-score", "result": "W", "kind": "score"}
    assert by_id["game-draw"] == {"gameId": "game-draw", "result": "draw", "kind": "draw"}


def test_source_file_map_rejects_inconsistent_gameid(tmp_path):
    positions = tmp_path / "positions.jsonl"
    with open(positions, "w") as f:
        f.write(json.dumps(_positions_row("g1", "a.sgf")) + "\n")
        f.write(json.dumps(_positions_row("g1", "b.sgf")) + "\n")
    with pytest.raises(ValueError, match="two source files"):
        SR.build_results(str(tmp_path), str(positions), str(tmp_path / "out.jsonl"))


# ---- to-move conversion (black win / white win / draw), joined into a real dataset ----

def _game(gid_seed, moves):
    """Mirrors test_dataset.py's ``_game`` helper: one row per turn 0..len(moves). Each game gets
    a distinct komi so that no two games' turn-0 (empty board) position collides under dedupe --
    real corpora don't have dozens of games sharing byte-identical opening positions, and a
    collision here would make position-count assertions depend on which game happens to win the
    dedup priority instead of on the result-kind logic under test."""
    rows = []
    komi = 7 + (gid_seed % 1000) * 0.01
    gid = D.canonical_hash(S, komi, [], "B", moves)
    for t in range(len(moves) + 1):
        pid = D.canonical_hash(S, komi, [], "B", moves[:t])
        legal = [1] * (S * S + 1)
        for _, m in moves[:t]:
            legal[gtp_to_index(m, S)] = 0
        spatial = [0] * (S * S * 32)
        rows.append({"positionId": pid, "gameId": gid, "boardSize": S, "komi": komi, "rulesId": D.RULES_ID,
                     "initialStones": [], "initialPlayer": "B", "moves": moves[:t], "turnNumber": t,
                     "toMove": "B" if t % 2 == 0 else "W", "spatial": spatial, "global": [0, S / 19, t / 162, 0],
                     "legal": legal, "sourceFile": f"{gid_seed}.sgf"})
    return gid, rows


def _label(row):
    pol = [0.0] * (S * S + 1)
    legal_idx = [i for i, l in enumerate(row["legal"]) if l]
    pol[legal_idx[0]] = 1.0
    return {"positionId": row["positionId"], "gameId": row["gameId"], "turnNumber": row["turnNumber"], "toMove": row["toMove"],
            "boardSize": S, "policy": pol, "expectedResult": 0.5, "score": 0.0, "ownership": [0.0] * (S * S),
            "teacherVisits": 1, "sourceType": "teacher", "teacherId": "t"}


def test_sgf_results_feeds_dataset_build_with_correct_to_move_wdl(tmp_path):
    """The exact pipeline docs/spec/05-validation.md §5 describes: sgf-results' output plugs
    directly into ``build-data --results`` (``dataset.build_dataset``'s existing
    ``wdl_from_results``), and the resulting wdl one-hot must be to-move perspective: the winner's
    to-move positions get "win", the loser's get "loss"; a draw gets "draw" on both sides.

    Uses 60 synthetic games so train/validation/test are all non-empty (``_game``, matching
    test_dataset.py's pattern); only 3 of them ("black_wins"/"white_wins"/"drawn") have SGF files
    and thus a real result -- the other 57 keep the dataset's holdout non-empty without needing a
    results entry."""
    rng = np.random.default_rng(3)
    pts = [f"{c}{r}" for c in "ABCDEFGHJ" for r in range(1, 10)]
    rows = []
    game_ids = {}
    for g in range(60):
        seq = list(rng.permutation(pts)[:5])
        moves = [["B" if i % 2 == 0 else "W", m] for i, m in enumerate(seq)]
        gid, grows = _game(g, moves)
        game_ids[g] = gid
        rows += grows
    # re-tag three specific games' sourceFile/gameId so their SGF files can carry a real result
    named = {"black_wins": 0, "white_wins": 1, "drawn": 2}
    for name, g in named.items():
        for r in rows:
            if r["gameId"] == game_ids[g]:
                r["sourceFile"] = f"{name}.sgf"

    sgf_dir = tmp_path / "kifu"
    sgf_dir.mkdir()
    _write_sgf(sgf_dir / "black_wins.sgf", "B+R")
    _write_sgf(sgf_dir / "white_wins.sgf", "W+4.5")
    _write_sgf(sgf_dir / "drawn.sgf", "0")

    positions_path = tmp_path / "positions.jsonl"
    with open(positions_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    labels_path = tmp_path / "labels.jsonl"
    with open(labels_path, "w") as f:
        for r in rows:
            f.write(json.dumps(_label(r)) + "\n")

    results_path = tmp_path / "results.jsonl"
    report = SR.build_results(str(sgf_dir), str(positions_path), str(results_path))
    assert report["gamesWritten"] == 3

    wdl_from_results = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            wdl_from_results[r["gameId"]] = r["result"]

    out_dir = tmp_path / "ds"
    manifest = D.build_dataset(str(positions_path), str(labels_path), str(out_dir), {}, wdl_from_results=wdl_from_results)
    assert manifest["provenance"]["buildReport"]["wdl"] == 6 * 3  # 6 positions/game (turns 0..5) x 3 games with a result

    by_game: dict[str, list[tuple[int, list]]] = {}
    for split in ("train", "validation", "test"):
        for _, a in D.iter_split(str(out_dir), split, verify=False):
            wm = a["target_mask"][:, 4] == 1
            for i in np.nonzero(wm)[0]:
                gid_hex = bytes(a["game_id"][i]).hex()
                by_game.setdefault(gid_hex, []).append((int(i), a["wdl"][i].tolist()))

    black_rows = [r for r in rows if r["gameId"] == game_ids[0]]
    white_rows = [r for r in rows if r["gameId"] == game_ids[1]]
    draw_rows = [r for r in rows if r["gameId"] == game_ids[2]]
    assert len(by_game[game_ids[0]]) == len(black_rows) == 6
    assert len(by_game[game_ids[1]]) == len(white_rows) == 6
    assert len(by_game[game_ids[2]]) == len(draw_rows) == 6

    # turnNumber t's toMove is "B" if t even else "W" (both games start with initialPlayer "B",
    # see `_game`): black won -> even turns (B to move) get win, odd turns (W to move) get loss.
    for t in range(6):
        to_move = "B" if t % 2 == 0 else "W"
        expected_black_win_game = [1, 0, 0] if to_move == "B" else [0, 0, 1]
        expected_white_win_game = [1, 0, 0] if to_move == "W" else [0, 0, 1]
        # match by turn via the recorded row list order (rows were appended turn 0..5 in order)
        assert by_game[game_ids[0]][t][1] == expected_black_win_game
        assert by_game[game_ids[1]][t][1] == expected_white_win_game
        assert by_game[game_ids[2]][t][1] == [0, 1, 0]
