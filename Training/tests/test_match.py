import json
import os
import random
import sys

import pytest

from ichigo_train.__main__ import EXIT_CONFIG
from ichigo_train.match import (
    GTPClient,
    GTPError,
    GTPTimeout,
    MatchRunner,
    UniformBaselinePlayer,
    gtp_to_xy,
    index_to_gtp,
    load_openings,
    parse_engine_spec,
    sgf_point,
)

FAKE = os.path.join(os.path.dirname(__file__), "fake_gtp_engine.py")


def fake(*flags):
    return [sys.executable, FAKE, *flags]


# -- coordinates -----------------------------------------------------------------------------

def test_coordinate_round_trip_matches_swift_convention():
    # Sources/IchiGoFeatures/Coordinates.swift: (0,0) on 9x9 -> "A9"; letter I skipped.
    assert index_to_gtp(0, 9) == "A9"
    assert index_to_gtp(8 * 9 + 8, 9) == "J1"
    assert gtp_to_xy("A9", 9) == (0, 0)
    assert gtp_to_xy("J1", 9) == (8, 8)
    assert gtp_to_xy("pass", 9) is None
    assert sgf_point("A9", 9) == "aa"
    assert sgf_point("pass", 9) == ""
    # I is skipped: the 9th letter is J, not I.
    assert "I" not in [index_to_gtp(i, 19) for i in range(19)]


# -- engine spec parsing (no naive string split) ----------------------------------------------

def test_parse_engine_spec_uniform():
    assert parse_engine_spec("uniform") == ("uniform", None)
    assert parse_engine_spec("Uniform") == ("uniform", None)


def test_parse_engine_spec_json_array():
    kind, argv = parse_engine_spec('["/path/to/ichigo","gtp","--model-9","m.ichigo","--visits","100"]')
    assert kind == "gtp"
    assert argv == ["/path/to/ichigo", "gtp", "--model-9", "m.ichigo", "--visits", "100"]


def test_parse_engine_spec_shlex_handles_quoted_spaces():
    kind, argv = parse_engine_spec('/path/to/ichigo gtp --model-9 "m with space.ichigo" --visits 100')
    assert kind == "gtp"
    assert argv[-3] == "m with space.ichigo"


def test_parse_engine_spec_empty_raises():
    with pytest.raises(ValueError):
        parse_engine_spec("   ")


# -- openings ---------------------------------------------------------------------------------

def test_load_openings_none_is_empty_board():
    assert load_openings(None) == [{"id": "none", "moves": []}]
    assert load_openings("none") == [{"id": "none", "moves": []}]


def test_load_openings_from_file(tmp_path):
    path = tmp_path / "openings.jsonl"
    path.write_text('{"id": "op0", "moves": ["E5"]}\n{"id": "op1", "moves": ["C3", "G7"]}\n')
    openings = load_openings(str(path))
    assert openings == [{"id": "op0", "moves": ["E5"]}, {"id": "op1", "moves": ["C3", "G7"]}]


# -- GTPClient ----------------------------------------------------------------------------------

def test_gtpclient_rejects_non_list_argv():
    with pytest.raises(ValueError):
        GTPClient("ichigo gtp --model-9 m.ichigo")  # naive string, not an argv list


def test_gtpclient_basic_roundtrip():
    c = GTPClient(fake(), name="fake")
    try:
        assert c.send("protocol_version").strip() == "2"
        c.boardsize(9)
        c.clear_board()
        c.komi(7)
        assert c.play("B", "D4") is True
        assert c.final_score() == "0"
    finally:
        c.close()
    assert c.proc.poll() is not None


def test_gtpclient_play_rejection_returns_false_not_raise():
    c = GTPClient(fake("--reject-play", "D4"), name="fake")
    try:
        c.boardsize(9)
        c.clear_board()
        assert c.play("B", "D4") is False
        assert c.play("B", "E5") is True
    finally:
        c.close()


def test_gtpclient_timeout_then_kill():
    c = GTPClient(fake("--sleep-genmove", "5"), name="slow", timeout=0.3)
    try:
        with pytest.raises(GTPTimeout):
            c.genmove("B")
    finally:
        c.kill()
    assert c.proc.poll() is not None


def test_gtpclient_unknown_command_raises_gtperror():
    c = GTPClient(fake(), name="fake")
    try:
        with pytest.raises(GTPError):
            c.send("not_a_real_command")
    finally:
        c.close()


# -- UniformBaselinePlayer ---------------------------------------------------------------------

def test_uniform_baseline_reproducible_given_same_seed():
    def run_once(seed):
        c = GTPClient(fake(), name="oracle")
        try:
            c.boardsize(9)
            c.clear_board()
            baseline = UniformBaselinePlayer(size=9, rng=random.Random(seed))
            return [baseline.choose_move(c, "B") for _ in range(6)]
        finally:
            c.close()

    assert run_once(123) == run_once(123)


def test_uniform_baseline_falls_back_to_pass_when_everything_rejected():
    c = GTPClient(fake(), name="oracle")
    # The fake's --reject-play only rejects one fixed vertex; simulate "no legal move anywhere"
    # (every candidate rejected) by monkeypatching play() directly instead.
    c.play = lambda color, vertex, timeout=None: False  # type: ignore[method-assign]
    try:
        baseline = UniformBaselinePlayer(size=3, rng=random.Random(1))
        assert baseline.choose_move(c, "B") == "pass"
    finally:
        c.close()


# -- MatchRunner end-to-end with fake engines ---------------------------------------------------

def test_two_game_fake_vs_fake_match_end_to_end(tmp_path):
    engine_a = ("gtp", fake("--moves", "E5,C3", "--result", "B+3.5", "--model-hash", "aaaa1111"))
    engine_b = ("gtp", fake("--moves", "G7,F3", "--result", "B+3.5", "--model-hash", "bbbb2222"))
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=engine_b, games=2, size=9, komi=7.0, out_dir=str(out), seed=1, max_moves=20)
    report = runner.run()

    assert report["games"] == 2
    assert report["countedGames"] == 2
    games = [json.loads(l) for l in open(out / "games.jsonl")]
    assert len(games) == 2
    # colour swap: game 0 has A black, game 1 has B black, same opening ("none").
    assert games[0]["colours"]["black"] == "A" and games[0]["colours"]["white"] == "B"
    assert games[1]["colours"]["black"] == "B" and games[1]["colours"]["white"] == "A"
    # Turns alternate colour, not grouped by engine: game 0 is black=A/white=B so black's two
    # scripted moves (E5,C3) interleave with white's (G7,F3) as E5,G7,C3,F3; game 1 swaps colour
    # so black=B leads with its own script first (both engines' move_i resets on clear_board).
    assert games[0]["moves"] == ["E5", "G7", "C3", "F3", "pass", "pass"]
    assert games[1]["moves"] == ["G7", "E5", "F3", "C3", "pass", "pass"]
    for g in games:
        assert g["result"] == "B+3.5"
        assert g["countedResult"] is True
        assert g["modelHashes"] == {"A": "aaaa1111", "B": "bbbb2222"}
        assert len(g["moveTimesSeconds"]) == len(g["moves"])
        assert os.path.exists(out / g["sgfPath"])
        sgf = (out / g["sgfPath"]).read_text()
        assert sgf.startswith("(;GM[1]FF[4]SZ[9]KM[7") and "RE[B+3.5]" in sgf
    assert os.path.exists(out / "report.json")
    # Both games score "B+3.5" (whichever engine is black wins): game 0 A is black -> A wins;
    # game 1 B is black (colour swap) -> A loses. Net: 1 win, 1 loss, mean score 0.5.
    report_json = json.load(open(out / "report.json"))
    assert report_json["wins"] == 1 and report_json["losses"] == 1
    assert report_json["meanScore"] == 0.5


def test_timeout_incident_recorded_and_match_continues(tmp_path):
    engine_a = ("gtp", fake("--moves", "E5", "--result", "B+3.5"))
    engine_b = ("gtp", fake("--sleep-genmove", "5", "--result", "B+3.5"))
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=engine_b, games=2, size=9, komi=7.0, out_dir=str(out), seed=1, max_moves=20, genmove_timeout=0.4)
    report = runner.run()

    games = [json.loads(l) for l in open(out / "games.jsonl")]
    assert len(games) == 2  # a timeout aborts one game, the match still plays the next one
    for g in games:
        assert g["result"] == "timeout"
        assert g["countedResult"] is False
        assert any(inc["type"] == "timeout" for inc in g["incidents"])
    assert report["incidents"]["timeouts"] == 2
    assert report["countedGames"] == 0


def test_illegal_move_incident_aborts_game(tmp_path):
    engine_a = ("gtp", fake("--moves", "E5,C3", "--result", "B+3.5"))
    engine_b = ("gtp", fake("--moves", "G7,F3", "--result", "B+3.5", "--reject-play", "C3"))
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=engine_b, games=1, size=9, komi=7.0, out_dir=str(out), seed=1, max_moves=20)
    report = runner.run()

    games = [json.loads(l) for l in open(out / "games.jsonl")]
    assert games[0]["result"] == "illegal_move"
    assert games[0]["countedResult"] is False
    assert any(inc["type"] == "illegal_move" for inc in games[0]["incidents"])
    assert games[0]["moves"] == ["E5", "G7"]  # C3 (rejected) is never appended
    assert report["incidents"]["illegalMoves"] == 1


def test_openings_reused_across_pairs_with_colour_swap(tmp_path):
    openings_path = tmp_path / "openings.jsonl"
    openings_path.write_text('{"id": "op0", "moves": ["E5"]}\n{"id": "op1", "moves": ["C3", "G7"]}\n')
    engine_a = ("gtp", fake("--moves", "D4,F6", "--result", "0"))
    engine_b = ("gtp", fake("--moves", "C4,F5", "--result", "0"))
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=engine_b, games=4, size=9, komi=7.0, out_dir=str(out), seed=1,
                          openings_path=str(openings_path), max_moves=20)
    report = runner.run()

    games = [json.loads(l) for l in open(out / "games.jsonl")]
    assert [g["openingId"] for g in games] == ["op0", "op0", "op1", "op1"]
    assert [g["colours"]["black"] for g in games] == ["A", "B", "A", "B"]
    assert games[0]["openingMoves"] == ["E5"]
    assert games[2]["openingMoves"] == ["C3", "G7"]
    assert report["openingsFileHash"] != "none"


def test_match_runner_rejects_both_engines_uniform(tmp_path):
    with pytest.raises(ValueError):
        MatchRunner(engine_a=("uniform", None), engine_b=("uniform", None), games=1, out_dir=str(tmp_path / "out"), seed=1)


def test_match_runner_with_uniform_baseline_smoke(tmp_path):
    engine_a = ("gtp", fake("--result", "0"))  # no scripted moves: always passes when asked
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=("uniform", None), games=1, size=9, komi=7.0, out_dir=str(out), seed=7, max_moves=8)
    report = runner.run()

    games = [json.loads(l) for l in open(out / "games.jsonl")]
    assert len(games) == 1
    g = games[0]
    assert g["colours"]["black"] == "A" and g["colours"]["white"] == "uniform"
    assert len(g["moves"]) > 0
    assert g["incidents"] == []  # the fake oracle never rejects, so no crash/illegal/timeout incidents
    assert os.path.exists(out / g["sgfPath"])
    assert report["engines"]["B"]["kind"] == "uniform"


def test_visits_flag_conflicts_with_argv_visits_is_rejected(tmp_path):
    engine_a = ("gtp", fake() + ["--visits", "50"])
    engine_b = ("gtp", fake())
    out = tmp_path / "out"
    runner = MatchRunner(engine_a=engine_a, engine_b=engine_b, games=1, out_dir=str(out), seed=1, visits_a=100)
    with pytest.raises(ValueError):
        runner.run()


# -- CLI wiring (python -m ichigo_train match) --------------------------------------------------

def _fake_cmdline(*flags):
    return " ".join([sys.executable, FAKE, *flags])


def test_cli_match_subcommand_end_to_end(tmp_path):
    from ichigo_train.__main__ import main

    engine_a = _fake_cmdline("--moves", "D4,Q16", "--result", "B+3.5")
    out = tmp_path / "out"
    rc = main([
        "match", "--engine-a", engine_a, "--engine-b", "uniform", "--games", "1",
        "--size", "9", "--komi", "7", "--openings", "none", "--out", str(out), "--seed", "3", "--max-moves", "6",
    ])
    assert rc == 0
    assert (out / "report.json").exists()
    assert (out / "games.jsonl").exists()


def test_cli_match_rejects_time_and_visits_together(tmp_path, capsys):
    from ichigo_train.__main__ import main

    rc = main([
        "match", "--engine-a", "uniform", "--engine-b", "uniform", "--games", "1",
        "--out", str(tmp_path / "out"), "--seed", "1", "--visits-a", "100", "--time-main-seconds", "5",
    ])
    assert rc == EXIT_CONFIG
    assert "cannot be combined" in capsys.readouterr().err


def test_cli_match_reports_config_error_for_double_uniform(tmp_path, capsys):
    from ichigo_train.__main__ import main

    rc = main(["match", "--engine-a", "uniform", "--engine-b", "uniform", "--games", "1", "--out", str(tmp_path / "out"), "--seed", "1"])
    assert rc == EXIT_CONFIG
    assert "uniform" in capsys.readouterr().err
