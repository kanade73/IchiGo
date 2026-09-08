"""End-to-end CGOS integration tests: two real `ichigo gtp` engines, each driven by a real
`ichigo_cgos_client.py` subprocess, playing through `fake_cgos_server.py` (docs/spec/04-tasks.md
T30 acceptance: "まず9路20局、途中切断replay、analysisが保存される、対局途中model差替えなし、
既存RinGoに干渉しない" -- scaled down to a handful of games for CI, since T30's own gate is a
20-game *manual/M2a* run, not a unit test). Uses `--visits 4` (docs/spec/04-tasks.md T30 test
plan) and a small per-game time budget so genmove's deadline controller (docs/spec/03-engine.md
§8) keeps each move fast; the server's own time control is otherwise ignored by this fixture
(neither side is expected to time out).
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

from conftest import engine_argv, spawn_client
from fake_cgos_server import FakeCGOSServer

# Small main time so genmove's per-move deadline budget (docs/spec/03-engine.md §8) stays well
# under a second even early in the game, keeping full games fast without starving the search.
TEST_LEVEL_MS = 8_000
GAME_TIMEOUT_SECONDS = 180.0

# An SGF-style RE result: "Draw" or "<colour>+<reason-or-score>". Also guards against the
# `gameover <date> <result>` fake-server wire bug this suite once caught -- a date string
# containing a space (e.g. "%Y-%m-%d %H:%M:%S") shifted every field after it, so the client ended
# up treating a *time-of-day* fragment as the result. A plain `assert game.result` (non-empty)
# would not have caught that; this shape check would.
RESULT_PATTERN = re.compile(r"^(Draw|[BW]\+(Resign|Time|Illegal|\d+(\.\d+)?))$")


def _write_client_config(
    tmp_path: Path,
    *,
    name: str,
    port: int,
    username: str,
    password: str,
    argv: list[str],
    max_games: int,
) -> Path:
    password_file = tmp_path / name / "password.txt"
    password_file.parent.mkdir(parents=True, exist_ok=True)
    password_file.write_text(password + "\n", encoding="utf-8")

    config = {
        "host": "127.0.0.1",
        "port": port,
        "username": username,
        "password_file": "password.txt",
        "board_size": 9,
        "engine_argv": argv,
        "analysis": True,
        "state_dir": str(tmp_path / name / "state"),
        "log_dir": str(tmp_path / name / "logs"),
        "max_games": max_games,
    }
    config_path = tmp_path / name / "cgos.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def _wait_exit(proc: subprocess.Popen, timeout: float) -> int:
    return proc.wait(timeout=timeout)


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


@pytest.fixture()
def two_engine_argv(release_binary, model_path):
    return (
        engine_argv(release_binary, model_path, visits=4),
        engine_argv(release_binary, model_path, visits=4),
    )


def test_two_full_games_reach_gameover_with_analysis(tmp_path, two_engine_argv):
    """docs/spec/04-tasks.md T30 test plan: "run 2 complete 9x9 games ... assert both games reach
    gameover with results, analysis strings were received for our moves"."""
    argv_a, argv_b = two_engine_argv
    server = FakeCGOSServer(
        accounts={"ichigo-a": "pw-a", "ichigo-b": "pw-b"}, level_ms=TEST_LEVEL_MS, max_moves=250,
    )
    server.start()
    proc_a = proc_b = None
    try:
        cfg_a = _write_client_config(
            tmp_path, name="a", port=server.port, username="ichigo-a", password="pw-a", argv=argv_a, max_games=2,
        )
        cfg_b = _write_client_config(
            tmp_path, name="b", port=server.port, username="ichigo-b", password="pw-b", argv=argv_b, max_games=2,
        )
        proc_a = spawn_client(cfg_a, games=2)
        proc_b = spawn_client(cfg_b, games=2)

        games = server.wait_for_completed_games(2, timeout=GAME_TIMEOUT_SECONDS)
        assert len(games) == 2
        for game in games:
            assert RESULT_PATTERN.match(game.result), game.result
            assert {game.white, game.black} == {"ichigo-a", "ichigo-b"}
            assert len(game.moves) > 0

        # Colours alternated between the two games (pairing alternates white/black; docs/spec
        # /05-validation.md §6 "色交換" spirit, even though this fixture isn't the promotion
        # league itself).
        assert {games[0].white, games[0].black} == {games[1].white, games[1].black}
        assert games[0].white != games[1].white

        analysis = server.analysis_log()
        assert len(analysis) > 0, "expected at least one analysis string from kata-genmove_analyze"
        for entry in analysis:
            parsed = json.loads(entry["analysis"])
            assert isinstance(parsed, dict)
            assert "moves" in parsed and isinstance(parsed["moves"], list) and parsed["moves"]
            assert "move" in parsed["moves"][0]
            assert "winrate" in parsed["moves"][0]

        rc_a = _wait_exit(proc_a, timeout=30)
        rc_b = _wait_exit(proc_b, timeout=30)
        assert rc_a == 0
        assert rc_b == 0
        proc_a = proc_b = None
    finally:
        for p in (proc_a, proc_b):
            if p is not None:
                _terminate(p)
        server.stop()


def test_disconnect_then_setup_replay_resumes_without_duplicated_moves(tmp_path, two_engine_argv):
    """docs/spec/04-tasks.md T30 test plan: "a mid-game disconnect + setup replay resumes without
    duplicated moves"."""
    argv_a, argv_b = two_engine_argv
    server = FakeCGOSServer(
        accounts={"ichigo-a": "pw-a", "ichigo-b": "pw-b"}, level_ms=TEST_LEVEL_MS, max_moves=250,
    )
    server.start()
    proc_a = proc_b = None
    try:
        cfg_a = _write_client_config(
            tmp_path, name="a", port=server.port, username="ichigo-a", password="pw-a", argv=argv_a, max_games=1,
        )
        cfg_b = _write_client_config(
            tmp_path, name="b", port=server.port, username="ichigo-b", password="pw-b", argv=argv_b, max_games=1,
        )

        server.inject_disconnect("ichigo-a", before_move_number=2)

        proc_a = spawn_client(cfg_a, games=1)
        proc_b = spawn_client(cfg_b, games=1)

        games = server.wait_for_completed_games(1, timeout=GAME_TIMEOUT_SECONDS)
        game = games[0]
        assert RESULT_PATTERN.match(game.result), game.result

        # No (colour, coordinate) pair repeats back-to-back -- the signature of a double-apply
        # at the server's own authoritative move list.
        for a, b in zip(game.moves, game.moves[1:]):
            assert not (a.color == b.color and a.coord == b.coord)

        # Cross-check against client A's own persisted ledger (docs/spec/03-engine.md §9: "同じ
        # 着手を二重適用しない" / T30: "keep a move ledger in the state dir"): the count of A's
        # own moves recorded in its ledger must equal the count of A's moves the server actually
        # has, i.e. nothing was replayed twice into the ledger either.
        rc_a = _wait_exit(proc_a, timeout=30)
        rc_b = _wait_exit(proc_b, timeout=30)
        assert rc_a == 0
        assert rc_b == 0
        proc_a = proc_b = None

        a_color = game.color_for("ichigo-a")
        server_move_count_for_a = sum(1 for m in game.moves if m.color == a_color)

        ledger_dir = tmp_path / "a" / "state" / "games"
        ledger_files = list(ledger_dir.glob("*.json"))
        assert len(ledger_files) == 1, f"expected exactly one ledger file, found {ledger_files}"
        ledger = json.loads(ledger_files[0].read_text(encoding="utf-8"))
        ledger_move_count_for_a = sum(1 for m in ledger["moves"] if m["color"] == a_color)
        assert ledger_move_count_for_a == server_move_count_for_a
        assert ledger["result"] == game.result
    finally:
        for p in (proc_a, proc_b):
            if p is not None:
                _terminate(p)
        server.stop()


def test_sigterm_stops_only_after_the_current_game(tmp_path, two_engine_argv):
    """docs/spec/04-tasks.md T30 test plan: "a SIGTERM during a game stops only after that
    game"."""
    argv_a, argv_b = two_engine_argv
    server = FakeCGOSServer(
        accounts={"ichigo-a": "pw-a", "ichigo-b": "pw-b"}, level_ms=TEST_LEVEL_MS, max_moves=250,
    )
    server.start()
    proc_a = proc_b = None
    try:
        cfg_a = _write_client_config(
            tmp_path, name="a", port=server.port, username="ichigo-a", password="pw-a", argv=argv_a, max_games=None,
        )
        cfg_b = _write_client_config(
            tmp_path, name="b", port=server.port, username="ichigo-b", password="pw-b", argv=argv_b, max_games=None,
        )
        # max_games=None in the config, but the CLI --games flag (spawn_client) still caps B so
        # the test can clean it up deterministically; A is signalled directly instead.
        proc_a = spawn_client(cfg_a, games=5)
        proc_b = spawn_client(cfg_b, games=5)

        # Let game 1 finish, so SIGTERM below lands squarely inside game 2 (mid-game), not at the
        # ambiguous boundary right at process start.
        server.wait_for_completed_games(1, timeout=GAME_TIMEOUT_SECONDS)
        server.wait_for_move_count("2", n=1, timeout=GAME_TIMEOUT_SECONDS)

        sigterm_sent_at = time.monotonic()
        proc_a.send_signal(signal.SIGTERM)

        # It must not die immediately -- the whole point is that it keeps playing game 2 out.
        time.sleep(0.5)
        assert proc_a.poll() is None, "client exited immediately on SIGTERM instead of finishing the game"

        rc_a = _wait_exit(proc_a, timeout=GAME_TIMEOUT_SECONDS)
        stopped_at = time.monotonic()
        assert rc_a == 0

        games = server.wait_for_completed_games(2, timeout=10)
        assert len(games) == 2
        game_2 = next(g for g in games if g.gid == "2")
        assert game_2.finished_at is not None
        # The game that was in progress when we signalled did complete, and only after that did
        # the process exit -- i.e. the stop took effect strictly between games, never mid-game.
        assert game_2.started_at <= sigterm_sent_at + 0.5
        assert stopped_at >= sigterm_sent_at

        # And it must not have gone on to a 3rd game.
        time.sleep(1.0)
        assert len(server.completed_games()) == 2
    finally:
        if proc_a is not None and proc_a.poll() is None:
            _terminate(proc_a)
        if proc_b is not None:
            _terminate(proc_b)
        server.stop()
