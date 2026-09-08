"""Fast, deterministic protocol-only tests for fake_cgos_server.py, using plain-socket stub
clients instead of real engines (the real-engine end-to-end tests live in
test_cgos_integration.py). Covers pairing (white/black order, black-first), move relay, analysis
logging, double-pass gameover, and disconnect+resume replay without duplicated moves -- the same
scenarios docs/spec/04-tasks.md T30 asks for, at a fraction of the wall-clock cost.
"""

from __future__ import annotations

import socket
import threading
from typing import List, Optional

import pytest

from fake_cgos_server import FakeCGOSServer


class StubClient:
    """A minimal hand-rolled CGOS player: replies mechanically from a scripted move queue."""

    def __init__(self, host: str, port: int, username: str, password: str, moves: List[str]):
        self.host, self.port = host, port
        self.username, self.password = username, password
        self.moves = list(moves)
        self.log: List[str] = []
        self.setup_lines: List[str] = []
        self.done = threading.Event()
        self.disconnected = threading.Event()
        self._connect()

    def _connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
        self.rfile = self.sock.makefile("r", encoding="utf-8", newline="\n")
        self.wfile = self.sock.makefile("w", encoding="utf-8", newline="\n")

    def _send(self, text: str) -> None:
        self.wfile.write(text + "\n")
        self.wfile.flush()

    def _readline(self) -> Optional[str]:
        line = self.rfile.readline()
        return None if line == "" else line.rstrip("\n")

    def run(self) -> None:
        try:
            while True:
                line = self._readline()
                if line is None:
                    self.disconnected.set()
                    return
                self.log.append(line)
                parts = line.split()
                cmd = parts[0]
                if cmd == "protocol":
                    self._send("e1 stub 1.0 genmove_analyze")
                elif cmd == "username":
                    self._send(self.username)
                elif cmd == "password":
                    self._send(self.password)
                elif cmd == "setup":
                    self.setup_lines.append(line)
                elif cmd == "play":
                    pass
                elif cmd == "genmove":
                    mv = self.moves.pop(0) if self.moves else "pass"
                    self._send(mv if mv == "pass" else f'{mv} {{"stub":true}}')
                elif cmd == "gameover":
                    self._send("quit")
                    self.done.set()
                    return
                elif cmd == "info":
                    pass
        except OSError:
            self.disconnected.set()

    def reconnect(self, moves: List[str]) -> None:
        self._connect()
        self.moves = list(moves)


@pytest.fixture()
def server():
    srv = FakeCGOSServer(accounts={"p1": "pw1", "p2": "pw2"}, level_ms=30_000, max_moves=20)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def test_black_moves_first_and_setup_names_white_then_black(server):
    p1 = StubClient("127.0.0.1", server.port, "p1", "pw1", moves=["D4", "pass", "pass"])
    p2 = StubClient("127.0.0.1", server.port, "p2", "pw2", moves=["Q4", "pass", "pass"])
    t1 = threading.Thread(target=p1.run)
    t2 = threading.Thread(target=p2.run)
    t1.start()
    t2.start()
    assert p1.done.wait(5) and p2.done.wait(5)

    games = server.wait_for_completed_games(1, timeout=5)
    game = games[0]
    assert game.result == "Draw"
    assert [(m.color, m.coord) for m in game.moves] == [
        ("b", "D4" if game.black == "p1" else "Q4"),
        ("w", "Q4" if game.black == "p1" else "D4"),
        ("b", "pass"),
        ("w", "pass"),
    ]
    # setup names white before black (verified against the pinned server's own formatting).
    setup = p1.setup_lines[0].split()
    assert setup[0] == "setup"
    # setup <gid> <boardsize> <komi> <time_ms> <white>(<rating>) <black>(<rating>) [...]
    white_name = setup[5].split("(")[0]
    black_name = setup[6].split("(")[0]
    assert {white_name, black_name} == {"p1", "p2"}
    assert game.white == white_name and game.black == black_name


def test_analysis_strings_are_recorded(server):
    p1 = StubClient("127.0.0.1", server.port, "p1", "pw1", moves=["D4", "pass", "pass"])
    p2 = StubClient("127.0.0.1", server.port, "p2", "pw2", moves=["Q4", "pass", "pass"])
    threading.Thread(target=p1.run).start()
    threading.Thread(target=p2.run).start()
    server.wait_for_completed_games(1, timeout=5)

    log = server.analysis_log()
    assert len(log) >= 2
    for entry in log:
        assert entry["analysis"] == '{"stub":true}'
        assert entry["color"] in ("b", "w")


def test_disconnect_then_setup_replay_resumes_without_duplicated_moves(server):
    server.inject_disconnect("p1", before_move_number=2)
    p1 = StubClient("127.0.0.1", server.port, "p1", "pw1", moves=["D4", "E4", "F4"])
    p2 = StubClient("127.0.0.1", server.port, "p2", "pw2", moves=["Q4", "Q5", "Q6", "pass", "pass"])
    threading.Thread(target=p1.run).start()
    threading.Thread(target=p2.run).start()

    assert p1.disconnected.wait(5)
    p1.reconnect(moves=["F4", "G4"])
    threading.Thread(target=p1.run).start()

    assert p1.done.wait(5) and p2.done.wait(5)
    games = server.wait_for_completed_games(1, timeout=5)
    game = games[0]

    coords = [m.coord for m in game.moves]
    assert len(game.moves) == 8  # both scripts are "3 real moves then pass, pass" either way round
    # No (colour, coordinate) pair repeats back-to-back -- the signature of a double-apply.
    for a, b in zip(game.moves, game.moves[1:]):
        assert not (a.color == b.color and a.coord == b.coord)
    assert coords[-2:] == ["pass", "pass"]
    assert coords.count("pass") == 2  # exactly the final double-pass, not replayed extra passes

    # The resume setup line carried exactly the moves played before the disconnect (no more, no
    # fewer): p1's own first move, plus every move the opponent made in between -- one fewer if
    # p1 is white (opponent already had one move in before p1's first) than if p1 is black.
    p1_color = "b" if game.black == "p1" else "w"
    expected_replayed = 2 if p1_color == "b" else 3
    resume_setup = p1.setup_lines[-1].split()
    move_time_tokens = resume_setup[7:]
    assert len(move_time_tokens) % 2 == 0
    replayed_moves = move_time_tokens[0::2]
    assert len(replayed_moves) == expected_replayed
