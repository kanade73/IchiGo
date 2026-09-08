#!/usr/bin/env python3
"""Minimal fake CGOS server for local/CI tests (docs/spec/04-tasks.md T30,
docs/spec/05-validation.md §2 "運用": "切断setup", "ログrotate", "別state干渉なし" fixtures need a
server that can actually run two GTP-backed clients through a full game and inject a disconnect).

Deliberately does *not* implement Go rules (docs/spec/04-tasks.md T30: "fake serverはGo rules不要
... 両側とも実エンジンにする"). It pairs exactly two logged-in clients per game and relays moves
between them; each side is a real ``ichigo gtp`` engine via ``ichigo_cgos_client.py``, so both
sides already enforce legality themselves. A game ends when: both sides pass in a row (reported
as a flat "Draw" -- this fake server cannot score, it just needs *a* valid CGOS result string),
either side sends "resign", or the game exceeds ``max_moves`` (an anti-hang guard for this test
fixture only -- *not* a stand-in for any real server's move-limit policy; docs/spec/03-engine.md
§9 explicitly says not to copy RinGo's 400-move tournament rule onto every server).

Wire protocol: same grammar as ``ichigo_cgos_client.py`` documents in its module docstring
(pinned to https://github.com/zakki/cgos revision 4dcff8754400fa43a71323ffc3d2c5eb4a19f7f1); this
server plays the server side of that exchange -- ``protocol genmove_analyze`` first, then
``username``/``password``, then ``setup``/``genmove``/``play``/``gameover``, with WHITE named
before BLACK in ``setup`` and black always moving first. See that module's docstring for the
full grammar and citations; not repeated here.

Usable two ways:
  - Imported directly by tests as ``FakeCGOSServer`` (this is the intended use -- it runs the
    listener on a background thread so a synchronous test can drive real client subprocesses and
    poll/inspect server state at the same time; see ``inject_disconnect``,
    ``wait_for_completed_games``, ``analysis_log``).
  - Run standalone for manual/local testing: ``python3 fake_cgos_server.py --port 6867
    --account name:pass ...``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import socket
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

DEFAULT_BOARD_SIZE = 9
DEFAULT_KOMI = "7.0"
DEFAULT_LEVEL_MS = 30_000
DEFAULT_MAX_MOVES = 320


@dataclasses.dataclass
class MoveRecord:
    color: str
    coord: str
    time_left_ms: int
    analysis: Optional[str]


@dataclasses.dataclass
class GameRecord:
    gid: str
    white: str
    black: str
    boardsize: int
    komi: str
    level_ms: int
    moves: List[MoveRecord] = dataclasses.field(default_factory=list)
    result: Optional[str] = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    finished_at: Optional[float] = None

    def to_move(self) -> str:
        return "b" if len(self.moves) % 2 == 0 else "w"

    def username_for(self, color: str) -> str:
        return self.white if color == "w" else self.black

    def color_for(self, username: str) -> Optional[str]:
        if username == self.white:
            return "w"
        if username == self.black:
            return "b"
        return None


class _LineSocket:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._rfile = sock.makefile("r", encoding="utf-8", newline="\n")
        self._wfile = sock.makefile("w", encoding="utf-8", newline="\n")

    def readline(self) -> Optional[str]:
        try:
            line = self._rfile.readline()
        except OSError:
            return None
        if line == "":
            return None
        return line.rstrip("\r\n")

    def send_line(self, text: str) -> None:
        try:
            self._wfile.write(text + "\n")
            self._wfile.flush()
        except OSError:
            pass

    def close(self) -> None:
        for f in (self._rfile, self._wfile):
            try:
                f.close()
            except Exception:
                pass
        try:
            self._sock.close()
        except Exception:
            pass


class _PlayerConn:
    def __init__(self, sock: socket.socket, addr: Tuple[str, int]) -> None:
        self.sock = sock
        self.addr = addr
        self.io = _LineSocket(sock)
        self.username: Optional[str] = None
        self.analysis_capable = False
        self.gid: Optional[str] = None
        self.expect = "handshake"  # handshake -> username -> password -> genmove / post_gameover / None


class FakeCGOSServer:
    """A minimal, deterministic, in-process fake CGOS server. See module docstring."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        boardsize: int = DEFAULT_BOARD_SIZE,
        komi: str = DEFAULT_KOMI,
        level_ms: int = DEFAULT_LEVEL_MS,
        accounts: Optional[Dict[str, str]] = None,
        max_moves: int = DEFAULT_MAX_MOVES,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.boardsize = boardsize
        self.komi = komi
        self.level_ms = level_ms
        self.accounts = dict(accounts) if accounts else {}
        self.max_moves = max_moves
        self.log = log or logging.getLogger("fake_cgos_server")

        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._stopped = threading.Event()

        self._connections: Dict[str, _PlayerConn] = {}
        self._waiting: List[str] = []
        self._games: Dict[str, GameRecord] = {}
        self._completed_games: List[GameRecord] = []
        self._next_gid = 1
        self._disconnect_after: Dict[str, int] = {}
        self._analysis_log: List[dict] = []
        self._pair_game_count: Dict[frozenset, int] = {}

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept_loop, name="fake-cgos-accept", daemon=True)
        self._accept_thread.start()
        self.log.info("fake CGOS server listening on %s:%s", self.host, self.port)

    def stop(self) -> None:
        self._stopped.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._lock:
            conns = list(self._connections.values())
        for conn in conns:
            self._force_close(conn)
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)

    def __enter__(self) -> "FakeCGOSServer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- test-facing accessors -------------------------------------------------------

    def inject_disconnect(self, username: str, before_move_number: int) -> None:
        """Forcibly closes ``username``'s connection right before their own
        ``before_move_number``-th move (1-indexed, counting only this player's own moves in their
        current/next game -- colour-independent, since which colour a username is dealt depends
        on connection-race ordering during pairing). One-shot: fires once, then clears itself.
        The game stays open server-side (matching the real server: only the socket entry is
        dropped) so a later reconnect+login gets a resume ``setup`` with the full move history so
        far.
        """
        with self._lock:
            self._disconnect_after[username] = before_move_number

    def analysis_log(self) -> List[dict]:
        with self._lock:
            return list(self._analysis_log)

    def completed_games(self) -> List[GameRecord]:
        with self._lock:
            return list(self._completed_games)

    def wait_for_completed_games(self, n: int, timeout: float) -> List[GameRecord]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            games = self.completed_games()
            if len(games) >= n:
                return games
            time.sleep(0.05)
        raise TimeoutError(f"only {len(self.completed_games())} of {n} games completed within {timeout}s")

    def wait_for_move_count(self, gid: str, n: int, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                game = self._games.get(gid)
                count = len(game.moves) if game is not None else 0
            if count >= n:
                return count
            time.sleep(0.02)
        raise TimeoutError(f"game {gid} only reached {count} of {n} moves within {timeout}s")

    # -- accept / connection loop -----------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while not self._stopped.is_set():
            try:
                sock, addr = self._sock.accept()
            except OSError:
                return
            conn = _PlayerConn(sock, addr)
            t = threading.Thread(target=self._handle_connection, args=(conn,), name=f"fake-cgos-conn-{addr}", daemon=True)
            t.start()

    def _handle_connection(self, conn: _PlayerConn) -> None:
        conn.io.send_line("protocol genmove_analyze")
        try:
            while not self._stopped.is_set():
                line = conn.io.readline()
                if line is None:
                    return
                if line == "":
                    continue
                self._on_line(conn, line)
        finally:
            self._on_disconnect(conn)

    def _on_disconnect(self, conn: _PlayerConn) -> None:
        with self._lock:
            if conn.username is not None and self._connections.get(conn.username) is conn:
                del self._connections[conn.username]
            if conn.username in self._waiting:
                self._waiting.remove(conn.username)
        self.log.info("disconnected: %s %s", conn.username, conn.addr)
        conn.io.close()

    def _force_close(self, conn: _PlayerConn) -> None:
        try:
            conn.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.sock.close()
        except OSError:
            pass

    # -- dispatch -----------------------------------------------------------------

    def _on_line(self, conn: _PlayerConn, line: str) -> None:
        expect = conn.expect
        if expect == "handshake":
            self._handle_handshake(conn, line)
        elif expect == "username":
            self._handle_username(conn, line)
        elif expect == "password":
            self._handle_password(conn, line)
        elif expect == "genmove":
            self._handle_genmove_reply(conn, line)
        elif expect == "post_gameover":
            self._handle_post_gameover(conn, line)
        else:
            self.log.warning("unexpected line from %s (expect=%r): %r", conn.username or conn.addr, expect, line)

    def _handle_handshake(self, conn: _PlayerConn, line: str) -> None:
        conn.analysis_capable = "genmove_analyze" in line.split()
        conn.expect = "username"
        conn.io.send_line("username")

    def _handle_username(self, conn: _PlayerConn, line: str) -> None:
        conn.username = line.strip()
        conn.expect = "password"
        conn.io.send_line("password")

    def _handle_password(self, conn: _PlayerConn, line: str) -> None:
        password = line.strip()
        if self.accounts and self.accounts.get(conn.username) != password:
            conn.io.send_line("Error: Sorry, password doesn't match")
            self._force_close(conn)
            return

        with self._lock:
            self._connections[conn.username] = conn

            resume_gid = None
            for gid, game in self._games.items():
                if game.result is None and conn.username in (game.white, game.black):
                    resume_gid = gid
                    break

            if resume_gid is not None:
                game = self._games[resume_gid]
                conn.gid = resume_gid
                conn.expect = None
                move_pairs = " ".join(f"{m.coord} {m.time_left_ms}" for m in game.moves)
                setup = f"setup {resume_gid} {game.boardsize} {game.komi} {game.level_ms} {game.white}(0) {game.black}(0)"
                if move_pairs:
                    setup += f" {move_pairs}"
                conn.io.send_line(setup)
                self.log.info("resumed %s into game %s (%d moves replayed)", conn.username, resume_gid, len(game.moves))
                color = game.color_for(conn.username)
                if color is not None and game.to_move() == color and game.result is None:
                    self._request_move(game, color)
                return

            conn.expect = None
            if conn.username not in self._waiting:
                self._waiting.append(conn.username)
            self._try_pair()

    def _try_pair(self) -> None:
        with self._lock:
            while len(self._waiting) >= 2:
                a = self._waiting.pop(0)
                b = self._waiting.pop(0)
                conn_a = self._connections.get(a)
                conn_b = self._connections.get(b)
                if conn_a is None or conn_b is None:
                    # Stale entry (disconnected between queuing and pairing); drop it and retry.
                    if conn_a is not None:
                        self._waiting.insert(0, a)
                    if conn_b is not None:
                        self._waiting.insert(0, b)
                    continue
                # Alternate who plays white/black across successive games between this specific
                # pair of usernames (docs/spec/05-validation.md §6 "色交換"), keyed by the pair
                # itself rather than connection-race pop order so it is deterministic regardless
                # of which client happened to log in/reconnect first.
                key = frozenset((a, b))
                count = self._pair_game_count.get(key, 0)
                first, second = sorted((a, b))
                white, black = (first, second) if count % 2 == 0 else (second, first)
                self._pair_game_count[key] = count + 1
                self._start_game(white, black)

    def _start_game(self, white: str, black: str) -> None:
        gid = str(self._next_gid)
        self._next_gid += 1
        game = GameRecord(gid=gid, white=white, black=black, boardsize=self.boardsize, komi=self.komi, level_ms=self.level_ms)
        with self._lock:
            self._games[gid] = game
            for uname in (white, black):
                conn = self._connections.get(uname)
                if conn is not None:
                    conn.gid = gid
            setup = f"setup {gid} {self.boardsize} {self.komi} {self.level_ms} {white}(0) {black}(0)"
            for uname in (white, black):
                conn = self._connections.get(uname)
                if conn is not None:
                    conn.io.send_line(setup)
        self.log.info("game %s started: white=%s black=%s", gid, white, black)
        self._request_move(game, "b")

    def _request_move(self, game: GameRecord, color: str) -> None:
        uname = game.username_for(color)
        with self._lock:
            conn = self._connections.get(uname)
            if conn is None:
                # Currently disconnected; the resume path in _handle_password will pick this
                # back up (it checks game.to_move() against the reconnecting colour).
                return
            own_move_number = sum(1 for m in game.moves if m.color == color) + 1
            target = self._disconnect_after.get(uname)
            if target is not None and target == own_move_number:
                del self._disconnect_after[uname]
                self.log.info(
                    "test hook: disconnecting %s before their own move %d of game %s",
                    uname, own_move_number, game.gid,
                )
                self._force_close(conn)
                return
            conn.expect = "genmove"
            conn.io.send_line(f"genmove {color} {self.level_ms}")

    def _handle_genmove_reply(self, conn: _PlayerConn, line: str) -> None:
        with self._lock:
            game = self._games.get(conn.gid) if conn.gid is not None else None
            if game is None or game.result is not None:
                self.log.warning("genmove reply from %s for a finished/unknown game: %r", conn.username, line)
                return

            color = game.color_for(conn.username)
            if color is None or game.to_move() != color:
                self.log.warning(
                    "genmove reply from %s out of turn (to_move=%s): %r", conn.username, game.to_move(), line,
                )
                return

            tokens = line.split(None, 1)
            move = tokens[0]
            analysis = tokens[1] if len(tokens) > 1 else None
            conn.expect = None

            record = MoveRecord(color=color, coord=move, time_left_ms=self.level_ms, analysis=analysis)
            game.moves.append(record)
            if analysis:
                self._analysis_log.append({"gid": game.gid, "color": color, "move": move, "analysis": analysis})

            other_color = "w" if color == "b" else "b"
            other_uname = game.username_for(other_color)
            other_conn = self._connections.get(other_uname)
            if other_conn is not None:
                other_conn.io.send_line(f"play {color} {move} {self.level_ms}")

            if move.lower() == "resign":
                self._finish_game_locked(game, f"{'W' if color == 'b' else 'B'}+Resign")
                return

            if move.lower() == "pass" and len(game.moves) >= 2 and game.moves[-2].coord.lower() == "pass":
                self._finish_game_locked(game, "Draw")
                return

            if len(game.moves) >= self.max_moves:
                self.log.warning("game %s hit the fake-server move cap (%d); ending as Draw", game.gid, self.max_moves)
                self._finish_game_locked(game, "Draw")
                return

        self._request_move(game, other_color)

    def _finish_game_locked(self, game: GameRecord, result: str) -> None:
        """Caller must hold self._lock."""
        game.result = result
        game.finished_at = time.monotonic()
        # A single whitespace-free token: the wire protocol is whitespace-delimited, and a date
        # containing a space (e.g. "%Y-%m-%d %H:%M:%S") would silently shift every field after it.
        date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for uname in (game.white, game.black):
            conn = self._connections.get(uname)
            if conn is not None:
                conn.expect = "post_gameover"
                conn.io.send_line(f"gameover {date} {result}")
        self._completed_games.append(game)
        self.log.info("game %s over: %s (%d moves)", game.gid, result, len(game.moves))

    def _handle_post_gameover(self, conn: _PlayerConn, line: str) -> None:
        msg = line.strip()
        if msg == "ready":
            with self._lock:
                conn.gid = None
                conn.expect = None
                if conn.username not in self._waiting:
                    self._waiting.append(conn.username)
            self._try_pair()
        elif msg == "quit":
            self._force_close(conn)
        else:
            self.log.warning("unexpected post-gameover message from %s: %r", conn.username, line)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fake_cgos_server", description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6867)
    p.add_argument("--boardsize", type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--komi", default=DEFAULT_KOMI)
    p.add_argument("--level-ms", type=int, default=DEFAULT_LEVEL_MS)
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    p.add_argument(
        "--account", action="append", default=[], metavar="NAME:PASSWORD",
        help="restrict logins to these name:password pairs; may be repeated. Default: accept any.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    accounts = {}
    for spec in args.account:
        name, _, password = spec.partition(":")
        accounts[name] = password

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = FakeCGOSServer(
        host=args.host, port=args.port, boardsize=args.boardsize, komi=args.komi,
        level_ms=args.level_ms, max_moves=args.max_moves, accounts=accounts,
    )
    server.start()
    print(f"fake CGOS server listening on {server.host}:{server.port} (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
