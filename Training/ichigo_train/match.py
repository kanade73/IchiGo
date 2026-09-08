"""Match runner: GTP engine-vs-engine (or engine-vs-the-built-in uniform-legal baseline) games
(docs/spec/04-tasks.md T31; docs/spec/05-validation.md SS6 "機能完走と棋力評価" M1, SS5 paired
comparison design at a smaller scale).

CLI: ``python -m ichigo_train match --engine-a SPEC --engine-b SPEC|uniform --games N --size 9
--komi 7 --openings FILE|none --out DIR --seed S [--visits-a N --visits-b N |
--time-main-seconds T] [--max-moves N]`` (see ``ichigo_train.__main__``'s ``match`` subparser for
every flag).

Architecture note on legality: this runner does not reimplement Go rules. Every move's legality
is decided by a real GTP engine's ``play`` response (accept/reject), never by a Python-side board:

  * engine-vs-engine games: each engine tracks its own board. After the mover's ``genmove`` the
    move is echoed to the *other* engine via ``play``; a rejection there is recorded as an
    "illegal_move" incident and ends the game (result "illegal_move", not counted -- see
    ``_request_move``).
  * engine-vs-uniform-baseline games: the baseline (``UniformBaselinePlayer``) has no board of its
    own. Its "legality check" *is* the one real engine's ``play`` response: it proposes a
    uniformly random candidate point and calls ``play <color> <point>`` on that engine; acceptance
    is simultaneously the legality proof and the actual application of the move
    (docs/spec/05-validation.md SS6: "baselineは合法点から一様..."); a rejection is retried with
    another candidate. See ``UniformBaselinePlayer.choose_move``.

SGF/JSONL result convention: two consecutive passes end a game; its ``result`` is the *black*
engine's ``final_score`` (GTP area-scoring string, e.g. "B+3.5", "W+0.5", "0"). The white engine's
``final_score`` is also recorded and compared (``scoreDisagreement``) whenever both sides are real
engines. Hitting ``--max-moves`` without two passes ends the game as "truncated"; a rejected
opening/echoed move ends it as "illegal_move"; a per-command timeout ends it as "timeout"; an
unreadable/crashed engine ends it as "crash". None of these four count toward win/draw/loss
(docs/spec/04-tasks.md T31, mirroring SelfPlay.swift's "truncated" convention). A game ending this
way does not stop the match: the offending engine process is killed and a fresh instance (same
argv) is spawned before the next game (see ``MatchRunner._kill_and_note``).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass

from .stats import build_report

GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # skips I, matches Sources/IchiGoFeatures/Coordinates.swift
NON_COUNTED_RESULTS = ("truncated", "illegal_move", "timeout", "crash")


# ------------------------------------------------------------------------------------------------
# Coordinates (GTP <-> index <-> SGF), same conventions as Sources/IchiGoFeatures/Coordinates.swift
# and Training/ichigo_train/teacher.py's gtp_to_index: x=0 left, y=0 top, GTP row = size - y,
# letter I is skipped; SGF point is chr(97+x)+chr(97+y), empty string for pass.
# ------------------------------------------------------------------------------------------------


def index_to_gtp(index: int, size: int) -> str:
    x, y = index % size, index // size
    return f"{GTP_LETTERS[x]}{size - y}"


def gtp_to_xy(vertex: str, size: int) -> tuple[int, int] | None:
    v = vertex.strip().upper()
    if v == "PASS":
        return None
    x = GTP_LETTERS.index(v[0])
    row = int(v[1:])
    return x, size - row


def sgf_point(vertex: str, size: int) -> str:
    xy = gtp_to_xy(vertex, size)
    if xy is None:
        return ""
    x, y = xy
    return chr(97 + x) + chr(97 + y)


# ------------------------------------------------------------------------------------------------
# GTP client
# ------------------------------------------------------------------------------------------------


class GTPError(RuntimeError):
    """The engine answered a command with a `?...` (protocol failure) response."""


class GTPTimeout(RuntimeError):
    """No complete response arrived within the per-command timeout."""


class GTPClient:
    """One GTP engine subprocess, spawned from an explicit argv list (never a shell string split
    on whitespace: use ``parse_engine_spec`` to turn a CLI string into an argv list first, so a
    quoted path with spaces round-trips correctly). Commands are sent with an auto-incrementing
    numeric id; responses are read on a background thread so a per-command timeout can fire
    without blocking forever on a hung engine (mirrors ``ichigo_train.teacher.TeacherProcess``)."""

    def __init__(self, argv: list[str], name: str = "engine", timeout: float = 30.0, stderr=None, env=None):
        if not isinstance(argv, (list, tuple)) or not argv or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"argv must be a non-empty list of strings, got {argv!r}")
        self.argv = list(argv)
        self.name = name
        self.timeout = timeout
        self._id = 0
        self.closed = False
        self.proc = subprocess.Popen(
            self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr if stderr is not None else subprocess.DEVNULL, text=True, bufsize=1, env=env,
        )
        self._lines: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)  # EOF sentinel

    def _read_response(self, timeout: float) -> str:
        """Reads lines until the blank line that terminates one GTP response; returns the text
        without that trailing blank line. Raises GTPTimeout / EOFError."""
        deadline = time.monotonic() + timeout
        chunks: list[str] = []
        started = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GTPTimeout(f"{self.name}: no response within {timeout}s")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise GTPTimeout(f"{self.name}: no response within {timeout}s")
            if line is None:
                raise EOFError(f"{self.name}: engine process closed stdout")
            line = line.rstrip("\n").rstrip("\r")
            if line == "":
                if started:
                    return "\n".join(chunks)
                continue  # ignore blank lines before the response starts
            started = True
            chunks.append(line)

    def send(self, command: str, timeout: float | None = None) -> str:
        """Sends one GTP command with a fresh numeric id and returns the response payload (the
        text after the `=id`/`?id` marker). Raises GTPError for a `?` response."""
        if self.closed:
            raise GTPError(f"{self.name}: engine already closed")
        self._id += 1
        cid = self._id
        try:
            self.proc.stdin.write(f"{cid} {command}\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise GTPError(f"{self.name}: cannot write to engine: {e}") from e
        resp = self._read_response(timeout if timeout is not None else self.timeout)
        marker, _, rest = resp.partition(" ")
        ok = marker.startswith("=")
        got_id = marker[1:]
        if got_id and got_id != str(cid):
            rest = f"[id mismatch: expected {cid} got {got_id}] {rest}"
        if not ok:
            raise GTPError(f"{self.name}: {command!r} failed: {rest.strip()}")
        return rest

    def boardsize(self, n: int) -> None:
        self.send(f"boardsize {n}")

    def clear_board(self) -> None:
        self.send("clear_board")

    def komi(self, k: float) -> None:
        self.send(f"komi {k}")

    def play(self, color: str, vertex: str, timeout: float | None = None) -> bool:
        """True if accepted; False if the engine rejected the move as illegal (any other failure
        -- a crash, a timeout -- is re-raised, since the caller cannot tell "illegal" from "broken"
        without distinguishing GTPError/GTPTimeout/EOFError itself)."""
        try:
            self.send(f"play {color} {vertex}", timeout=timeout)
            return True
        except GTPError:
            return False

    def genmove(self, color: str, timeout: float | None = None) -> str:
        return self.send(f"genmove {color}", timeout=timeout).strip()

    def time_settings(self, main_seconds: float, byo_seconds: float = 0, byo_stones: int = 0) -> None:
        self.send(f"time_settings {main_seconds} {byo_seconds} {byo_stones}")

    def time_left(self, color: str, seconds: float, stones: int = 0) -> None:
        self.send(f"time_left {color} {seconds} {stones}")

    def final_score(self, timeout: float | None = None) -> str:
        return self.send("final_score", timeout=timeout).strip()

    def close(self, timeout: float = 5.0) -> None:
        """Best-effort clean shutdown: `quit`, then wait; falls back to ``kill`` on any trouble.
        Always returns (never raises)."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.proc.poll() is None:
                self.send("quit", timeout=timeout)
        except Exception:
            pass
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._force_kill()

    def kill(self) -> None:
        """Immediate, unconditional teardown (used after a timeout/crash: the engine is assumed
        wedged, so no `quit` round-trip is attempted)."""
        self.closed = True
        self._force_kill()

    def _force_kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def __enter__(self) -> "GTPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def parse_engine_spec(spec: str) -> tuple[str, list[str] | None]:
    """Turns one ``--engine-a``/``--engine-b`` CLI value into ``("uniform", None)`` or
    ``("gtp", argv)``. Never a naive ``str.split()`` (which mangles a quoted path containing a
    space): a value starting with `[` parses as a JSON array of strings (unambiguous, no quoting
    rules to get wrong); anything else is tokenised with `shlex.split` (shell-style quoting, same
    as typing the command at a shell prompt)."""
    s = spec.strip()
    if s.lower() == "uniform":
        return "uniform", None
    if s.startswith("["):
        argv = json.loads(s)
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError(f"--engine argv JSON must be a list of strings: {spec!r}")
        return "gtp", argv
    argv = shlex.split(s)
    if not argv:
        raise ValueError(f"empty engine command: {spec!r}")
    return "gtp", argv


# ------------------------------------------------------------------------------------------------
# Uniform-legal baseline (docs/spec/05-validation.md SS6)
# ------------------------------------------------------------------------------------------------


class UniformBaselinePlayer:
    """"baselineは合法点から一様、点がある間pass確率1%、なければpass、seed固定" -- extremely weak
    on purpose (docs/spec/05-validation.md SS6 explicitly warns not to call this "tournament
    strength"). Has no Go rules of its own; see the module docstring for how legality is decided.

    ``choose_move``:
      1. w.p. 1% returns "pass" outright (this "1% while points remain" IS the unconditional 1%
         draw: when no legal point remains the retry loop below falls through to "pass" too, so
         a forced pass and a voluntary 1% pass are indistinguishable in the returned value, which
         is all the spec requires).
      2. otherwise draws a uniformly random permutation of every board point from the seeded RNG
         and calls ``oracle.play(color, point)`` in that order, returning the first accepted
         point. Sampling a random order without replacement and stopping at the first success is
         exactly rejection sampling: the result is uniform over the *legal* points, not merely
         the empty ones, with no separate legality check required.
      3. if every point is rejected, passes.

    ``oracle`` must be the one real GTP engine in the match (the caller passes it in; see
    ``MatchRunner._oracle_for_baseline``). A "pass" return is NOT applied to the oracle by this
    method (there is nothing to legality-check about a pass) -- the caller must still send
    `play <color> pass` itself so the oracle's board/history stays in sync.
    """

    def __init__(self, size: int, rng: random.Random):
        self.size = size
        self.rng = rng

    def choose_move(self, oracle: GTPClient, color: str) -> str:
        if self.rng.random() < 0.01:
            return "pass"
        order = list(range(self.size * self.size))
        self.rng.shuffle(order)
        for idx in order:
            vertex = index_to_gtp(idx, self.size)
            if oracle.play(color, vertex):
                return vertex
        return "pass"


# ------------------------------------------------------------------------------------------------
# Openings
# ------------------------------------------------------------------------------------------------


def load_openings(path: str | None) -> list[dict]:
    """``path`` is JSONL, one opening per line: ``{"id": "...", "moves": ["D4", "Q16", ...]}``
    with GTP vertices alternating colour starting Black. ``None``/"none" means the empty board (a
    single synthetic opening with no moves), reused for every pair."""
    if path is None or path == "none":
        return [{"id": "none", "moves": []}]
    openings = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            openings.append({"id": row.get("id", f"opening-{i}"), "moves": list(row["moves"])})
    if not openings:
        raise ValueError(f"openings file {path} has no rows")
    return openings


def openings_file_sha256(path: str | None) -> str:
    if path is None or path == "none":
        return "none"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------
# Match runner
# ------------------------------------------------------------------------------------------------


class _GameAbort(Exception):
    def __init__(self, result: str, incident: dict):
        super().__init__(result)
        self.result = result
        self.incident = incident


@dataclass
class _ScoreOutcome:
    result: str
    counted: bool
    black_final_score: str | None
    white_final_score: str | None
    score_disagreement: bool | None


class MatchRunner:
    """Runs a full match: ``games`` games, colour-swapped in pairs (game ``2k`` has engine "A"
    black, game ``2k+1`` has engine "B" black, both on the same opening -- docs/spec/04-tasks.md
    T31 "対応開局の色交換"), writes ``<out>/games.jsonl`` (one row per game), ``<out>/games/*.sgf``
    and ``<out>/report.json`` (via ``ichigo_train.stats.build_report``, candidate = engine "A").

    ``engine_a``/``engine_b`` are ``("gtp", argv)`` or ``("uniform", None)`` (see
    ``parse_engine_spec``); at least one side must be a real engine (the baseline needs one as its
    legality oracle). Real engines are spawned once and reused across games (``clear_board`` /
    ``komi`` between games) for speed; an engine that times out or crashes is killed and respawned
    fresh before the next game that needs it (that game's own result is not counted; see the
    module docstring).
    """

    def __init__(
        self, *, engine_a: tuple[str, list[str] | None], engine_b: tuple[str, list[str] | None],
        games: int, out_dir: str, seed: int, size: int = 9, komi: float = 7.0,
        openings_path: str | None = None, visits_a: int | None = None, visits_b: int | None = None,
        time_main_seconds: float | None = None, max_moves: int | None = None,
        command_timeout: float = 30.0, genmove_timeout: float | None = None,
        label_a: str = "A", label_b: str | None = None, resamples: int = 10000, log=lambda msg: None,
    ):
        if engine_a[0] == "uniform" and engine_b[0] == "uniform":
            raise ValueError("both engines cannot be 'uniform': the baseline needs a real GTP engine as its legality oracle")
        self.engine_specs = {"A": engine_a, "B": engine_b}
        self.games_n = games
        self.size = size
        self.komi = komi
        self.openings = load_openings(openings_path)
        self.openings_hash = openings_file_sha256(openings_path)
        self.visits = {"A": visits_a, "B": visits_b}
        self.time_main_seconds = time_main_seconds
        self.max_moves = max_moves if max_moves is not None else 4 * size * size
        self.command_timeout = command_timeout
        self.genmove_timeout = genmove_timeout if genmove_timeout is not None else max(120.0, 4.0 * (time_main_seconds or 0))
        self.seed = seed
        self.labels = {"A": label_a, "B": label_b if label_b is not None else ("uniform" if engine_b[0] == "uniform" else "B")}
        self.resamples = resamples
        self.out_dir = out_dir
        self.log = log
        self.clients: dict[str, GTPClient | None] = {"A": None, "B": None}
        self.baseline_players: dict[str, UniformBaselinePlayer] = {}
        self._log_paths: dict[str, str] = {}
        self._log_files: dict[str, object] = {}
        self._model_hash_cache: dict[str, str | None] = {"A": None, "B": None}
        self._time_remaining: dict[str, float] | None = None
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "games"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # -- engine lifecycle -------------------------------------------------------------------

    def _spawn(self, key: str) -> None:
        kind, argv = self.engine_specs[key]
        if kind != "gtp":
            return
        argv = list(argv)
        v = self.visits[key]
        if v is not None:
            if "--visits" in argv:
                raise ValueError(f"engine {key}: --visits given both in its argv and via --visits-{key.lower()}")
            argv = argv + ["--visits", str(v)]
        old_log = self._log_files.pop(key, None)
        if old_log is not None:
            try:
                old_log.close()
            except Exception:
                pass
        log_path = os.path.join(self.out_dir, "logs", f"engine-{key.lower()}.stderr.log")
        log_file = open(log_path, "a")
        client = GTPClient(argv, name=self.labels[key], timeout=self.command_timeout, stderr=log_file)
        client.boardsize(self.size)
        self.clients[key] = client
        self._log_paths[key] = log_path
        self._log_files[key] = log_file

    def _ensure_spawned(self) -> None:
        for key in ("A", "B"):
            if self.engine_specs[key][0] == "gtp" and self.clients[key] is None:
                self._spawn(key)

    def _kill_and_note(self, key: str) -> None:
        client = self.clients.get(key)
        if client is not None:
            client.kill()
        self.clients[key] = None

    def _baseline_for(self, key: str) -> UniformBaselinePlayer:
        if key not in self.baseline_players:
            # Independent RNG stream per baseline side, both derived from the match seed so the
            # whole match is reproducible end to end.
            seed = self.seed if key == "A" else self.seed + 1_000_003
            self.baseline_players[key] = UniformBaselinePlayer(size=self.size, rng=random.Random(seed))
        return self.baseline_players[key]

    def _model_hash(self, key: str) -> str | None:
        if self._model_hash_cache.get(key):
            return self._model_hash_cache[key]
        log_path = self._log_paths.get(key)
        if not log_path or not os.path.exists(log_path):
            return None
        try:
            text = open(log_path, errors="ignore").read()
        except OSError:
            return None
        m = re.search(r"model[=: ]([0-9a-fA-F]{6,64})", text)
        h = m.group(1) if m else None
        if h:
            self._model_hash_cache[key] = h
        return h

    # -- one game -----------------------------------------------------------------------------

    def _apply_opening_move(self, black_key: str, white_key: str, colour: str, vertex: str) -> bool:
        ok = True
        for key in (black_key, white_key):
            if self.engine_specs[key][0] == "gtp" and not self.clients[key].play(colour, vertex):
                ok = False
        return ok

    def _request_move(self, mover_key: str, other_key: str, colour: str) -> tuple[str, float]:
        """Returns ``(move, elapsed_seconds)``. Raises ``_GameAbort`` on a fatal problem."""
        t0 = time.monotonic()
        if self.engine_specs[mover_key][0] == "uniform":
            oracle = self.clients[other_key]
            if oracle is None:
                raise _GameAbort("crash", {"type": "crash", "engine": self.labels[other_key], "detail": "no oracle engine available"})
            baseline = self._baseline_for(mover_key)
            try:
                move = baseline.choose_move(oracle, colour)
                if move == "pass" and not oracle.play(colour, "pass"):
                    raise _GameAbort("illegal_move", {"type": "illegal_move", "engine": self.labels[other_key], "detail": "oracle rejected pass"})
            except GTPTimeout as e:
                self._kill_and_note(other_key)
                raise _GameAbort("timeout", {"type": "timeout", "engine": self.labels[other_key], "detail": str(e)}) from e
            except (GTPError, EOFError) as e:
                self._kill_and_note(other_key)
                raise _GameAbort("crash", {"type": "crash", "engine": self.labels[other_key], "detail": str(e)}) from e
            return move, time.monotonic() - t0

        mover = self.clients[mover_key]
        try:
            if self._time_remaining is not None:
                mover.time_left(colour, max(0.0, self._time_remaining[mover_key]), 0)
            move = mover.genmove(colour, timeout=self.genmove_timeout)
        except GTPTimeout as e:
            self._kill_and_note(mover_key)
            raise _GameAbort("timeout", {"type": "timeout", "engine": self.labels[mover_key], "detail": str(e)}) from e
        except (GTPError, EOFError) as e:
            self._kill_and_note(mover_key)
            raise _GameAbort("crash", {"type": "crash", "engine": self.labels[mover_key], "detail": str(e)}) from e
        elapsed = time.monotonic() - t0
        if self._time_remaining is not None:
            self._time_remaining[mover_key] = max(0.0, self._time_remaining[mover_key] - elapsed)

        if self.engine_specs[other_key][0] == "gtp":
            other = self.clients[other_key]
            try:
                accepted = other.play(colour, move, timeout=self.command_timeout)
            except GTPTimeout as e:
                self._kill_and_note(other_key)
                raise _GameAbort("timeout", {"type": "timeout", "engine": self.labels[other_key], "detail": str(e)}) from e
            except EOFError as e:
                self._kill_and_note(other_key)
                raise _GameAbort("crash", {"type": "crash", "engine": self.labels[other_key], "detail": str(e)}) from e
            if not accepted:
                self._kill_and_note(other_key)
                raise _GameAbort("illegal_move", {"type": "illegal_move", "engine": self.labels[mover_key], "detail": f"{move} rejected by {self.labels[other_key]}"})
        return move, elapsed

    def _score_game(self, black_key: str, white_key: str) -> _ScoreOutcome:
        scores: dict[str, str | None] = {}
        for key in (black_key, white_key):
            if self.engine_specs[key][0] != "gtp":
                continue
            try:
                scores[key] = self.clients[key].final_score(timeout=self.command_timeout)
            except (GTPError, GTPTimeout, EOFError):
                scores[key] = None
        black_score, white_score = scores.get(black_key), scores.get(white_key)
        disagreement = None if black_score is None or white_score is None else black_score != white_score
        record = black_score if black_score is not None else white_score
        return _ScoreOutcome(record if record is not None else "crash", record is not None, black_score, white_score, disagreement)

    def _write_sgf(self, game_id: str, opening: dict, moves: list[str], move_colours: list[str], black_key: str, white_key: str, result: str) -> str:
        parts = []
        for i, mv in enumerate(opening["moves"]):
            parts.append(f";{'B' if i % 2 == 0 else 'W'}[{sgf_point(mv, self.size)}]")
        for colour, mv in zip(move_colours, moves):
            parts.append(f";{colour}[{sgf_point(mv, self.size)}]")
        re_tag = f"RE[{result}]" if result not in NON_COUNTED_RESULTS else "RE[Void]"
        a_desc = " ".join(self.engine_specs["A"][1]) if self.engine_specs["A"][0] == "gtp" else "uniform"
        b_desc = " ".join(self.engine_specs["B"][1]) if self.engine_specs["B"][0] == "gtp" else "uniform"
        comment = f"match seed={self.seed} game={game_id} opening={opening['id']} A=[{a_desc}] B=[{b_desc}]"
        sgf = (
            f"(;GM[1]FF[4]SZ[{self.size}]KM[{self.komi}]RU[Chinese]"
            f"PB[{self.labels[black_key]}]PW[{self.labels[white_key]}]{re_tag}C[{comment}]" + "".join(parts) + ")\n"
        )
        rel_path = os.path.join("games", f"{game_id}.sgf")
        with open(os.path.join(self.out_dir, rel_path), "w") as f:
            f.write(sgf)
        return rel_path

    def _play_one_game(self, game_index: int) -> dict:
        self._ensure_spawned()
        pair_index = game_index // 2
        swap = (game_index % 2) == 1
        black_key, white_key = ("B", "A") if swap else ("A", "B")
        opening = self.openings[pair_index % len(self.openings)]
        game_id = f"game-{game_index:05d}"
        incidents: list[dict] = []
        self._time_remaining = {"A": self.time_main_seconds, "B": self.time_main_seconds} if self.time_main_seconds else None

        def record(result, counted, moves, move_colours, move_times, scoring: _ScoreOutcome | None = None):
            return {
                "gameId": game_id, "pairIndex": pair_index, "openingId": opening["id"],
                "colours": {"black": self.labels[black_key], "white": self.labels[white_key]},
                "result": result, "countedResult": counted,
                "openingMoves": opening["moves"], "moves": moves, "moveColours": move_colours,
                "moveTimesSeconds": move_times,
                "blackFinalScore": scoring.black_final_score if scoring else None,
                "whiteFinalScore": scoring.white_final_score if scoring else None,
                "scoreDisagreement": scoring.score_disagreement if scoring else None,
                "incidents": incidents,
                "modelHashes": {"A": self._model_hash("A"), "B": self._model_hash("B")},
                "sgfPath": self._write_sgf(game_id, opening, moves, move_colours, black_key, white_key, result),
            }

        try:
            for key in (black_key, white_key):
                if self.engine_specs[key][0] == "gtp":
                    client = self.clients[key]
                    client.clear_board()
                    client.komi(self.komi)
                    if self.time_main_seconds:
                        client.time_settings(self.time_main_seconds, 0, 0)
        except (GTPError, GTPTimeout, EOFError) as e:
            incidents.append({"type": "crash", "engine": "setup", "detail": str(e)})
            return record("crash", False, [], [], [])

        colour_seq = []
        for i, mv in enumerate(opening["moves"]):
            colour = "B" if i % 2 == 0 else "W"
            colour_seq.append(colour)
            if not self._apply_opening_move(black_key, white_key, colour, mv):
                incidents.append({"type": "illegal_move", "engine": "opening", "detail": f"opening move {i} ({mv}) rejected"})
                return record("illegal_move", False, [], [], [])

        turn_is_black = len(opening["moves"]) % 2 == 0
        consec_passes = 0
        moves: list[str] = []
        move_colours: list[str] = []
        move_times: list[float] = []
        n = 0
        result, counted, scoring = None, False, None
        while True:
            if n >= self.max_moves:
                result, counted = "truncated", False
                break
            mover_key = black_key if turn_is_black else white_key
            other_key = white_key if turn_is_black else black_key
            colour = "B" if turn_is_black else "W"
            try:
                move, elapsed = self._request_move(mover_key, other_key, colour)
            except _GameAbort as abort:
                incidents.append(abort.incident)
                result, counted = abort.result, False
                break
            moves.append(move)
            move_colours.append(colour)
            move_times.append(elapsed)
            consec_passes = consec_passes + 1 if move == "pass" else 0
            n += 1
            turn_is_black = not turn_is_black
            if consec_passes >= 2:
                scoring = self._score_game(black_key, white_key)
                result, counted = scoring.result, scoring.counted
                break

        return record(result, counted, moves, move_colours, move_times, scoring)

    # -- whole match ----------------------------------------------------------------------------

    def run(self) -> dict:
        games_path = os.path.join(self.out_dir, "games.jsonl")
        games = []
        with open(games_path, "w") as gf:
            for g in range(self.games_n):
                rec = self._play_one_game(g)
                games.append(rec)
                gf.write(json.dumps(rec, sort_keys=True) + "\n")
                gf.flush()
                self.log(f"[match] game {g}: result={rec['result']} counted={rec['countedResult']} moves={len(rec['moves'])}")
        for key in ("A", "B"):
            client = self.clients[key]
            if client is not None:
                client.close()
        for log_file in self._log_files.values():
            try:
                log_file.close()
            except Exception:
                pass

        engines_report = {
            key: {"kind": self.engine_specs[key][0], "argv": self.engine_specs[key][1], "label": self.labels[key], "visits": self.visits[key]}
            for key in ("A", "B")
        }
        settings = {
            "size": self.size, "komi": self.komi, "maxMoves": self.max_moves,
            "timeMainSeconds": self.time_main_seconds, "commandTimeoutSeconds": self.command_timeout,
            "genmoveTimeoutSeconds": self.genmove_timeout,
        }
        report = build_report(
            games, candidate="A", seed=self.seed, engines=engines_report, settings=settings,
            openings_file_hash=self.openings_hash, resamples=self.resamples,
        )
        with open(os.path.join(self.out_dir, "report.json"), "w") as rf:
            json.dump(report, rf, indent=2, sort_keys=True)
        return report
