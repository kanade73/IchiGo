#!/usr/bin/env python3
"""IchiGo CGOS client (docs/spec/03-engine.md §9, docs/spec/04-tasks.md T30).

Speaks the public CGOS wire protocol implemented by https://github.com/zakki/cgos, pinned to
revision ``4dcff8754400fa43a71323ffc3d2c5eb4a19f7f1`` (fetched via the GitHub REST/raw API on
2026-09-09; the repository is a "Computer Go Server mirror"). This module is a clean-room
reimplementation targeting the ``ichigo gtp`` engine directly (Python 3.12 stdlib only, no
vendored/copied source) -- see "Protocol grammar" below for the exact wire format this was
checked against, and docs/provenance/cgos-import.json for the reference files read (path +
sha256) while writing it.

Two files at that revision were read to recover the wire protocol, cross-checked against each
other (client parser vs. server formatter must agree on the grammar):

  - ``client-python/src/cgosclient.py`` and ``client-python/src/gtpengine.py``
    (GPL-3, Copyright (C) 2009 Christian Nentwich and contributors) -- the reference *client*:
    how it dispatches server commands, negotiates ``genmove_analyze``, and turns a
    ``kata-genmove_analyze`` GTP response into the wire-format analysis string.
  - ``server-python/cgos/app/cgos.py`` and ``server-python/cgos/app/config.py``
    (MIT, Copyright (C) 2009 Don Dailey and Jason House, (c) 2022 Kensuke Matsuzaki) -- the
    reference *server*: exact ``setup``/``genmove``/``play``/``gameover`` message formatting,
    confirming field order and units against the client's parser instead of trusting either
    side alone.

  Also present locally (not authoritative, just corroborating): a byte-for-byte prior audit of
  the same v1.1.0-era client under ``~/dev/univ/koubou/katago-mlx/.tools/cgos-client`` (that
  in-house wrapper, ``~/dev/univ/koubou/katago-mlx/Scripts/cgos/safe_cgos_client.py``, and its
  tests, informed the reconnect/backoff/shutdown *shape* below -- state-dir move ledger so a
  ``setup`` replay after reconnect cannot double-apply a move, SIGTERM/SIGINT setting a
  stop-after-this-game flag instead of tearing the connection down mid-game, and
  ``RotatingFileHandler`` log rotation -- but no lines were copied; this module talks to
  ``ichigo gtp`` directly instead of spawning the upstream client as a subprocess).

Protocol grammar (server -> client unless noted; verified against both the pinned client's
parser and the pinned server's formatter, not assumed from either alone):

  ``protocol [genmove_analyze]``
      Sent immediately on connect, before the client says anything. Reply with
      ``<CLIENT_ID>[ genmove_analyze]`` -- the suffix only if the server offered it *and* this
      client's config has analysis enabled.
  ``username`` / ``password``
      Reply with the configured username / the contents of ``password_file``.
  ``setup <gid> <boardsize> <komi> <time_ms> <white>(<rating>) <black>(<rating>) [<move> <time_ms>]*``
      One line, whitespace-separated. ``<time_ms>`` is main time per player in milliseconds
      (``cfg.level`` on the reference server, sent to the engine as
      ``time_settings <time_ms//1000> 0 0`` -- sudden death, matching docs/spec/03-engine.md §8).
      The name field order is WHITE then BLACK (confirmed from the server's own ``wp``/``bp``
      variable names in the f-string that builds this line); a trailing move/time list means a
      resume (either a genuinely new reconnect, or catching up a game that was already partly
      played when we logged in). Every ``setup`` -- new game or resume -- gets the same
      treatment: ``boardsize`` -> ``komi`` -> ``clear_board`` -> ``time_settings`` -> replay each
      move via ``play`` -> ``time_left`` for both colours restored from the last time value seen
      per colour in the replay (full main time if a colour made no moves yet). Because the
      engine's own board is always rebuilt from ``clear_board`` plus this authoritative replay,
      a move is never double-applied to the engine regardless of what happened before the
      reconnect; the per-game ledger under ``state_dir/games/<gid>.json`` is rewritten from
      scratch on every ``setup`` for the same reason (it mirrors the engine's state, not an
      independently-trusted diff).
  ``play <colour> <coord> <time_left_ms>``
      Opponent's move. Forwarded to the engine as ``play <colour> <coord>`` (coordinate
      lower-cased). No time notification here -- CGOS only sends time information right before
      it asks *us* to move.
  ``genmove <colour> <time_left_ms>``
      Our move is requested. We send ``time_left <colour> <sec> 0`` to the engine first, then
      ``kata-genmove_analyze <colour>`` if analysis is negotiated on, else plain
      ``genmove <colour>``. Reply to the server with ``<move>[ <analysis-json>]`` -- the analysis
      suffix is present only when the engine actually returned an ``info`` line (it may not, if
      genmove's own deadline watchdog fired and returned a fallback move with no search result;
      docs/spec/03-engine.md §8/§9).
  ``gameover <date> <result> [<detail>]``
      Game over; ``<result>`` is an SGF-style ``RE`` value (``B+2.5``, ``W+Resign``, ``Draw``,
      ...). Reply ``ready`` to be queued for another game, unless a stop was requested (SIGTERM/
      SIGINT, or the configured game count was reached) in which case reply ``quit`` and end the
      connection -- a stop is only ever acted on here, between games, never mid-game.
  ``info <text>``
      Informational; logged and otherwise ignored.

Analysis wire format (the "CGOS analysis/comment extension"): a single-space-separated
``<move> <json>`` reply to ``genmove``, where ``<json>`` is produced by parsing our engine's
``kata-genmove_analyze`` ``info`` line exactly the way the pinned reference client's
``AnalyzeResultParser`` does (see ``parse_kata_info_line`` below, and its docstring for the
token-by-token replication, including two easy-to-miss quirks reproduced on purpose: the
``pv`` field drops the *first* PV token -- and is omitted entirely if that was the only one --
because it duplicates the ``move`` field, and unrecognised tokens such as ``order`` are silently
consumed and dropped, never surfaced). The pinned server round-trips this JSON through
``json.loads``/``json.dumps(..., separators=(",", ":"))`` and discards it unless it parses as a
JSON object, so this client produces exactly that: compact (no spaces) JSON with keys in
insertion order ``moves, visits, winrate, score`` at the top level and, per candidate,
``move, visits, winrate, score, prior, pv``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

CLIENT_ID = "e1 IchiGoCGOS 0.1.0"
CONNECT_TIMEOUT_SECONDS = 30.0

# 03-engine.md §9: "backoffは1/2/4/8/16/30秒上限".
BACKOFF_SEQUENCE_SECONDS: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)

# 03-engine.md §9: "ログrotateは20MiB×5".
LOG_ROTATE_MAX_BYTES = 20 * 1024 * 1024
LOG_ROTATE_BACKUP_COUNT = 5


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised for a structurally invalid config JSON."""


class CGOSProtocolError(Exception):
    """Raised for anything unexpected from the CGOS server (bad line, bad state, EOF)."""


class GTPProtocolError(Exception):
    """Raised for anything unexpected from the local ``ichigo gtp`` subprocess."""


class _StopRequested(Exception):
    """Internal control-flow signal: stop between games, raised only after a gameover."""


# --------------------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------------------


class Backoff:
    """1/2/4/8/16/30s reconnect backoff (docs/spec/03-engine.md §9), capped at the last value."""

    def __init__(self, sequence: Sequence[float] = BACKOFF_SEQUENCE_SECONDS) -> None:
        self._sequence = tuple(sequence)
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def next_delay(self) -> float:
        delay = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return delay


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ClientConfig:
    host: str
    port: int
    username: str
    password_file: Path
    board_size: int
    engine_argv: Tuple[str, ...]
    analysis_enabled: bool
    state_dir: Path
    log_dir: Path
    max_games: Optional[int] = None
    client_id: str = CLIENT_ID


_REQUIRED_STRING_KEYS = ("host", "username", "password_file", "state_dir", "log_dir")


def _is_absolute_on_any_platform(raw: str) -> bool:
    # A password path must be relative regardless of the OS the config was authored on, so a
    # config JSON committed on one platform cannot silently carry an absolute path on another.
    return PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute()


def _validate_and_build(raw: Any, base_dir: Path) -> ClientConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")

    for key in _REQUIRED_STRING_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"config.{key} must be a non-empty string")

    port = raw.get("port")
    if not isinstance(port, int) or isinstance(port, bool):
        raise ConfigError("config.port must be an integer")

    board_size = raw.get("board_size")
    if board_size not in (9, 19):
        raise ConfigError("config.board_size must be 9 or 19")

    argv = raw.get("engine_argv")
    if not isinstance(argv, list) or not argv:
        raise ConfigError("config.engine_argv must be a non-empty JSON array (argv must be a list)")
    if not all(isinstance(a, str) and a for a in argv):
        raise ConfigError("config.engine_argv entries must all be non-empty strings")

    analysis = raw.get("analysis")
    if not isinstance(analysis, bool):
        raise ConfigError("config.analysis must be a boolean")

    password_raw = raw["password_file"]
    if _is_absolute_on_any_platform(password_raw):
        raise ConfigError(
            "config.password_file must be a relative path -- an absolute path must never be "
            "committed inline in a config JSON (it would leak local filesystem layout and risks "
            "a stale/shared secret across machines); point it at a file next to the config or "
            "under state_dir instead"
        )
    password_file = (base_dir / password_raw).resolve()

    state_dir_raw = raw["state_dir"]
    state_dir = Path(state_dir_raw)
    if not state_dir.is_absolute():
        state_dir = (base_dir / state_dir_raw).resolve()

    log_dir_raw = raw["log_dir"]
    log_dir = Path(log_dir_raw)
    if not log_dir.is_absolute():
        log_dir = (base_dir / log_dir_raw).resolve()

    max_games = raw.get("max_games")
    if max_games is not None and (not isinstance(max_games, int) or isinstance(max_games, bool) or max_games < 1):
        raise ConfigError("config.max_games must be a positive integer if present")

    client_id = raw.get("client_id", CLIENT_ID)
    if not isinstance(client_id, str) or not client_id:
        raise ConfigError("config.client_id must be a non-empty string if present")

    return ClientConfig(
        host=raw["host"],
        port=port,
        username=raw["username"],
        password_file=password_file,
        board_size=board_size,
        engine_argv=tuple(argv),
        analysis_enabled=analysis,
        state_dir=state_dir,
        log_dir=log_dir,
        max_games=max_games,
        client_id=client_id,
    )


def load_config(path: Path) -> ClientConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"config {path} is not valid JSON: {e}") from e
    return _validate_and_build(raw, base_dir=path.resolve().parent)


def read_password(config: ClientConfig) -> str:
    try:
        text = config.password_file.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read password_file {config.password_file}: {e}") from e
    password = text.strip()
    if not password:
        raise ConfigError(f"password_file {config.password_file} is empty")
    return password


# --------------------------------------------------------------------------------------
# Hashing (model/binary hashes fixed at game start; docs/spec/03-engine.md §9)
# --------------------------------------------------------------------------------------


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_model_dir(model_dir: Path) -> str:
    """sha256 over the sorted (relative_path, file_sha256) manifest of a model directory."""
    h = hashlib.sha256()
    files = sorted(p for p in model_dir.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(model_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hash_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


_MODEL_FLAGS = ("--model-9", "--model-19", "--model")


def model_dirs_from_argv(argv: Sequence[str]) -> List[Path]:
    dirs = []
    for i, tok in enumerate(argv):
        if tok in _MODEL_FLAGS and i + 1 < len(argv):
            dirs.append(Path(argv[i + 1]))
    return dirs


def hash_engine_and_models(argv: Sequence[str]) -> Dict[str, str]:
    """Returns {"engine_binary": sha256, "model_manifest": sha256-of-sorted-manifests}."""
    engine_path = Path(argv[0])
    result = {"engine_binary": hash_file(engine_path)}
    model_dirs = model_dirs_from_argv(argv)
    if model_dirs:
        combined = hashlib.sha256()
        for d in model_dirs:
            combined.update(d.as_posix().encode("utf-8"))
            combined.update(b"\0")
            combined.update(hash_model_dir(d).encode("ascii"))
            combined.update(b"\n")
        result["model_manifest"] = combined.hexdigest()
    else:
        result["model_manifest"] = ""
    return result


# --------------------------------------------------------------------------------------
# kata-genmove_analyze parsing (replicates the pinned reference client's AnalyzeResultParser)
# --------------------------------------------------------------------------------------

# Tokens the reference parser recognises by name inside an "info" block; anything else is an
# unrecognised/custom attribute and is silently consumed-and-dropped (this engine never sends
# custom attributes, but "order" -- which our engine's info line always includes -- is one of
# these and is dropped on purpose, matching the reference).
_INFO_KEYWORDS = {"move", "winrate", "score", "scoreLead", "pv", "prior", "visits"}
_NUMBER_ATTRIBUTES = {
    "visits", "winrate", "prior", "lcb", "utility", "scoreMean", "scoreStdev", "scoreLead",
    "scoreSelfplay", "utilityLcb", "weight", "order", "pvEdgeVisits", "movesOwnership",
    "movesOwnershipStdev",
}


class _TokenCursor:
    __slots__ = ("tokens", "pos")

    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def has_next(self) -> bool:
        return self.pos < len(self.tokens)

    def next(self) -> str:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None


def _next_number(cur: _TokenCursor):
    t = cur.next()
    try:
        return int(t)
    except ValueError:
        return float(t)


def _parse_info_block(cur: _TokenCursor) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    while True:
        t = cur.peek()
        if t is None or t == "info" or t == "ownership":
            return m
        t = cur.next()

        if t not in _INFO_KEYWORDS:
            # Unrecognised attribute (e.g. "order"): consume its value and drop both, exactly
            # like the reference parser with sendsCustomAttributes == False (its only mode, for
            # both kata- and lz-genmove_analyze).
            if cur.has_next():
                cur.next()
            continue

        if t == "pv":
            pv: List[str] = []
            while True:
                pv.append(cur.next())
                nxt = cur.peek()
                if nxt is None or nxt == "info" or nxt == "ownership":
                    if len(pv) > 1:
                        # Drop the first PV token: it duplicates "move". If it was the only
                        # token, "pv" is omitted entirely (not an empty string) -- matches the
                        # reference exactly.
                        m["pv"] = " ".join(pv[1:])
                    return m
            # unreachable
        if t in ("winrate", "prior"):
            m[t] = _next_number(cur)
        elif t == "scoreLead":
            m["score"] = _next_number(cur)
        elif t in _NUMBER_ATTRIBUTES:
            m[t] = _next_number(cur)
        else:
            m[t] = cur.next()


def parse_kata_info_line(tokens: List[str]) -> Dict[str, Any]:
    """Parses a ``kata-genmove_analyze`` result (the "info ... [ownership ...]" tokens, with the
    leading GTP "=" marker already stripped) into the dict the reference client would send to
    the CGOS server as JSON. See the module docstring for the exact field/order guarantees.
    """
    cur = _TokenCursor(tokens)
    info: Dict[str, Any] = {"moves": []}
    total_visits = 0

    while cur.has_next():
        t = cur.next()
        if t == "info":
            m = _parse_info_block(cur)
            if "visits" in m:
                total_visits += m["visits"]
            info["moves"].append(m)
        elif t == "ownership":
            # Our engine never emits this (docs/spec/03-engine.md §9: ownership is optional and
            # not implemented), but consume defensively so a future ownership-capable engine
            # cannot desync this parser.
            while cur.has_next():
                cur.next()
        # else: an unknown top-level token is logged by the reference and otherwise ignored.

    if total_visits > 0:
        winrate = 0.0
        has_winrate = False
        score = 0.0
        has_score = False
        for m in info["moves"]:
            if "visits" not in m:
                continue
            if "winrate" in m:
                winrate += m["winrate"] * m["visits"] / total_visits
                has_winrate = True
            if "score" in m:
                score += m["score"] * m["visits"] / total_visits
                has_score = True
        info["visits"] = total_visits
        if has_winrate:
            info["winrate"] = winrate
        if has_score:
            info["score"] = score

    return info


def encode_analysis(info: Dict[str, Any]) -> str:
    return json.dumps(info, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# GTP engine subprocess
# --------------------------------------------------------------------------------------


class GTPEngineProcess:
    """Owns the ``ichigo gtp`` subprocess for the lifetime of the client process. Never
    restarted or replaced mid-game (or at all, once started) -- docs/spec/03-engine.md §9.
    """

    def __init__(self, argv: Sequence[str], log: logging.Logger) -> None:
        self.argv = list(argv)
        self.log = log
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.log.info("started engine subprocess argv=%r pid=%s", self.argv, self._proc.pid)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            self._send_raw("quit")
        except Exception:
            pass
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log.warning("engine did not exit after quit; terminating")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def _send_raw(self, command: str) -> List[str]:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        if self._proc.poll() is not None:
            raise GTPProtocolError(f"engine process has already exited (rc={self._proc.returncode})")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

        response: List[str] = []
        error: Optional[str] = None
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise GTPProtocolError(f"engine closed stdout while responding to {command!r}")
            if line.strip() == "":
                break
            if line[0] == "=":
                line = line[1:]
            elif line[0] == "?":
                error = line[1:].strip()
            line = line.strip()
            if line:
                response.append(line)

        if error is not None:
            raise GTPProtocolError(f"engine rejected {command!r}: {error}")
        return response

    def notify_boardsize(self, size: str) -> None:
        self._send_raw(f"boardsize {size}")

    def notify_komi(self, komi: str) -> None:
        self._send_raw(f"komi {komi}")

    def notify_clear_board(self) -> None:
        self._send_raw("clear_board")

    def notify_time_settings(self, total_ms: int) -> None:
        self._send_raw(f"time_settings {int(total_ms) // 1000} 0 0")

    def notify_time_left(self, color: str, remaining_ms: int) -> None:
        self._send_raw(f"time_left {color} {int(remaining_ms) // 1000} 0")

    def notify_play(self, color: str, coord: str) -> None:
        self._send_raw(f"play {color} {coord}")

    def request_genmove(self, color: str, use_analysis: bool) -> Tuple[str, Optional[str]]:
        cmd = f"kata-genmove_analyze {color}" if use_analysis else f"genmove {color}"
        result = self._send_raw(cmd)

        move: Optional[str] = None
        analysis_line: Optional[str] = None
        for line in result:
            if line.startswith("play "):
                move = line.split(" ")[-1]
                break
            analysis_line = line

        if move is None:
            raise GTPProtocolError(f"engine produced no move for {cmd!r}: {result!r}")

        analysis_json: Optional[str] = None
        if use_analysis and analysis_line is not None:
            info = parse_kata_info_line(analysis_line.split(" "))
            analysis_json = encode_analysis(info)

        return move.lower(), analysis_json


# --------------------------------------------------------------------------------------
# Move ledger (state_dir/games/<gid>.json) -- rebuilt from scratch on every "setup" so a replay
# can never double-apply a move regardless of what was recorded before a reconnect.
# --------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class MoveLedger:
    def __init__(self, path: Path, gid: str, boardsize: str, komi: str) -> None:
        self.path = path
        self.gid = gid
        self.boardsize = boardsize
        self.komi = komi
        self.moves: List[Dict[str, Any]] = []
        self.result: Optional[str] = None
        self._write()

    @classmethod
    def reset(cls, state_dir: Path, gid: str, boardsize: str, komi: str) -> "MoveLedger":
        path = state_dir / "games" / f"{gid}.json"
        return cls(path, gid, boardsize, komi)

    def append(self, color: str, coord: str, analysis: Optional[str], source: str) -> None:
        self.moves.append({"color": color, "coord": coord, "analysis": analysis, "source": source})
        self._write()

    def mark_complete(self, result: str) -> None:
        self.result = result
        self._write()

    def _write(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "gid": self.gid,
                "boardsize": self.boardsize,
                "komi": self.komi,
                "moves": self.moves,
                "result": self.result,
            },
        )


# --------------------------------------------------------------------------------------
# CGOS socket line I/O
# --------------------------------------------------------------------------------------


class _LineSocket:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._rfile = sock.makefile("r", encoding="utf-8", newline="\n")
        self._wfile = sock.makefile("w", encoding="utf-8", newline="\n")

    def readline(self) -> Optional[str]:
        line = self._rfile.readline()
        if line == "":
            return None
        return line.rstrip("\r\n")

    def send_line(self, text: str) -> None:
        self._wfile.write(text + "\n")
        self._wfile.flush()

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


def _split_name_rating(spec: str) -> Tuple[str, str]:
    if "(" in spec and spec.endswith(")"):
        name, _, rating = spec.partition("(")
        return name, rating[:-1]
    return spec, ""


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------


class CGOSClient:
    def __init__(
        self,
        config: ClientConfig,
        *,
        log: logging.Logger,
        stop_event: Optional[threading.Event] = None,
        clock: Any = time,
    ) -> None:
        self.config = config
        self.log = log
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        self._clock = clock
        self._password = read_password(config)
        self._engine = GTPEngineProcess(config.engine_argv, log=log.getChild("engine"))
        self._io: Optional[_LineSocket] = None
        self._use_analysis = False
        self._current_gid: Optional[str] = None
        self._current_color: Optional[str] = None
        self._ledger: Optional[MoveLedger] = None
        self.games_completed = 0
        self._engine_hashes: Dict[str, str] = {}

        self._handlers = {
            "protocol": self._on_protocol,
            "username": self._on_username,
            "password": self._on_password,
            "info": self._on_info,
            "setup": self._on_setup,
            "play": self._on_play,
            "genmove": self._on_genmove,
            "gameover": self._on_gameover,
        }

    def request_stop(self) -> None:
        self._stop_event.set()

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self.log.info("received signal %s: will stop after the current game", signum)
        self.request_stop()

    # -- main loop --------------------------------------------------------------------

    def run(self) -> int:
        self._engine.start()
        self._engine_hashes = hash_engine_and_models(self.config.engine_argv)
        self.log.info(
            "engine hashes fixed for this process: engine_binary=%s model_manifest=%s",
            self._engine_hashes["engine_binary"],
            self._engine_hashes["model_manifest"],
        )

        backoff = Backoff()
        try:
            while not self._stop_event.is_set():
                if self.config.max_games is not None and self.games_completed >= self.config.max_games:
                    break

                try:
                    self._connect()
                except OSError as e:
                    self.log.warning("connect to %s:%s failed: %s", self.config.host, self.config.port, e)
                    delay = backoff.next_delay()
                    self._sleep_unless_stopping(delay)
                    continue

                backoff.reset()
                try:
                    self._session_loop()
                except _StopRequested:
                    break
                except CGOSProtocolError as e:
                    self.log.warning("protocol error, will reconnect: %s", e)
                except OSError as e:
                    self.log.warning("connection lost, will reconnect: %s", e)
                finally:
                    self._close_connection()

                if not self._stop_event.is_set():
                    self._sleep_unless_stopping(backoff.next_delay())
        finally:
            self._close_connection()
            self._engine.shutdown()
        return 0

    def _sleep_unless_stopping(self, delay: float) -> None:
        # Sleep in short slices so a stop request during backoff is noticed promptly rather than
        # only after the full delay elapses.
        deadline = self._clock.monotonic() + delay
        while not self._stop_event.is_set() and self._clock.monotonic() < deadline:
            self._clock.sleep(min(0.2, max(0.0, deadline - self._clock.monotonic())))

    # -- connection management ---------------------------------------------------------

    def _connect(self) -> None:
        sock = socket.create_connection((self.config.host, self.config.port), timeout=CONNECT_TIMEOUT_SECONDS)
        sock.settimeout(None)
        self._io = _LineSocket(sock)
        self.log.info("connected to %s:%s", self.config.host, self.config.port)

    def _close_connection(self) -> None:
        if self._io is not None:
            self._io.close()
            self._io = None

    def _session_loop(self) -> None:
        assert self._io is not None
        self._use_analysis = False
        while True:
            line = self._io.readline()
            if line is None:
                raise CGOSProtocolError("server closed the connection")
            if line == "":
                continue
            if line.startswith("Error:"):
                raise CGOSProtocolError(f"server error: {line[len('Error:'):].strip()}")

            parts = line.split()
            cmd, params = parts[0], parts[1:]
            handler = self._handlers.get(cmd)
            if handler is None:
                self.log.warning("unsupported CGOS command %r (line=%r)", cmd, line)
                continue
            handler(params)

    def _send(self, text: str) -> None:
        assert self._io is not None
        self._io.send_line(text)

    # -- CGOS command handlers ----------------------------------------------------------

    def _on_protocol(self, params: List[str]) -> None:
        server_offers_analysis = "genmove_analyze" in params
        self._use_analysis = self.config.analysis_enabled and server_offers_analysis
        reply = self.config.client_id + (" genmove_analyze" if self._use_analysis else "")
        self._send(reply)

    def _on_username(self, params: List[str]) -> None:
        self._send(self.config.username)

    def _on_password(self, params: List[str]) -> None:
        self._send(self._password)

    def _on_info(self, params: List[str]) -> None:
        self.log.info("server info: %s", " ".join(params))

    def _on_setup(self, params: List[str]) -> None:
        if len(params) < 6:
            raise CGOSProtocolError(f"setup requires at least 6 parameters, got {len(params)}: {params!r}")
        if (len(params) - 6) % 2 != 0:
            raise CGOSProtocolError("setup move/time list must be complete (move, time) pairs")

        gid, boardsize_s, komi_s, level_ms_s, white_spec, black_spec = params[:6]
        move_pairs = params[6:]

        white_name, _white_rating = _split_name_rating(white_spec)
        black_name, _black_rating = _split_name_rating(black_spec)

        if white_name == self.config.username:
            color, opponent = "w", black_name
        elif black_name == self.config.username:
            color, opponent = "b", white_name
        else:
            self.log.warning(
                "setup for game %s names neither side as %r (white=%r black=%r); assuming black",
                gid, self.config.username, white_name, black_name,
            )
            color, opponent = "b", black_name

        try:
            level_ms = int(level_ms_s)
        except ValueError:
            raise CGOSProtocolError(f"setup time_ms is not an integer: {level_ms_s!r}")

        self.log.info(
            "game %s start: opponent=%s color=%s boardsize=%s komi=%s time_ms=%s "
            "engine_binary_sha256=%s model_manifest_sha256=%s resume_moves=%d",
            gid, opponent, color, boardsize_s, komi_s, level_ms,
            self._engine_hashes.get("engine_binary", ""), self._engine_hashes.get("model_manifest", ""),
            len(move_pairs) // 2,
        )

        self._engine.notify_boardsize(boardsize_s)
        self._engine.notify_komi(komi_s)
        self._engine.notify_clear_board()
        self._engine.notify_time_settings(level_ms)

        ledger = MoveLedger.reset(self.config.state_dir, gid, boardsize=boardsize_s, komi=komi_s)

        remaining_ms = {"b": level_ms, "w": level_ms}
        mv_color = "b"
        it = iter(move_pairs)
        for mv, tm in zip(it, it):
            coord = mv.lower()
            self._engine.notify_play(mv_color, coord)
            ledger.append(mv_color, coord, analysis=None, source="replay")
            try:
                remaining_ms[mv_color] = int(tm)
            except ValueError:
                pass
            mv_color = "w" if mv_color == "b" else "b"

        self._engine.notify_time_left("b", remaining_ms["b"])
        self._engine.notify_time_left("w", remaining_ms["w"])

        self._current_gid = gid
        self._current_color = color
        self._ledger = ledger

    def _on_play(self, params: List[str]) -> None:
        if len(params) != 3:
            raise CGOSProtocolError(f"play requires 3 parameters, got {params!r}")
        color, coord, _timeleft_ms = params
        coord = coord.lower()
        self._engine.notify_play(color, coord)
        if self._ledger is not None:
            self._ledger.append(color, coord, analysis=None, source="opponent")

    def _on_genmove(self, params: List[str]) -> None:
        if len(params) != 2:
            raise CGOSProtocolError(f"genmove requires 2 parameters, got {params!r}")
        color, timeleft_ms_s = params
        try:
            timeleft_ms = int(timeleft_ms_s)
        except ValueError:
            raise CGOSProtocolError(f"genmove time_left is not an integer: {timeleft_ms_s!r}")

        if self._current_color is not None and color != self._current_color:
            self.log.warning(
                "genmove asked us to play %s but setup assigned us %s; trusting the server",
                color, self._current_color,
            )

        self._engine.notify_time_left(color, timeleft_ms)
        move, analysis = self._engine.request_genmove(color, use_analysis=self._use_analysis)

        reply = move if analysis is None else f"{move} {analysis}"
        self._send(reply)

        if self._ledger is not None:
            self._ledger.append(color, move, analysis=analysis, source="self")

        self.log.info(
            "game %s: played %s %s%s", self._current_gid, color, move, " (+analysis)" if analysis else "",
        )

    def _on_gameover(self, params: List[str]) -> None:
        if len(params) < 2:
            raise CGOSProtocolError(f"gameover requires at least 2 parameters, got {params!r}")
        date, result = params[0], params[1]
        detail = params[2:]

        self.log.info(
            "game %s over: result=%s%s", self._current_gid, result, f" detail={' '.join(detail)}" if detail else "",
        )

        if self._ledger is not None:
            self._ledger.mark_complete(result)

        self.games_completed += 1
        self._current_gid = None
        self._current_color = None
        self._ledger = None

        should_stop = self._stop_event.is_set() or (
            self.config.max_games is not None and self.games_completed >= self.config.max_games
        )
        if should_stop:
            self.log.info("stopping between games after %d completed game(s)", self.games_completed)
            self._send("quit")
            raise _StopRequested()

        self._send("ready")


# --------------------------------------------------------------------------------------
# Logging / CLI
# --------------------------------------------------------------------------------------


def build_logger(log_dir: Path, name: str = "ichigo_cgos_client") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=LOG_ROTATE_MAX_BYTES,
            backupCount=LOG_ROTATE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ichigo_cgos_client",
        description="IchiGo CGOS client (docs/spec/03-engine.md §9). Never runs concurrently "
        "with another CGOS client sharing the same state_dir/log_dir/username.",
    )
    parser.add_argument("--config", required=True, type=Path, help="path to a client config JSON")
    parser.add_argument(
        "--games", type=int, default=None,
        help="stop after this many completed games (overrides config.max_games; default: unlimited)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"ichigo_cgos_client: invalid config: {e}", file=sys.stderr)
        return 2
    if args.games is not None:
        config = dataclasses.replace(config, max_games=args.games)

    logger = build_logger(config.log_dir)
    try:
        client = CGOSClient(config, log=logger)
    except ConfigError as e:
        print(f"ichigo_cgos_client: invalid config: {e}", file=sys.stderr)
        return 2

    client.install_signal_handlers()
    return client.run()


if __name__ == "__main__":
    raise SystemExit(main())
