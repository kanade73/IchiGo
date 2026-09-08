"""KataGo analysis-engine teacher adapter (docs/spec/02-training.md §3, T12).

The teacher is an external process speaking the KataGo analysis JSONL protocol. One query per
game carries every requested turn in ``analyzeTurns``; responses are joined on ``(id, turnNumber)``
regardless of order. Duplicates are ignored, missing turns are retried (whole query resubmitted
under a new id, at most ``retries`` times) and finally rejected.

Perspective is explicit: ``perspective`` must match the teacher's ``reportAnalysisWinratesAs``
(``black``/``white``/``sidetomove``) and every value (winrate, scoreLead, ownership) is converted
to the side-to-move perspective before it is written.

Output label row (JSONL):
  positionId, gameId, turnNumber, toMove, boardSize,
  policy: dense list [S*S+1] (visits normalised over legal moves, illegal = 0),
  expectedResult (to-move winrate in [0,1]), score (to-move lead), ownership [S*S] to-move,
  teacherVisits, sourceType="teacher", teacherId
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

RULES_JSON = {
    "ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "suicide": False,
    "hasButton": False, "whiteHandicapBonus": "0", "friendlyPassOk": True,
}
RULES_ID = "cgos-area-psk-v1"
GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


@dataclass
class TeacherConfig:
    command: list[str]                      # argv, e.g. ["katago","analysis","-config",cfg,"-model",model]
    teacher_id: str                         # free-form id recorded in every label (binary+model hash)
    perspective: str = "sidetomove"         # black | white | sidetomove (== reportAnalysisWinratesAs)
    visits: int = 128
    timeout_seconds: float = 60.0           # idle timeout: reset whenever a turn of the query is answered;
                                            # the first response may take timeout*queue_depth (queueing)
    retries: int = 2
    queue_depth: int = 16                   # max in-flight queries
    stderr_log: str | None = None


@dataclass
class TeacherStats:
    queries: int = 0
    responses: int = 0
    duplicates: int = 0
    labels: int = 0
    rejected: int = 0
    retries: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1


def gtp_to_index(move: str, size: int) -> int | None:
    m = move.strip().upper()
    if m == "PASS":
        return size * size
    if len(m) < 2 or m[0] not in GTP_LETTERS:
        return None
    x = GTP_LETTERS.index(m[0])
    try:
        row = int(m[1:])
    except ValueError:
        return None
    if not (1 <= row <= size and x < size):
        return None
    return (size - row) * size + x


def to_move_sign(perspective: str, to_move: str) -> float:
    """Multiplier turning a value in ``perspective`` into the to-move perspective (score/ownership)."""
    if perspective == "sidetomove":
        return 1.0
    if perspective not in ("black", "white"):
        raise ValueError(f"unknown perspective {perspective!r}")
    return 1.0 if perspective[0].upper() == to_move else -1.0


def convert_winrate(w: float, perspective: str, to_move: str) -> float:
    return w if to_move_sign(perspective, to_move) > 0 else 1.0 - w


def build_query(qid: str, game: dict, turns: list[int], visits: int) -> dict:
    return {
        "id": qid,
        "moves": game["moves"],
        "initialStones": game["initialStones"],
        "initialPlayer": game["initialPlayer"],
        "rules": RULES_JSON,
        "komi": game["komi"],
        "boardXSize": game["boardSize"],
        "boardYSize": game["boardSize"],
        "analyzeTurns": sorted(turns),
        "maxVisits": visits,
        "includeOwnership": True,
        "includePolicy": False,
    }


def response_to_label(resp: dict, pos: dict, cfg: TeacherConfig) -> tuple[dict | None, str | None]:
    """Convert one analysis response for one position. Returns (label, reject_reason)."""
    S = pos["boardSize"]
    to_move = pos["toMove"]
    root = resp.get("rootInfo")
    if not isinstance(root, dict) or "winrate" not in root or "scoreLead" not in root:
        return None, "missing rootInfo"
    if root.get("currentPlayer") not in (None, to_move):
        return None, "currentPlayer mismatch"
    legal = pos["legal"]
    if len(legal) != S * S + 1:
        return None, "legal length mismatch"
    policy = [0.0] * (S * S + 1)
    total = 0.0
    for mi in resp.get("moveInfos", []):
        idx = gtp_to_index(str(mi.get("move", "")), S)
        v = mi.get("visits", 0)
        if idx is None or not isinstance(v, (int, float)) or v < 0:
            return None, "bad moveInfo"
        if v == 0:
            continue
        if legal[idx] != 1:
            return None, "teacher visited illegal move"
        policy[idx] += float(v)
        total += float(v)
    if total <= 0:
        return None, "policy sum zero"
    policy = [p / total for p in policy]
    own = resp.get("ownership")
    if not isinstance(own, list) or len(own) != S * S:
        return None, "ownership length mismatch"
    sign = to_move_sign(cfg.perspective, to_move)
    try:
        winrate = float(root["winrate"])
        lead = float(root["scoreLead"])
        own_f = [sign * float(o) for o in own]
    except (TypeError, ValueError):
        return None, "non-numeric value"
    expected = convert_winrate(winrate, cfg.perspective, to_move)
    score = sign * lead
    vals = [expected, score] + own_f
    if not all(math.isfinite(v) for v in vals):
        return None, "non-finite value"
    if not (0.0 <= expected <= 1.0) or any(abs(o) > 1.0 + 1e-6 for o in own_f):
        return None, "value out of range"
    return {
        "positionId": pos["positionId"], "gameId": pos["gameId"], "turnNumber": pos["turnNumber"],
        "toMove": to_move, "boardSize": S, "policy": policy, "expectedResult": expected, "score": score,
        "ownership": [max(-1.0, min(1.0, o)) for o in own_f],
        "teacherVisits": int(root.get("visits", 0)), "sourceType": "teacher", "teacherId": cfg.teacher_id,
    }, None


class TeacherProcess:
    """Owns the subprocess and the reader thread. ``submit`` is non-blocking; ``responses`` is a queue."""

    def __init__(self, cfg: TeacherConfig):
        self.cfg = cfg
        self.proc = subprocess.Popen(cfg.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(cfg.stderr_log, "ab") if cfg.stderr_log else subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.responses: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    self.responses.put({"error": "unparseable response", "raw": line[:200]})
        finally:
            self.responses.put(None)

    def submit(self, query: dict) -> None:
        self.proc.stdin.write(json.dumps(query) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def label_positions(positions_path: str, out_path: str, cfg: TeacherConfig, max_positions: int | None = None,
                    progress=None) -> TeacherStats:
    """Streams positions JSONL (grouped by gameId in file order), labels them, writes labels JSONL
    and ``<out>.rejects.jsonl``. Never loads all features into RAM: positions are held only while
    their query is in flight."""
    stats = TeacherStats()
    rejects = open(out_path + ".rejects.jsonl", "w")
    out = open(out_path, "w")
    proc = TeacherProcess(cfg)
    in_flight: dict[str, dict] = {}   # qid -> {"game": game, "positions": {turn: pos}, "attempt": n, "sent": t, "got": set()}
    game_key_to_qid: dict[str, str] = {}
    counter = 0

    def flush_group(game: dict, positions: dict[int, dict], attempt: int):
        nonlocal counter
        counter += 1
        qid = f"{game['gameId'][:24]}-{counter}"
        in_flight[qid] = {"game": game, "positions": positions, "attempt": attempt, "sent": time.monotonic(), "last": None, "got": set()}
        proc.submit(build_query(qid, game, list(positions), cfg.visits))
        stats.queries += 1

    def handle_response(resp: dict):
        if resp is None:
            return
        qid = resp.get("id")
        entry = in_flight.get(qid)
        if entry is None:
            return  # stale (already retried/finished) or unknown
        if "error" in resp:
            for pos in entry["positions"].values():
                stats.reject(f"teacher error: {str(resp.get('error'))[:80]}")
                rejects.write(json.dumps({"positionId": pos["positionId"], "reason": str(resp.get("error"))}) + "\n")
            del in_flight[qid]
            return
        turn = resp.get("turnNumber")
        if turn in entry["got"]:
            stats.duplicates += 1
            return
        pos = entry["positions"].get(turn)
        if pos is None:
            return
        stats.responses += 1
        entry["got"].add(turn)
        entry["last"] = time.monotonic()
        label, reason = response_to_label(resp, pos, cfg)
        if label is None:
            stats.reject(reason)
            rejects.write(json.dumps({"positionId": pos["positionId"], "reason": reason}) + "\n")
        else:
            out.write(json.dumps(label) + "\n")
            stats.labels += 1
        if entry["got"] == set(entry["positions"]):
            del in_flight[qid]

    def drain(block_until_below: int):
        while len(in_flight) > block_until_below:
            try:
                resp = proc.responses.get(timeout=1.0)
            except queue.Empty:
                resp = None
            if resp is not None:
                handle_response(resp)
            check_timeouts()
            if proc.proc.poll() is not None and proc.responses.empty():
                break

    def check_timeouts():
        now = time.monotonic()
        for qid in list(in_flight):
            e = in_flight[qid]
            deadline = (e["last"] + cfg.timeout_seconds) if e["last"] is not None else (e["sent"] + cfg.timeout_seconds * max(1, cfg.queue_depth))
            if now > deadline:
                missing = {t: p for t, p in e["positions"].items() if t not in e["got"]}
                del in_flight[qid]
                if e["attempt"] < cfg.retries:
                    stats.retries += 1
                    flush_group(e["game"], missing, e["attempt"] + 1)
                else:
                    for pos in missing.values():
                        stats.reject("timeout")
                        rejects.write(json.dumps({"positionId": pos["positionId"], "reason": "timeout"}) + "\n")

    current_game = None
    current_positions: dict[int, dict] = {}
    n = 0
    with open(positions_path) as f:
        for line in f:
            if max_positions is not None and n >= max_positions:
                break
            pos = json.loads(line)
            n += 1
            gid = pos["gameId"]
            if current_game is not None and gid != current_game["gameId"]:
                drain(cfg.queue_depth - 1)
                flush_group(current_game, current_positions, 0)
                current_positions = {}
            if current_game is None or gid != current_game["gameId"]:
                current_game = {"gameId": gid, "moves": None, "initialStones": pos["initialStones"], "initialPlayer": pos["initialPlayer"], "komi": pos["komi"], "boardSize": pos["boardSize"]}
            if current_game["moves"] is None or len(pos["moves"]) > len(current_game["moves"]):
                current_game["moves"] = pos["moves"]
            current_positions[pos["turnNumber"]] = {k: pos[k] for k in ("positionId", "gameId", "turnNumber", "toMove", "boardSize", "legal")}
            if progress and n % 1000 == 0:
                progress(n, stats)
    if current_game is not None and current_positions:
        flush_group(current_game, current_positions, 0)
    # Wait for everything (respecting timeouts/retries)
    while in_flight:
        before = len(in_flight)
        drain(0)
        if proc.proc.poll() is not None and len(in_flight) == before:
            # process died: everything left is rejected
            for e in list(in_flight.values()):
                for t, pos in e["positions"].items():
                    if t not in e["got"]:
                        stats.reject("teacher process exited")
                        rejects.write(json.dumps({"positionId": pos["positionId"], "reason": "teacher process exited"}) + "\n")
            in_flight.clear()
    proc.close()
    out.close()
    rejects.close()
    return stats
