#!/usr/bin/env python3
"""Life-and-death ("死活") benchmark over IchiGo's *own* games, and the DAgger split that feeds
the same games' remaining positions back into training (docs/progress-summary.md §5,
2026-09-23: both CGOS losses were a group the model called alive while the teacher called it dead).

  split   positions JSONL (`ichigo features` over own-game SGFs) -> bench / train positions,
          split by game so no bench game leaks into training. Bench = every CGOS and gnugo game
          plus sha256(gameId) % 10 == 0 of the rest. Positions are de-duplicated by positionId
          and any positionId seen in a bench game is dropped from train (shared openings).
  score   bench positions + teacher labels + model outputs (`ichigo eval-batch`) -> metrics.
  run     one .ichigo model end to end: picks bench-v<featureVersion>.jsonl from --bench-dir
          (the model manifest says which), runs `ichigo eval-batch`, then `score`.

A stone is "teacher-dead" when the teacher's ownership (side-to-move view) points the other way
from the stone's colour with |ownership| >= --dead-threshold, and "teacher-alive" when it points
the same way with at least that magnitude; stones in between are left out. The model gets a stone
right when the sign of its ownership agrees. deadRecall is the headline number: of the stones the
teacher says are dead, how many the model also sees as dead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

OWN_STONE, OPP_STONE = 0, 1  # spatial channels (docs/spec/01-network.md §1), same in v1 and v2
CHANNELS = 32


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_bench_game(source_file: str, game_id: str) -> bool:
    if source_file.startswith(("cgos-", "gnugo-")):
        return True
    return int(hashlib.sha256(game_id.encode()).hexdigest(), 16) % 10 == 0


def cmd_split(a) -> int:
    bench_ids: set[str] = set()
    rows_by_side = {"bench": {}, "train": {}}
    for r in iter_jsonl(a.positions):
        side = "bench" if is_bench_game(r.get("sourceFile", ""), r["gameId"]) else "train"
        if side == "bench":
            bench_ids.add(r["positionId"])
        if r["turnNumber"] < a.min_turn:
            continue
        rows_by_side[side].setdefault(r["positionId"], r)
    stats = {}
    for side, out in (("bench", a.bench_out), ("train", a.train_out)):
        n = 0
        with open(out, "w") as f:
            for pid, r in rows_by_side[side].items():
                if side == "train" and pid in bench_ids:
                    continue
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
                n += 1
        stats[side] = n
    print(json.dumps({"benchPositions": stats["bench"], "trainPositions": stats["train"], "minTurn": a.min_turn}))
    return 0


def cmd_score(a) -> int:
    labels = {r["positionId"]: r for r in iter_jsonl(a.labels)}
    outputs = {r["positionId"]: r for r in iter_jsonl(a.model_outputs)}
    dead_total = dead_hit = alive_total = alive_hit = 0
    positions = value_n = 0
    abs_err = signed_err = 0.0
    dead_positions = dead_positions_all_missed = 0
    for r in iter_jsonl(a.positions):
        pid = r["positionId"]
        lab, out = labels.get(pid), outputs.get(pid)
        if lab is None or out is None:
            continue
        positions += 1
        S = r["boardSize"]
        spatial = r["spatial"]
        t_own, m_own = lab["ownership"], out["ownership"]
        dead_here = hit_here = 0
        for p in range(S * S):
            own = spatial[p * CHANNELS + OWN_STONE]
            opp = spatial[p * CHANNELS + OPP_STONE]
            if not (own or opp):
                continue
            colour = 1.0 if own else -1.0  # side-to-move view, like the ownership targets
            t = t_own[p] * colour
            m = m_own[p] * colour
            if t <= -a.dead_threshold:
                dead_total += 1
                dead_here += 1
                if m < 0:
                    dead_hit += 1
                    hit_here += 1
            elif t >= a.dead_threshold:
                alive_total += 1
                if m > 0:
                    alive_hit += 1
        if dead_here >= a.min_dead_stones:
            dead_positions += 1
            if hit_here == 0:
                dead_positions_all_missed += 1
        if lab.get("expectedResult") is not None:
            value_n += 1
            d = out["expectedResult"] - lab["expectedResult"]
            abs_err += abs(d)
            signed_err += d
    res = {
        "positions": positions,
        "deadStones": dead_total,
        "deadRecall": dead_hit / dead_total if dead_total else None,
        "aliveStones": alive_total,
        "aliveAccuracy": alive_hit / alive_total if alive_total else None,
        "positionsWithDeadGroup": dead_positions,
        "positionsDeadGroupEntirelyMissed": dead_positions_all_missed,
        "valueMAE": abs_err / value_n if value_n else None,
        "valueBias": signed_err / value_n if value_n else None,
        "deadThreshold": a.dead_threshold,
        "minDeadStones": a.min_dead_stones,
    }
    if a.model_label:
        res["model"] = a.model_label
    print(json.dumps(res, indent=2))
    return 0


def cmd_run(a) -> int:
    with open(os.path.join(a.model, "manifest.json")) as f:
        version = json.load(f)["featureVersion"]
    positions = os.path.join(a.bench_dir, f"bench-v{version}.jsonl")
    with tempfile.TemporaryDirectory() as td:
        outputs = os.path.join(td, "outputs.jsonl")
        subprocess.run([a.ichigo, "eval-batch", "--model", a.model, "--positions", positions, "--out", outputs], check=True)
        return cmd_score(argparse.Namespace(
            positions=positions, labels=os.path.join(a.bench_dir, "bench-labels.jsonl"), model_outputs=outputs,
            dead_threshold=a.dead_threshold, min_dead_stones=a.min_dead_stones,
            model_label=a.model_label or os.path.basename(os.path.normpath(a.model)),
        ))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--positions", required=True)
    s.add_argument("--bench-out", required=True)
    s.add_argument("--train-out", required=True)
    s.add_argument("--min-turn", type=int, default=0, help="drop positions before this turn (both sides)")
    s.set_defaults(func=cmd_split)
    c = sub.add_parser("score")
    c.add_argument("--positions", required=True, help="bench positions JSONL (any feature version; only stones are read)")
    c.add_argument("--labels", required=True, help="teacher labels JSONL (ichigo_train label)")
    c.add_argument("--model-outputs", required=True, help="ichigo eval-batch output JSONL")
    c.add_argument("--dead-threshold", type=float, default=0.6)
    c.add_argument("--min-dead-stones", type=int, default=3, help="a position counts as having a dead group at this many teacher-dead stones")
    c.add_argument("--model-label", default=None)
    c.set_defaults(func=cmd_score)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, help=".ichigo directory")
    r.add_argument("--bench-dir", default="data/dagger-9")
    r.add_argument("--ichigo", default=".build/release/ichigo")
    r.add_argument("--dead-threshold", type=float, default=0.6)
    r.add_argument("--min-dead-stones", type=int, default=3)
    r.add_argument("--model-label", default=None)
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
