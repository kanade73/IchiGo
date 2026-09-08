#!/usr/bin/env python3
"""Teacher self-play SGF generation (docs/spec/02-training.md §3, T13). Used only when no game
records are available (the current 9x9 corpus makes this unnecessary; the script is provided but
was not run at scale).

One KataGo analysis-engine process evaluates the current position; the move is sampled from the
visit distribution with temperature 1 for the first 20 moves and 0.2 afterwards. No resignation.
The move cap is 4*S*S; games hitting it are listed as `truncated` and must not receive a result
target (they remain usable as teacher-evaluated positions).

Usage: generate_teacher_games.py --katago BIN --model M --config CFG --games N --size 9 --komi 7 --out DIR --seed S [--visits 128]
Outputs: DIR/game-XXXXX.sgf, DIR/generation.json (settings, seed, model sha256, truncated list).
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np

GTP = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
RULES = {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "suicide": False, "hasButton": False, "whiteHandicapBonus": "0", "friendlyPassOk": True}


def sgf_coord(move, size):
    if move == "pass":
        return ""
    x = GTP.index(move[0]); y = size - int(move[1:])
    return chr(97 + x) + chr(97 + y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True); ap.add_argument("--model", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--games", type=int, required=True); ap.add_argument("--size", type=int, default=9); ap.add_argument("--komi", type=float, default=7.0)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--visits", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(a.seed))
    proc = subprocess.Popen([a.katago, "analysis", "-config", a.config, "-model", a.model], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    cap = 4 * a.size * a.size
    truncated = []
    t0 = time.time()
    for g in range(a.games):
        moves = []
        passes = 0
        while len(moves) < cap and passes < 2:
            q = {"id": f"g{g}m{len(moves)}", "moves": moves, "initialStones": [], "initialPlayer": "B", "rules": RULES, "komi": a.komi,
                 "boardXSize": a.size, "boardYSize": a.size, "analyzeTurns": [len(moves)], "maxVisits": a.visits, "includeOwnership": False, "includePolicy": False}
            proc.stdin.write(json.dumps(q) + "\n"); proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            if "error" in resp:
                print("teacher error", resp, file=sys.stderr); sys.exit(3)
            infos = [(m["move"], m["visits"]) for m in resp["moveInfos"] if m["visits"] > 0]
            if not infos:
                move = "pass"
            else:
                temp = 1.0 if len(moves) < 20 else 0.2
                v = np.array([x[1] for x in infos], dtype=np.float64) ** (1.0 / temp)
                move = infos[int(rng.choice(len(infos), p=v / v.sum()))][0]
            color = "B" if len(moves) % 2 == 0 else "W"
            moves.append([color, move])
            passes = passes + 1 if move == "pass" else 0
        if len(moves) >= cap:
            truncated.append(g)
        sgf = f"(;GM[1]FF[4]SZ[{a.size}]KM[{a.komi:g}]RU[Chinese]PB[teacher]PW[teacher]" + "".join(f";{c}[{sgf_coord(m, a.size)}]" for c, m in moves) + ")"
        with open(os.path.join(a.out, f"game-{g:05d}.sgf"), "w") as f:
            f.write(sgf + "\n")
        print(f"game {g}: {len(moves)} moves{' (truncated)' if g in truncated else ''}", file=sys.stderr, flush=True)
    proc.stdin.close(); proc.wait(timeout=30)
    json.dump({"games": a.games, "size": a.size, "komi": a.komi, "seed": a.seed, "visits": a.visits, "temperature": {"first20": 1.0, "after": 0.2},
               "moveCap": cap, "truncated": truncated, "modelSha256": hashlib.sha256(open(a.model, "rb").read()).hexdigest(),
               "elapsedSeconds": time.time() - t0, "note": "SGF RE is intentionally absent: results come from on-board scoring in the dataset builder"},
              open(os.path.join(a.out, "generation.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
