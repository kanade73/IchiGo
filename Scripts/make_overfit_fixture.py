#!/usr/bin/env python3
"""Build the 16-position overfit fixture (docs/spec/02-training.md §7, T17) from labelled positions.

Picks 16 positions from different games with pairwise different legal masks and spread-out
expected results, writes <out>/fixture.npz (shard v1 arrays) and <out>/fixture-index.json.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Training"))
from ichigo_train.dataset import ShardBuffer, validate_arrays  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--positions", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--count", type=int, default=16)
a = ap.parse_args()

labels = {}
with open(a.labels) as f:
    for line in f:
        l = json.loads(line)
        labels.setdefault(l["positionId"], l)
chosen, games, legal_seen = [], set(), set()
targets = np.linspace(0.05, 0.95, a.count)
cands = []
with open(a.positions) as f:
    for line in f:
        r = json.loads(line)
        l = labels.get(r["positionId"])
        if l is None or r["turnNumber"] < 6:
            continue
        key = tuple(r["legal"])
        if r["gameId"] in games or key in legal_seen:
            continue
        games.add(r["gameId"]); legal_seen.add(key)
        cands.append((r, l))
        if len(cands) >= a.count * 20:
            break
# spread expected results: greedily pick the candidate closest to each target value
used = set()
for t in targets:
    best = min((i for i in range(len(cands)) if i not in used), key=lambda i: abs(cands[i][1]["expectedResult"] - t))
    used.add(best); chosen.append(cands[best])
S = chosen[0][0]["boardSize"]
buf = ShardBuffer(S)
index = []
for r, l in chosen:
    buf.add({"spatial": r["spatial"], "global": r["global"], "legal": r["legal"], "policy": l["policy"], "expected_result": l["expectedResult"],
             "score": l["score"], "ownership": l["ownership"], "wdl": [0, 0, 0], "mask": [1, 1, 1, 1, 0], "positionId": r["positionId"], "gameId": r["gameId"]})
    index.append({"positionId": r["positionId"], "gameId": r["gameId"], "turnNumber": r["turnNumber"], "expectedResult": l["expectedResult"], "top": int(np.argmax(l["policy"])), "sourceFile": r.get("sourceFile")})
arrays = buf.to_arrays()
validate_arrays(arrays, S)
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, "fixture.npz"), **arrays)
json.dump({"count": len(chosen), "boardSize": S, "positions": index, "note": "16-position overfit fixture; not split, train == validation"}, open(os.path.join(a.out, "fixture-index.json"), "w"), indent=2)
print(f"wrote {len(chosen)} positions to {a.out}; expected results {[round(l['expectedResult'], 2) for _, l in chosen]}")
