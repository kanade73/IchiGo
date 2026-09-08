#!/usr/bin/env python3
"""Split a positions JSONL into N chunks at game boundaries (for parallel teacher labelling)."""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--positions", required=True); ap.add_argument("--chunks", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
total = sum(1 for _ in open(a.positions))
per = total // a.chunks + 1
files = [open(os.path.join(a.out, f"chunk-{i:02d}.jsonl"), "w") for i in range(a.chunks)]
i = n = 0; last = None
with open(a.positions) as f:
    for line in f:
        gid = json.loads(line)["gameId"]
        if n >= per and gid != last and i < a.chunks - 1:
            i += 1; n = 0
        files[i].write(line); n += 1; last = gid
for fh in files: fh.close()
print("chunks", a.chunks, "total", total)
