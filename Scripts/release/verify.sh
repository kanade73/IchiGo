#!/usr/bin/env bash
# Scripts/release/verify.sh <dist-dir>
#
# T37 release verification (docs/spec/04-tasks.md T37; docs/spec/05-validation.md SS1/SS8).
# Standalone: takes only <dist-dir> and does not assume anything about the caller's cwd or about
# this repository -- run it from anywhere, pointing at any ichigo-*-*/ directory
# Scripts/release/build.sh produced, to prove the packaged binary/resources/model are genuinely
# self-contained and path-independent (this is exercised for real: the release-check flow runs
# it once right after build.sh, and it is meant to be re-run a second time from a different cwd).
#
# Checks, all run even if an earlier one fails (every failure is reported, not just the first):
#   1. Re-hash every file MANIFEST.json lists against what's actually on disk (sha256 + size);
#      report missing files, extra files not in the manifest, and hash/size mismatches.
#   2. `ichigo doctor` exits 0 and prints valid JSON.
#   3. `ichigo inspect --model <dist>/<manifest model path>` exits 0 and its payloadHash matches
#      MANIFEST.json's recorded model payload hash.
#   4. A 3-move GTP smoke test (boardsize/clear_board/komi/3x genmove/quit) on --backend
#      cpu-packed.
#   5. The same smoke test on --backend auto.
# Exits non-zero if any check failed.
set -uo pipefail # deliberately not -e: we want every failure, not just the first

DIST_DIR_ARG=${1:?"usage: Scripts/release/verify.sh <dist-dir>"}
[ -d "$DIST_DIR_ARG" ] || { echo "verify.sh: not a directory: $DIST_DIR_ARG" >&2; exit 2; }
DIST_DIR=$(cd "$DIST_DIR_ARG" && pwd)
[ -f "$DIST_DIR/MANIFEST.json" ] || { echo "verify.sh: no MANIFEST.json in $DIST_DIR" >&2; exit 2; }
[ -x "$DIST_DIR/ichigo" ] || { echo "verify.sh: no executable 'ichigo' in $DIST_DIR" >&2; exit 2; }

FAILED=0
ok()   { echo "verify.sh: OK   -- $1" >&2; }
fail() { echo "verify.sh: FAIL -- $1" >&2; FAILED=1; }

echo "== verify.sh: $DIST_DIR (cwd=$(pwd)) ==" >&2

# 1. Re-hash every manifest file.
MANIFEST_REPORT=$(python3 - "$DIST_DIR" <<'PY'
import hashlib, json, os, sys

dist_dir = sys.argv[1]
manifest = json.load(open(os.path.join(dist_dir, "MANIFEST.json")))
expected = manifest.get("files", {})

present = {}
for root, dirs, names in os.walk(dist_dir):
    dirs.sort()
    for name in sorted(names):
        path = os.path.join(root, name)
        rel = os.path.relpath(path, dist_dir)
        if rel == "MANIFEST.json":
            continue
        present[rel] = path

problems = []
for rel, meta in expected.items():
    if rel not in present:
        problems.append(f"missing on disk: {rel}")
        continue
    h = hashlib.sha256()
    with open(present[rel], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != meta.get("sha256"):
        problems.append(f"sha256 mismatch: {rel} (manifest {meta.get('sha256')}, actual {digest})")
    size = os.path.getsize(present[rel])
    if "bytes" in meta and size != meta["bytes"]:
        problems.append(f"size mismatch: {rel} (manifest {meta['bytes']}, actual {size})")
for rel in present:
    if rel not in expected:
        problems.append(f"extra file not listed in MANIFEST.json: {rel}")

print(f"checked {len(expected)} manifest entries against {len(present)} files on disk")
for p in problems:
    print("PROBLEM: " + p)
sys.exit(1 if problems else 0)
PY
)
MANIFEST_STATUS=$?
echo "$MANIFEST_REPORT" >&2
if [ "$MANIFEST_STATUS" -eq 0 ]; then
  ok "MANIFEST.json re-hash (all files present, sha256/size match, no extras)"
else
  fail "MANIFEST.json re-hash found problems (see PROBLEM lines above)"
fi

# 2. ichigo doctor.
DOCTOR_OUT=$("$DIST_DIR/ichigo" doctor 2>&1)
DOCTOR_STATUS=$?
if [ "$DOCTOR_STATUS" -eq 0 ] && printf '%s' "$DOCTOR_OUT" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
  ok "ichigo doctor (exit 0, valid JSON)"
else
  fail "ichigo doctor did not exit 0 with valid JSON (exit=$DOCTOR_STATUS): $DOCTOR_OUT"
fi

# 3. ichigo inspect --model, compared against MANIFEST.json's recorded payload hash.
MODEL_REL=$(python3 -c "import json; print(json.load(open('$DIST_DIR/MANIFEST.json'))['model']['path'])" 2>/dev/null || echo "")
if [ -z "$MODEL_REL" ]; then
  fail "MANIFEST.json has no model.path; cannot run ichigo inspect"
  MODEL_DIR=""
else
  MODEL_DIR="$DIST_DIR/$MODEL_REL"
  INSPECT_OUT=$("$DIST_DIR/ichigo" inspect --model "$MODEL_DIR" 2>&1)
  INSPECT_STATUS=$?
  if [ "$INSPECT_STATUS" -eq 0 ]; then
    GOT_HASH=$(printf '%s' "$INSPECT_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['payloadHash'])" 2>/dev/null || echo "")
    EXPECT_HASH=$(python3 -c "import json; print(json.load(open('$DIST_DIR/MANIFEST.json'))['model']['payloadHash'])" 2>/dev/null || echo "")
    if [ -n "$GOT_HASH" ] && [ "$GOT_HASH" = "$EXPECT_HASH" ]; then
      ok "ichigo inspect --model (payloadHash $GOT_HASH matches MANIFEST.json)"
    else
      fail "ichigo inspect --model payloadHash mismatch (manifest=$EXPECT_HASH, got=$GOT_HASH)"
    fi
  else
    fail "ichigo inspect --model exited $INSPECT_STATUS: $INSPECT_OUT"
  fi
fi

# 4/5. 3-move GTP smoke on cpu-packed then auto.
BOARD_SIZE=9
if [ -n "${INSPECT_OUT:-}" ]; then
  BOARD_SIZE=$(printf '%s' "$INSPECT_OUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['boardSizes'][0])" 2>/dev/null || echo 9)
fi
if [ "$BOARD_SIZE" = "9" ]; then MODEL_FLAG="--model-9"; KOMI=7; else MODEL_FLAG="--model-19"; KOMI=7.5; fi

gtp_smoke() {
  backend="$1"
  if [ -z "$MODEL_DIR" ]; then
    echo "PROBLEMS:"
    echo " - no model directory resolved (MANIFEST.json missing model.path)"
    return 1
  fi
  python3 - "$DIST_DIR/ichigo" "$MODEL_FLAG" "$MODEL_DIR" "$backend" "$BOARD_SIZE" "$KOMI" <<'PY'
import subprocess, sys

ichigo, model_flag, model_dir, backend, board_size, komi = sys.argv[1:7]
cmds = [
    f"1 boardsize {board_size}",
    "2 clear_board",
    f"3 komi {komi}",
    "4 genmove b",
    "5 genmove w",
    "6 genmove b",
    "7 quit",
]
stdin_text = "\n".join(cmds) + "\n"
proc = subprocess.Popen(
    [ichigo, "gtp", model_flag, model_dir, "--backend", backend, "--visits", "8"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
try:
    out, err = proc.communicate(input=stdin_text, timeout=90)
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate()
    print(f"exit=timeout(90s)")
    print("PROBLEMS:")
    print(" - process did not finish within 90s; stderr tail:")
    print(err[-4000:])
    sys.exit(1)

lines = [l for l in out.splitlines() if l.strip()]
ok_lines = [l for l in lines if l.startswith("=")]
err_lines = [l for l in lines if l.startswith("?")]
print(f"exit={proc.returncode} ok_replies={len(ok_lines)} error_replies={len(err_lines)}")
problems = []
if proc.returncode != 0:
    problems.append(f"process exited {proc.returncode}; stderr tail:\n{err[-4000:]}")
if err_lines:
    problems.append("GTP error reply(ies): " + " | ".join(err_lines))
if len(ok_lines) < 7:
    problems.append(f"expected >=7 successful replies (boardsize/clear_board/komi/3x genmove/quit), got {len(ok_lines)}: {lines}")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(" - " + p)
    sys.exit(1)
sys.exit(0)
PY
}

SMOKE_CPU_PACKED=$(gtp_smoke cpu-packed 2>&1)
if [ $? -eq 0 ]; then
  ok "3-move GTP smoke on --backend cpu-packed (boardsize/clear_board/komi/3x genmove/quit, 0 errors)"
else
  fail "3-move GTP smoke on --backend cpu-packed:"
  echo "$SMOKE_CPU_PACKED" >&2
fi

SMOKE_AUTO=$(gtp_smoke auto 2>&1)
if [ $? -eq 0 ]; then
  ok "3-move GTP smoke on --backend auto (boardsize/clear_board/komi/3x genmove/quit, 0 errors)"
else
  fail "3-move GTP smoke on --backend auto:"
  echo "$SMOKE_AUTO" >&2
fi

echo "==================================================" >&2
if [ "$FAILED" -eq 0 ]; then
  echo "verify.sh: ALL CHECKS PASSED for $DIST_DIR" >&2
  exit 0
else
  echo "verify.sh: FAILED for $DIST_DIR (see FAIL lines above)" >&2
  exit 1
fi
