#!/usr/bin/env bash
# Scripts/release/build.sh MODEL_DIR [DIST_DIR]
#
# T37 release build (docs/spec/04-tasks.md T37; docs/spec/05-validation.md SS1/SS8;
# docs/spec/03-engine.md SS9). Builds `ichigo` in release configuration and assembles a
# self-contained directory under dist/ichigo-<version>-<arch>/ containing:
#   - the release binary and the Metal-shader resource bundle(s) SwiftPM ships next to it
#   - the given model directory (MODEL_DIR, e.g. models/p4-local-wide512-200k.ichigo)
#   - configs/cgos.example.json and Scripts/cgos/*.py (the CGOS client + fake server)
#   - MANIFEST.json: sha256 of every shipped file, the model payload hash (from `ichigo
#     inspect`), toolchain/OS versions, and the git commit
#
# Run Scripts/release/verify.sh <dist-dir> afterwards -- it re-derives everything from scratch
# and does not trust this script's work.
#
# Resource lookup note (checked, not assumed): SwiftPM's generated `Bundle.module` accessor for
# LogicMetal tries `Bundle.main.bundleURL.appendingPathComponent("IchiGo_LogicMetal.bundle")`
# first, falling back to a hardcoded absolute .build path only if that fails. For a plain (non
# .app) executable, `Bundle.main.bundleURL` is the directory containing the running binary --
# *not* cwd and *not* the build machine's .build path -- so copying the binary and its
# `*.bundle` directory into the same directory (as this script does) is sufficient; no code
# change was needed in Sources/LogicMetal. Verified directly below: this script runs the packaged
# `ichigo eval --backend metal` from the assembled dist/ directory (a location with no
# relationship to .build) and treats a failure as fatal when a Metal device is present.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

MODEL_DIR=${1:?"usage: Scripts/release/build.sh MODEL_DIR [DIST_DIR]"}
[ -f "$MODEL_DIR/manifest.json" ] || { echo "build.sh: not a .ichigo model directory (no manifest.json): $MODEL_DIR" >&2; exit 2; }
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
MODEL_NAME=$(basename "$MODEL_DIR")

echo "== swift build -c release --product ichigo ==" >&2
# swift build's own progress output goes to stdout by default; redirect it to stderr so this
# script's stdout carries only the final dist-dir path (callers do `DIST=$(build.sh ...)`).
swift build -c release --product ichigo >&2
BIN_DIR=$(swift build -c release --show-bin-path)
BIN_PATH="$BIN_DIR/ichigo"
[ -x "$BIN_PATH" ] || { echo "build.sh: $BIN_PATH missing after swift build" >&2; exit 2; }

if git rev-parse --git-dir >/dev/null 2>&1; then
  GIT_COMMIT=$(git rev-parse HEAD)
  if [ -z "$(git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY=false; else GIT_DIRTY=true; fi
else
  GIT_COMMIT="unknown"
  GIT_DIRTY=true
fi
PY_PROJECT_VERSION=$(python3 -c "
import re
text = open('Training/pyproject.toml').read()
m = re.search(r'(?m)^version\s*=\s*\"([^\"]+)\"', text)
print(m.group(1) if m else '0.0.0')
")
VERSION="${PY_PROJECT_VERSION}-${GIT_COMMIT:0:12}"
if [ "$GIT_DIRTY" = true ]; then VERSION="${VERSION}-dirty"; fi
ARCH=$(uname -m)
DIST_DIR=${2:-"$REPO_ROOT/dist/ichigo-$VERSION-$ARCH"}

echo "== packaging into $DIST_DIR ==" >&2
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/models" "$DIST_DIR/configs" "$DIST_DIR/Scripts/cgos"

cp "$BIN_PATH" "$DIST_DIR/ichigo"
chmod +x "$DIST_DIR/ichigo"

shopt -s nullglob
BUNDLES=("$BIN_DIR"/*.bundle)
shopt -u nullglob
if [ ${#BUNDLES[@]} -eq 0 ]; then
  echo "build.sh: no *.bundle resource directories found next to $BIN_PATH (expected the LogicMetal Metal-shader bundle)" >&2
  exit 2
fi
for b in "${BUNDLES[@]}"; do
  cp -R "$b" "$DIST_DIR/"
done

cp -R "$MODEL_DIR" "$DIST_DIR/models/$MODEL_NAME"
cp configs/cgos.example.json "$DIST_DIR/configs/"
cp Scripts/cgos/*.py "$DIST_DIR/Scripts/cgos/"

echo "== checking the packaged Metal resource bundle resolves next to the binary (not via .build) ==" >&2
DOCTOR_JSON=$("$DIST_DIR/ichigo" doctor)
METAL_AVAILABLE=$(printf '%s' "$DOCTOR_JSON" | python3 -c "import json,sys; print('true' if json.load(sys.stdin)['metal']['available'] else 'false')")

if [ "$METAL_AVAILABLE" = "true" ]; then
  TMP_POS=$(mktemp)
  TMP_ERR=$(mktemp)
  python3 - "$MODEL_DIR/manifest.json" "$TMP_POS" <<'PY'
import json, sys
manifest_path, out_path = sys.argv[1], sys.argv[2]
m = json.load(open(manifest_path))
S = m["boardSizes"][0]
pos = {
    "schemaVersion": 1, "boardSize": S,
    "spatial": [0] * (S * S * 32),
    "global": [0.0, 0.0, 0.0, 0.0],
    "legal": [1] * (S * S + 1),
}
json.dump(pos, open(out_path, "w"))
PY
  if "$DIST_DIR/ichigo" eval --model "$DIST_DIR/models/$MODEL_NAME" --position "$TMP_POS" --backend metal >/dev/null 2>"$TMP_ERR"; then
    echo "build.sh: OK -- metal backend resolved logic_byte.metal from the packaged dist/ layout (ran $DIST_DIR/ichigo, independent of .build)" >&2
    rm -f "$TMP_POS" "$TMP_ERR"
  else
    echo "build.sh: FATAL -- --backend metal failed against the packaged dist/ layout even though this host reports a Metal device available:" >&2
    cat "$TMP_ERR" >&2
    echo "build.sh: this means the resource-bundle lookup in Sources/LogicMetal (MetalBackend.kernelSource / MetalPackedBackend's equivalent) does not work once the binary is copied out of .build -- fix Bundle.module resolution there before shipping. Not treating this as an acceptable silent fallback." >&2
    rm -f "$TMP_POS" "$TMP_ERR"
    exit 3
  fi
else
  echo "build.sh: WARNING -- no Metal device on this build host (ichigo doctor: metal.available=false). The Metal resource bundle is still packaged (for portability to Metal-capable hosts), but --backend metal/metal-packed were NOT exercised here. '--backend auto' explicitly falls back to the cpu backend on hosts without a Metal device (Sources/ichigo/main.swift's makeBackend) -- this is the documented, explicit fallback the release still ships with, not a silent one." >&2
fi

echo "== ichigo inspect --model (payload hash) ==" >&2
INSPECT_JSON=$("$DIST_DIR/ichigo" inspect --model "$DIST_DIR/models/$MODEL_NAME")
PAYLOAD_HASH=$(printf '%s' "$INSPECT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['payloadHash'])")
echo "build.sh: model payload hash $PAYLOAD_HASH" >&2

SWIFT_VERSION=$(swift --version 2>&1 | head -1)
OS_PRODUCT_VERSION=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
OS_BUILD_VERSION=$(sw_vers -buildVersion 2>/dev/null || echo "unknown")
UV_VERSION=$(uv --version 2>/dev/null || echo "not found")
PYTHON_VERSION=$(python3 --version 2>&1)

echo "== writing MANIFEST.json ==" >&2
python3 - "$DIST_DIR" "$PAYLOAD_HASH" "$MODEL_NAME" "$GIT_COMMIT" "$GIT_DIRTY" "$VERSION" "$ARCH" \
  "$SWIFT_VERSION" "$OS_PRODUCT_VERSION" "$OS_BUILD_VERSION" "$UV_VERSION" "$PYTHON_VERSION" <<'PY'
import hashlib, json, os, sys, datetime

(dist_dir, payload_hash, model_name, git_commit, git_dirty, version, arch,
 swift_version, os_product_version, os_build_version, uv_version, python_version) = sys.argv[1:13]

files = {}
for root, dirs, names in os.walk(dist_dir):
    dirs.sort()
    for name in sorted(names):
        path = os.path.join(root, name)
        rel = os.path.relpath(path, dist_dir)
        if rel == "MANIFEST.json":
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files[rel] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(path)}

manifest = {
    "schemaVersion": 1,
    "version": version,
    "arch": arch,
    "buildTimestampUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "git": {"commit": git_commit, "dirty": git_dirty == "true"},
    "toolchain": {"swift": swift_version, "uv": uv_version, "python3": python_version},
    "os": {"productVersion": os_product_version, "buildVersion": os_build_version},
    "model": {"name": model_name, "path": "models/" + model_name, "payloadHash": payload_hash},
    "files": files,
}
with open(os.path.join(dist_dir, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"build.sh: MANIFEST.json written -- {len(files)} files", file=sys.stderr)
PY

echo "== build.sh done ==" >&2
echo "$DIST_DIR"
