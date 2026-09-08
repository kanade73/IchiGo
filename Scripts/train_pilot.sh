#!/usr/bin/env bash
# T17 pilot sequence (docs/spec/02-training.md §7, §11.3). Run from the repository root.
#   1. CPU tiny overfit acceptance (16 positions)          -> runs/overfit-tiny-9
#   2. 1-GPU small 100-step timing pilot                   -> runs/small-9-pilot100
#   3. 1-GPU small 2000-step pilot on the labelled corpus  -> runs/small-9-pilot2000
#   4. export best-hard checkpoint and run Swift parity
# Steps 2-3 need CUDA (A4000); on a Mac without CUDA they are skipped and reported as not run.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
REPORTS=reports/pilot
mkdir -p "$REPORTS"

# Every ichigo_train invocation runs with cwd = repository root (via `uv run --project Training`,
# not `cd Training`), because config files use repo-root-relative paths (e.g. "data":"data/..."),
# which load_config() resolves against the current process's cwd. Running with cwd=Training/ would
# silently resolve those to Training/data/... instead.
run_py() { "$UV" run --project Training python -m ichigo_train "$@"; }

echo "== 1. CPU tiny overfit =="
[ -f data/overfit-9/fixture.npz ] || { echo "missing data/overfit-9/fixture.npz (Scripts/make_overfit_fixture.py)"; exit 3; }
run_py train --config configs/train-tiny.json
run_py evaluate --checkpoint runs/overfit-tiny-9/checkpoint-best-hard.pt --data data/overfit-9 --mode hard --out "$REPORTS/overfit-tiny-hard.json"
run_py evaluate --checkpoint runs/overfit-tiny-9/checkpoint-best-hard.pt --data data/overfit-9 --mode soft --out "$REPORTS/overfit-tiny-soft.json"
"$UV" run --project Training python Scripts/check_overfit_gate.py --run runs/overfit-tiny-9 --hard "$REPORTS/overfit-tiny-hard.json" --out "$REPORTS/overfit-gate.json"

CUDA=$("$UV" run --project Training python -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA" != "1" ]; then
  echo "== CUDA not available: GPU pilots (100 step / 2000 step) NOT RUN ==" | tee "$REPORTS/gpu-pilot-not-run.txt"
  exit 0
fi
echo "== 2. 1-GPU small 100-step pilot =="
run_py doctor --out "$REPORTS/cuda-doctor.json"
run_py train --config configs/train-9-pilot100.json
echo "== 3. 1-GPU small 2000-step pilot =="
run_py train --config configs/train-9-pilot2000.json
echo "== 4. export + Swift parity =="
run_py export --checkpoint runs/small-9-pilot2000/checkpoint-best-hard.pt --out models/small-9-pilot2000.ichigo --board-sizes 9 --overwrite
"$UV" run --project Training python Scripts/check_model_parity.py --model models/small-9-pilot2000.ichigo --data data/dataset-9 --split validation --count 8
