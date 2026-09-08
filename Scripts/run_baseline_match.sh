#!/usr/bin/env bash
# T31 M1 baseline gate (docs/spec/05-validation.md SS6 "機能完走と棋力評価"):
#   "9路hardモデルで...合法手uniform baselineと100局（50色交換ペア）、勝1/引分.5で勝率の95%下限>0.5"
#
#   Scripts/run_baseline_match.sh MODEL.ichigo [OUT_DIR]
#     MODEL    path to a .ichigo model directory supporting board size 9 (required)
#     OUT_DIR  default reports/matches/<model-basename>
#
# Run from the repository root, like Scripts/train_pilot.sh. Builds .build/release/ichigo if
# missing, then runs 100 games (50 colour-swapped pairs) of
#   .build/release/ichigo gtp --model-9 MODEL --visits 100
# vs the built-in uniform-legal baseline (ichigo_train match --engine-b uniform) on 9x9 komi 7,
# empty-board openings (no fixed opening book is required for the M1 gate; see docs/spec/
# 05-validation.md SS6 -- the 400-game/200-pair official comparison in SS5 is a separate, later
# ticket). SEED is fixed (overridable) so the run and its paired-bootstrap CI are reproducible.
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
MODEL=${1:?"usage: Scripts/run_baseline_match.sh MODEL.ichigo [OUT_DIR]"}
BASENAME=$(basename "$MODEL")
BASENAME=${BASENAME%.ichigo}
OUT_DIR=${2:-reports/matches/$BASENAME}
SEED=${SEED:-20260908}
VISITS=${VISITS:-100}
GAMES=${GAMES:-100}

[ -f "$MODEL"/manifest.json ] || { echo "model not found or not a .ichigo directory: $MODEL" >&2; exit 2; }
if [ ! -x .build/release/ichigo ]; then
  echo "== building .build/release/ichigo ==" >&2
  swift build -c release --product ichigo
fi

echo "== T31 baseline match: $GAMES games ($((GAMES / 2)) colour-swapped pairs), model=$MODEL visits=$VISITS seed=$SEED ==" >&2
"$UV" run --project Training python -m ichigo_train match \
  --engine-a ".build/release/ichigo gtp --model-9 $MODEL --visits $VISITS" \
  --engine-b uniform \
  --games "$GAMES" \
  --size 9 \
  --komi 7 \
  --openings none \
  --out "$OUT_DIR" \
  --seed "$SEED"

echo "== report: $OUT_DIR/report.json ==" >&2
"$UV" run --project Training python -c "
import json
r = json.load(open('$OUT_DIR/report.json'))
low = r['pairedCI95']['low']
print(f\"games={r['games']} counted={r['countedGames']} wins={r['wins']} draws={r['draws']} losses={r['losses']} meanScore={r['meanScore']}\")
print(f\"pairedCI95 low={low} high={r['pairedCI95']['high']}\")
print(f\"incidents={r['incidents']}\")
gate = low is not None and low > 0.5
print('M1 baseline gate (paired 95% CI lower bound > 0.5):', 'PASS' if gate else 'FAIL')
raise SystemExit(0 if gate else 1)
"
