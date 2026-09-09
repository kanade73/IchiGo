#!/usr/bin/env bash
# T26 single-server multi-GPU DDP training wrapper (docs/spec/02-training.md §7/§10
# "4GPU検収の具体条件"; Training/ichigo_train/distributed.py, train.py).
#
#   Scripts/train_ddp.sh [N] [CONFIG] [--resume PATH]
#     N       number of GPUs/processes on this single server (default 4)
#     CONFIG  training config path, repo-root-relative (default configs/train-ddp.json)
#
# Run from the repository root, like train_pilot.sh: config "data"/"out" paths are repo-root
# relative and load_config() resolves them against the current process's cwd, so this uses
# `uv run --project Training torchrun ...` rather than `cd Training && uv run torchrun ...` --
# torchrun's spawned worker processes inherit ITS cwd, so that cwd must stay at the repo root too
# (see Scripts/train_pilot.sh's header comment for the same reasoning / the bug this avoids).
#
# Without torchrun's RANK/WORLD_SIZE/LOCAL_RANK env vars, ichigo_train.train runs as a single
# process (world_size=1, no process group); torchrun always sets them (even for N=1), so this
# still exercises distributed.py's init path.
set -euo pipefail
# NCCL peer-to-peer hangs on the university server (PXB topology, verified 2026-09-09 with a 2-rank
# all_reduce that only completes with P2P disabled). Override by exporting the variables yourself.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
cd "$(dirname "$0")/.."
UV=${UV:-uv}
N=${1:-4}
CONFIG=${2:-configs/train-ddp.json}
[ "$#" -ge 1 ] && shift
[ "$#" -ge 1 ] && shift

echo "== T26 DDP training: N=$N processes, config=$CONFIG =="
exec "$UV" run --project Training torchrun --standalone --nproc_per_node="$N" -m ichigo_train train --config "$CONFIG" "$@"
