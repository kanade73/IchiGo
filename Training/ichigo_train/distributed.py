"""Single-server multi-GPU DDP process-group management (docs/spec/02-training.md §7/§10, T26).

Detects torchrun's environment variables (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``). When they are
absent the process runs standalone: rank 0 of world size 1, no process group is initialised, and
every collective in this module degenerates to a local no-op (``all_reduce_sum``/``barrier`` are
pass-throughs, ``broadcast_object`` returns its argument unchanged). When present, :func:`init`
creates a NCCL process group (``device=="cuda"``) or a gloo one (``device=="cpu"``) with an
explicit timeout, so a stuck or crashed peer does not hang the others forever.

Rank-failure contract: torchrun's elastic agent supervises every worker it launches and sends
SIGTERM to the rest of the group as soon as one exits non-zero, so the *only* thing this module
and the training loop need to guarantee is that a rank which hits an error actually terminates
with a non-zero exit code instead of hanging or silently continuing. ``train.py`` wraps the whole
run loop in try/except: on any exception it logs, calls :func:`cleanup`, and re-raises so the
process exits non-zero and torchrun tears down the remaining ranks. A raised exception inside a
NCCL collective (e.g. because a peer died) will otherwise hang until the collective's timeout
fires; keep that timeout well under the shell/job-level watchdog if this is orchestrated further.

Only rank 0 (``is_main()``) may write run artifacts: metrics.csv, training.log, validation/*.json,
checkpoints and run-summary.json. Non-zero ranks must not open those files for writing. Use
``barrier()`` at the same points on every rank around validation and checkpointing so the ranks
that are not writing wait for rank 0 instead of racing ahead onto the next optimizer step (or, for
validation, computing it redundantly) — see the "4GPU検収の具体条件" note in docs/spec/02-training.md
about why rank0-only validation forward + broadcast is required instead of every rank evaluating.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist

DEFAULT_TIMEOUT = datetime.timedelta(minutes=30)

_STATE = {"initialized": False, "rank": 0, "world_size": 1, "local_rank": 0, "backend": None}


def detect() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank) from torchrun's env vars, or (0, 1, 0) if absent."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, 0


def init(device_kind: str, timeout: datetime.timedelta = DEFAULT_TIMEOUT) -> None:
    """Initialises the process group if torchrun's env vars are present and WORLD_SIZE > 1.

    ``device_kind`` is the resolved config "device" ("cpu" or "cuda") and picks the backend:
    NCCL for cuda, gloo for cpu. Safe to call from a single (non-torchrun) process: it just
    records rank=0/world_size=1 and never touches ``torch.distributed``.
    """
    rank, world_size, local_rank = detect()
    _STATE["rank"], _STATE["world_size"], _STATE["local_rank"] = rank, world_size, local_rank
    if world_size <= 1:
        _STATE["initialized"] = False
        _STATE["backend"] = None
        return
    backend = "nccl" if device_kind == "cuda" else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)
    _STATE["initialized"] = True
    _STATE["backend"] = backend


def is_initialized() -> bool:
    return _STATE["initialized"]


def rank() -> int:
    return _STATE["rank"]


def world_size() -> int:
    return _STATE["world_size"]


def local_rank() -> int:
    return _STATE["local_rank"]


def is_main() -> bool:
    return _STATE["rank"] == 0


def resolve_device(device_kind: str) -> torch.device:
    """cuda -> this rank's local GPU (``cuda:LOCAL_RANK``); cpu -> cpu."""
    if device_kind == "cuda":
        return torch.device("cuda", _STATE["local_rank"])
    return torch.device("cpu")


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sums ``tensor`` across all ranks in place and returns it (no-op if not initialised)."""
    if _STATE["initialized"]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcasts a picklable Python object from rank ``src`` to every rank; returns it. When the
    process group is not initialised, returns ``obj`` unchanged (the caller is rank 0=only rank)."""
    if not _STATE["initialized"]:
        return obj
    box = [obj if _STATE["rank"] == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def gather_to_main(obj, dst: int = 0):
    """Gathers one picklable Python object per rank to rank ``dst``. Returns the list there
    (index == rank) and ``None`` on every other rank; not initialised -> returns ``[obj]`` (the
    caller is the only rank). Used for per-rank throughput/device/peak-memory reporting in
    run-summary.json -- not one of the five collectives T26 named explicitly, but "peak GPU memory
    per rank (gathered to rank 0)" needs a gather, and this is the natural extension of
    broadcast_object's pattern (dist.gather_object works for both the gloo and nccl backends)."""
    if not _STATE["initialized"]:
        return [obj]
    out = [None] * _STATE["world_size"] if _STATE["rank"] == dst else None
    dist.gather_object(obj, out, dst=dst)
    return out


def barrier() -> None:
    if _STATE["initialized"]:
        dist.barrier()


def cleanup() -> None:
    if _STATE["initialized"]:
        dist.destroy_process_group()
    _STATE["initialized"] = False
    _STATE["backend"] = None
