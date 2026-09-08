#!/usr/bin/env python3
"""Launch/monitor the T33 single-server experiment matrix (docs/spec/04-tasks.md T33): one
single-GPU ``python -m ichigo_train train`` process per row of a
``configs/experiments/matrix.json``-shaped file, one process per GPU on one 10-GPU server.

Stdlib only (no torch/numpy import) so this script runs with the system Python, independent of
the Training/ uv environment it launches into.

Commands
--------
``launch --matrix configs/experiments/matrix.json [--force] [--state runs/experiments-state.json]``
    For each matrix entry not marked ``"skip": true`` and whose config's ``out`` directory does not
    already hold ``run-summary.json`` (unless ``--force``), start::

        CUDA_VISIBLE_DEVICES=<gpu> nohup uv run --project Training python -m ichigo_train train \\
            --config <config>

    as a detached background process with cwd = the repository root (so the config's relative
    paths, e.g. ``"data": "data/dataset-9"``, resolve the same way config.load_config would running
    interactively from the repo root), stdout+stderr redirected to
    ``runs/<runId>.stdout.log``, and records pid/gpu/config/runId/out/log/start time in
    ``runs/experiments-state.json``. An entry whose recorded pid is still alive is also skipped
    (again unless ``--force``), to avoid double-starting a run onto the same GPU.
    ``"skip": true`` entries (e.g. one already started by hand outside this tool) are ALWAYS
    skipped by ``launch``, even with ``--force`` -- that flag only overrides the "already
    completed" / "already running" checks, never an explicit ``skip`` in the matrix file.

``status --matrix configs/experiments/matrix.json [--state ...]``
    One line per matrix entry: alive (``os.kill(pid, 0)`` against the recorded pid), last
    completed training step and seconds/optimizer-step (both from metrics.csv's ``phase=="train"``
    rows -- see ``seconds_per_step``), ETA (remaining steps * seconds/step), the latest hard
    validation policyTop1/expectedMAE/scoreMAEPoints (the highest-step
    ``runs/<out>/validation/step-*.json``), and a state column that flags ``CRASHED`` (recorded as
    started, no longer alive, but ``run-summary.json`` was never written).

``relaunch --matrix configs/experiments/matrix.json --run-id <runId> [--state ...]``
    Restart exactly one entry (by ``runId``), refusing if its last recorded pid is still alive.
    Adds ``--resume <out>/checkpoint-latest.pt`` when that file exists.

matrix.json shape::

    {"schemaVersion": 1, "entries": [
        {"runId": "small-9-v2", "gpu": 1, "config": "configs/experiments/small-9-v2.json"},
        {"runId": "small-9-v1", "gpu": 0, "config": "configs/experiments/small-9-v1.json", "skip": true},
        ...
    ]}

``config`` and ``out`` (read from the config JSON) are resolved relative to the repository root,
matching how ``python -m ichigo_train train`` (run from the repo root, per this script's
``launch``) resolves them itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MATRIX = "configs/experiments/matrix.json"
DEFAULT_STATE = "runs/experiments-state.json"
VALIDATION_STEP_RE = re.compile(r"^step-(\d+)(?:-.*)?\.json$")
SECONDS_PER_STEP_WINDOW = 5  # average over up to this many recent train-row intervals


# ---------------------------------------------------------------------------
# small stdlib-only helpers (kept pure / side-effect-free for testability)
# ---------------------------------------------------------------------------

def resolve_path(path: str, base: str = REPO_ROOT) -> str:
    """Relative paths resolve against ``base`` (the repository root by default) -- NOT the
    current working directory -- so this tool behaves the same regardless of where it is invoked
    from."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_matrix(path: str) -> dict:
    matrix = read_json(path)
    if "entries" not in matrix:
        raise ValueError(f"{path}: missing 'entries'")
    seen = set()
    for e in matrix["entries"]:
        for key in ("runId", "gpu", "config"):
            if key not in e:
                raise ValueError(f"{path}: entry {e} missing required key {key!r}")
        if e["runId"] in seen:
            raise ValueError(f"{path}: duplicate runId {e['runId']!r}")
        seen.add(e["runId"])
    return matrix


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"schemaVersion": 1, "entries": {}}
    state = read_json(path)
    state.setdefault("entries", {})
    return state


def pid_alive(pid) -> bool | None:
    """True/False if we can tell, None if there is no pid to check (never launched by this
    tool -- e.g. a "skip": true entry someone started by hand)."""
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else; we can't signal it but it is alive.
        return True
    except OSError:
        return False
    return True


def read_train_rows(metrics_csv_path: str) -> list[dict]:
    """``[{"step": int, "elapsedSeconds": float}, ...]`` for metrics.csv's ``phase=="train"``
    rows (docs/spec/02-training.md §10 run artifacts; metrics.py's ``CSV_FIELDS``/``MetricsCSV``),
    sorted by step. Missing file or unparseable rows are simply skipped, never an error -- this
    runs against a live, growing file that may be mid-write."""
    rows: list[dict] = []
    if not os.path.exists(metrics_csv_path):
        return rows
    with open(metrics_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "train":
                continue
            try:
                rows.append({"step": int(row["step"]), "elapsedSeconds": float(row["elapsedSeconds"])})
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r["step"])
    return rows


def seconds_per_step(train_rows: list[dict], window: int = SECONDS_PER_STEP_WINDOW) -> float | None:
    """Average seconds/optimizer-step over the last ``window`` elapsedSeconds deltas between
    consecutive train rows (train.py logs a train row every 10 steps, so this is typically an
    average over the last ~10*window steps). None if there are fewer than 2 train rows, or the
    step count did not advance (duplicate/out-of-order rows)."""
    if len(train_rows) < 2:
        return None
    recent = train_rows[-(window + 1):]
    dstep = recent[-1]["step"] - recent[0]["step"]
    if dstep <= 0:
        return None
    return (recent[-1]["elapsedSeconds"] - recent[0]["elapsedSeconds"]) / dstep


def latest_validation_record(validation_dir: str) -> dict | None:
    """The parsed content of the highest-step ``step-*.json`` under ``validation_dir``
    (train.py's ``validate()``); ties (e.g. a plain step and a same-step ``-freeze-N`` tag) prefer
    the untagged file. None if the directory does not exist or holds no validation files."""
    if not os.path.isdir(validation_dir):
        return None
    best_key = None
    best_name = None
    for name in os.listdir(validation_dir):
        m = VALIDATION_STEP_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        untagged = 1 if name == f"step-{step:07d}.json" else 0
        key = (step, untagged)
        if best_key is None or key > best_key:
            best_key, best_name = key, name
    if best_name is None:
        return None
    try:
        return read_json(os.path.join(validation_dir, best_name))
    except (OSError, ValueError):
        return None


def build_status(entry: dict, state: dict, repo_root: str = REPO_ROOT) -> dict:
    """Per-entry status dict used by both ``status`` printing and tests. Never raises for a
    missing/partial run directory (a not-yet-started or just-started entry) -- only for an
    unreadable config, which is reported via the ``"error"`` key."""
    run_id = entry["runId"]
    result: dict = {"runId": run_id, "gpu": entry.get("gpu"), "skip": bool(entry.get("skip"))}
    try:
        cfg = read_json(resolve_path(entry["config"], repo_root))
    except (OSError, ValueError) as e:
        result["error"] = f"cannot read config {entry['config']}: {e}"
        return result
    out_dir = resolve_path(cfg["out"], repo_root)
    result["out"] = os.path.relpath(out_dir, repo_root)
    max_steps = cfg.get("maxSteps")

    tracked = state.get("entries", {}).get(run_id)
    pid = tracked.get("pid") if tracked else None
    alive = pid_alive(pid)
    result["pid"] = pid
    result["alive"] = alive

    completed = os.path.exists(os.path.join(out_dir, "run-summary.json"))
    result["completed"] = completed
    result["crashed"] = tracked is not None and alive is False and not completed

    rows = read_train_rows(os.path.join(out_dir, "metrics.csv"))
    last_step = rows[-1]["step"] if rows else None
    sps = seconds_per_step(rows)
    result["lastStep"] = last_step
    result["secondsPerStep"] = sps
    if sps is not None and max_steps is not None and last_step is not None:
        result["etaSeconds"] = max(0.0, (max_steps - last_step) * sps)
    else:
        result["etaSeconds"] = None

    val = latest_validation_record(os.path.join(out_dir, "validation"))
    hard = val.get("hard") if val else None
    result["validationStep"] = val.get("step") if val else None
    result["policyTop1"] = hard.get("policyTop1") if hard else None
    result["expectedMAE"] = hard.get("expectedMAE") if hard else None
    result["scoreMAEPoints"] = hard.get("scoreMAEPoints") if hard else None
    return result


def format_status_row(s: dict) -> str:
    if "error" in s:
        return f"{s['runId']:<26} ERROR {s['error']}"
    if s["skip"]:
        state_note = "skip"
    elif s["crashed"]:
        state_note = "CRASHED"
    elif s["completed"]:
        state_note = "done"
    elif s["alive"] is True:
        state_note = "alive"
    elif s["alive"] is False:
        state_note = "dead"
    else:
        state_note = "?"

    def fmt(v, spec="{:.3f}"):
        return spec.format(v) if v is not None else "-"

    step = s["lastStep"] if s["lastStep"] is not None else "-"
    eta = f"{s['etaSeconds'] / 60:.0f}m" if s["etaSeconds"] is not None else "-"
    return (f"{s['runId']:<26} gpu={s['gpu']!s:<3} {state_note:<9} step={step!s:<8} "
            f"sec/step={fmt(s['secondsPerStep'], '{:.2f}'):<7} eta={eta:<7} "
            f"top1={fmt(s['policyTop1']):<6} MAE={fmt(s['expectedMAE']):<6} "
            f"scoreMAE={fmt(s['scoreMAEPoints'], '{:.2f}'):<6} out={s['out']}")


# ---------------------------------------------------------------------------
# process launching (impure; not exercised by tests)
# ---------------------------------------------------------------------------

def _spawn(run_id: str, gpu: int, config_path: str, resume: str | None = None) -> dict:
    runs_dir = os.path.join(REPO_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, f"{run_id}.stdout.log")
    cfg_rel = os.path.relpath(config_path, REPO_ROOT)
    cmd = ["nohup", "uv", "run", "--project", "Training", "python", "-m", "ichigo_train", "train", "--config", cfg_rel]
    if resume:
        cmd += ["--resume", os.path.relpath(resume, REPO_ROOT)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(log_path, "ab") as log_fh:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    return {"pid": proc.pid, "stdout": os.path.relpath(log_path, REPO_ROOT),
            "startTime": datetime.now(timezone.utc).isoformat()}


def _record_launch(state: dict, state_path: str, run_id: str, entry: dict, config_path: str, out_dir: str, spawned: dict) -> None:
    state.setdefault("entries", {})[run_id] = {
        "runId": run_id, "pid": spawned["pid"], "gpu": entry["gpu"],
        "config": os.path.relpath(config_path, REPO_ROOT), "out": os.path.relpath(out_dir, REPO_ROOT),
        "stdout": spawned["stdout"], "startTime": spawned["startTime"],
    }
    write_json_atomic(state_path, state)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_launch(args) -> int:
    matrix = load_matrix(resolve_path(args.matrix))
    state_path = resolve_path(args.state)
    state = load_state(state_path)
    launched = 0
    for entry in matrix["entries"]:
        run_id = entry["runId"]
        if entry.get("skip"):
            print(f"[skip] {run_id}: matrix entry marked skip (never started by launch, even with --force)")
            continue
        config_path = resolve_path(entry["config"])
        cfg = read_json(config_path)
        out_dir = resolve_path(cfg["out"])
        if os.path.exists(os.path.join(out_dir, "run-summary.json")) and not args.force:
            print(f"[skip] {run_id}: {out_dir} already has run-summary.json (use --force to relaunch)")
            continue
        tracked = state["entries"].get(run_id)
        if tracked and pid_alive(tracked.get("pid")) and not args.force:
            print(f"[skip] {run_id}: pid {tracked['pid']} still alive (use --force to start a duplicate)")
            continue
        spawned = _spawn(run_id, entry["gpu"], config_path)
        _record_launch(state, state_path, run_id, entry, config_path, out_dir, spawned)
        print(f"[launch] {run_id}: gpu={entry['gpu']} pid={spawned['pid']} log={spawned['stdout']}")
        launched += 1
    print(f"launched {launched} entr{'y' if launched == 1 else 'ies'}; state -> {os.path.relpath(state_path, REPO_ROOT)}")
    return 0


def cmd_status(args) -> int:
    matrix = load_matrix(resolve_path(args.matrix))
    state = load_state(resolve_path(args.state))
    for entry in matrix["entries"]:
        print(format_status_row(build_status(entry, state)))
    return 0


def cmd_relaunch(args) -> int:
    matrix = load_matrix(resolve_path(args.matrix))
    state_path = resolve_path(args.state)
    state = load_state(state_path)
    entry = next((e for e in matrix["entries"] if e["runId"] == args.run_id), None)
    if entry is None:
        print(f"error: no matrix entry with runId {args.run_id!r}", file=sys.stderr)
        return 2
    tracked = state["entries"].get(args.run_id)
    if tracked and pid_alive(tracked.get("pid")):
        print(f"error: {args.run_id} pid {tracked['pid']} still alive; kill it before relaunching", file=sys.stderr)
        return 1
    config_path = resolve_path(entry["config"])
    cfg = read_json(config_path)
    out_dir = resolve_path(cfg["out"])
    resume_path = os.path.join(out_dir, "checkpoint-latest.pt")
    resume = resume_path if os.path.exists(resume_path) else None
    spawned = _spawn(args.run_id, entry["gpu"], config_path, resume=resume)
    _record_launch(state, state_path, args.run_id, entry, config_path, out_dir, spawned)
    print(f"[relaunch] {args.run_id}: gpu={entry['gpu']} pid={spawned['pid']} resume={resume or '(none)'} log={spawned['stdout']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="launch_experiments.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--matrix", default=DEFAULT_MATRIX, help=f"matrix JSON path, relative to the repo root if not absolute (default {DEFAULT_MATRIX})")
        sp.add_argument("--state", default=DEFAULT_STATE, help=f"experiments-state.json path (default {DEFAULT_STATE})")

    l = sub.add_parser("launch", help="Start every non-skip, not-yet-run matrix entry.")
    add_common(l)
    l.add_argument("--force", action="store_true", help="also (re)launch entries with an existing run-summary.json or a still-alive tracked pid (never overrides an entry's own \"skip\": true)")
    l.set_defaults(func=cmd_launch)

    s = sub.add_parser("status", help="Print one status line per matrix entry.")
    add_common(s)
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("relaunch", help="Restart one matrix entry, resuming from its checkpoint-latest.pt if present.")
    add_common(r)
    r.add_argument("--run-id", required=True)
    r.set_defaults(func=cmd_relaunch)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
