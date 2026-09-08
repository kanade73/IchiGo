"""Scripts/launch_experiments.py's status-parsing logic (docs/spec/04-tasks.md T33), exercised on
synthetic run directories -- no real process is ever launched. The module lives outside the
Training package (it is stdlib-only, deliberately independent of the Training uv environment) so
it is imported here via its file path."""

import csv
import importlib.util
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_PATH = os.path.join(REPO_ROOT, "Scripts", "launch_experiments.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("launch_experiments", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def LE():
    return _load_module()


def write_metrics_csv(path, train_rows):
    fields = ["step", "phase", "mode", "elapsedSeconds"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for step, elapsed in train_rows:
            w.writerow({"step": step, "phase": "train", "mode": "soft", "elapsedSeconds": elapsed})
        # a validation row should never be mistaken for a train row
        w.writerow({"step": train_rows[-1][0] if train_rows else 0, "phase": "validation", "mode": "hard", "elapsedSeconds": 999})


def write_validation(dirpath, step, tag, policy_top1, expected_mae, score_mae):
    os.makedirs(dirpath, exist_ok=True)
    name = f"step-{step:07d}{('-' + tag) if tag else ''}.json"
    with open(os.path.join(dirpath, name), "w") as f:
        json.dump({"step": step, "hard": {"policyTop1": policy_top1, "expectedMAE": expected_mae, "scoreMAEPoints": score_mae},
                   "soft": {"policyTop1": policy_top1 - 0.05, "expectedMAE": expected_mae + 0.05, "scoreMAEPoints": score_mae + 1}}, f)


def make_run(tmp_path, run_id, train_rows=(), validations=(), run_summary=False):
    out_dir = tmp_path / "runs" / run_id
    out_dir.mkdir(parents=True)
    if train_rows:
        write_metrics_csv(out_dir / "metrics.csv", train_rows)
    val_dir = out_dir / "validation"
    for step, tag, top1, mae, smae in validations:
        write_validation(val_dir, step, tag, top1, mae, smae)
    if run_summary:
        with open(out_dir / "run-summary.json", "w") as f:
            json.dump({"runId": run_id, "completed": True}, f)
    cfg_path = tmp_path / f"{run_id}.json"
    with open(cfg_path, "w") as f:
        json.dump({"runId": run_id, "out": str(out_dir), "maxSteps": 20000}, f)
    return str(cfg_path), str(out_dir)


def entry_for(run_id, gpu, cfg_path, skip=False):
    e = {"runId": run_id, "gpu": gpu, "config": cfg_path}
    if skip:
        e["skip"] = True
    return e


# ---- read_train_rows / seconds_per_step ----

def test_read_train_rows_filters_phase_and_sorts(LE, tmp_path):
    _, out_dir = make_run(tmp_path, "r1", train_rows=[(10, 10.0), (30, 30.0), (20, 20.0)])
    rows = LE.read_train_rows(os.path.join(out_dir, "metrics.csv"))
    assert [r["step"] for r in rows] == [10, 20, 30]


def test_read_train_rows_missing_file_returns_empty(LE, tmp_path):
    assert LE.read_train_rows(str(tmp_path / "nope.csv")) == []


def test_seconds_per_step(LE):
    rows = [{"step": s, "elapsedSeconds": float(s) * 2.0} for s in range(0, 110, 10)]
    assert LE.seconds_per_step(rows) == pytest.approx(2.0)
    assert LE.seconds_per_step([{"step": 0, "elapsedSeconds": 0.0}]) is None
    assert LE.seconds_per_step([]) is None


# ---- latest_validation_record ----

def test_latest_validation_record_picks_highest_step_prefers_untagged(LE, tmp_path):
    val_dir = tmp_path / "val"
    write_validation(val_dir, 500, "", 0.5, 0.3, 5.0)
    write_validation(val_dir, 1000, "freeze-1", 0.6, 0.25, 4.5)
    write_validation(val_dir, 1000, "", 0.61, 0.24, 4.4)
    rec = LE.latest_validation_record(str(val_dir))
    assert rec["step"] == 1000 and rec["hard"]["policyTop1"] == 0.61


def test_latest_validation_record_missing_dir(LE, tmp_path):
    assert LE.latest_validation_record(str(tmp_path / "nope")) is None


# ---- build_status / format_status_row on a synthetic run dir ----

def test_build_status_running_entry(LE, tmp_path):
    cfg_path, out_dir = make_run(
        tmp_path, "running-run",
        train_rows=[(i, float(i) * 0.5) for i in range(0, 110, 10)],
        validations=[(100, "", 0.42, 0.31, 5.2)],
    )
    entry = entry_for("running-run", 3, cfg_path)
    state = {"entries": {"running-run": {"pid": os.getpid(), "gpu": 3}}}
    s = LE.build_status(entry, state, repo_root=str(tmp_path))
    assert s["alive"] is True
    assert s["completed"] is False
    assert s["crashed"] is False
    assert s["lastStep"] == 100
    assert s["secondsPerStep"] == pytest.approx(0.5)
    assert s["etaSeconds"] == pytest.approx((20000 - 100) * 0.5)
    assert s["policyTop1"] == pytest.approx(0.42)
    assert s["expectedMAE"] == pytest.approx(0.31)
    assert s["scoreMAEPoints"] == pytest.approx(5.2)
    line = LE.format_status_row(s)
    assert "running-run" in line and "alive" in line


def test_build_status_crashed_entry(LE, tmp_path):
    """A dead, never-completed pid (a pid nothing currently holds) is flagged crashed."""
    dead_pid = 2**30  # extremely unlikely to be a live pid
    cfg_path, out_dir = make_run(tmp_path, "crashed-run", train_rows=[(0, 0.0), (10, 5.0)])
    entry = entry_for("crashed-run", 4, cfg_path)
    state = {"entries": {"crashed-run": {"pid": dead_pid, "gpu": 4}}}
    s = LE.build_status(entry, state, repo_root=str(tmp_path))
    assert s["alive"] is False
    assert s["crashed"] is True
    line = LE.format_status_row(s)
    assert "CRASHED" in line


def test_build_status_completed_entry_not_crashed_even_if_dead(LE, tmp_path):
    dead_pid = 2**30
    cfg_path, out_dir = make_run(tmp_path, "done-run", train_rows=[(20000, 9000.0)], run_summary=True)
    entry = entry_for("done-run", 5, cfg_path)
    state = {"entries": {"done-run": {"pid": dead_pid, "gpu": 5}}}
    s = LE.build_status(entry, state, repo_root=str(tmp_path))
    assert s["completed"] is True
    assert s["crashed"] is False
    line = LE.format_status_row(s)
    assert "done" in line and "CRASHED" not in line


def test_build_status_never_launched_entry(LE, tmp_path):
    cfg_path, out_dir = make_run(tmp_path, "never-run")
    entry = entry_for("never-run", 6, cfg_path)
    state = {"entries": {}}
    s = LE.build_status(entry, state, repo_root=str(tmp_path))
    assert s["pid"] is None
    assert s["alive"] is None
    assert s["crashed"] is False
    assert s["lastStep"] is None
    line = LE.format_status_row(s)
    assert "?" in line.split()[2]  # state column


def test_build_status_skip_entry(LE, tmp_path):
    cfg_path, out_dir = make_run(tmp_path, "skip-run")
    entry = entry_for("skip-run", 0, cfg_path, skip=True)
    s = LE.build_status(entry, {"entries": {}}, repo_root=str(tmp_path))
    assert s["skip"] is True
    assert "skip" in LE.format_status_row(s)


def test_build_status_unreadable_config_reports_error(LE, tmp_path):
    entry = entry_for("bad-run", 7, str(tmp_path / "does-not-exist.json"))
    s = LE.build_status(entry, {"entries": {}}, repo_root=str(tmp_path))
    assert "error" in s
    assert "ERROR" in LE.format_status_row(s)


# ---- pid_alive ----

def test_pid_alive(LE):
    assert LE.pid_alive(os.getpid()) is True
    assert LE.pid_alive(2**30) is False
    assert LE.pid_alive(None) is None


# ---- matrix loading validation ----

def test_load_matrix_rejects_duplicate_run_ids(LE, tmp_path):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps({"entries": [
        {"runId": "a", "gpu": 0, "config": "a.json"},
        {"runId": "a", "gpu": 1, "config": "b.json"},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        LE.load_matrix(str(p))


def test_load_matrix_rejects_missing_keys(LE, tmp_path):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps({"entries": [{"runId": "a", "gpu": 0}]}))
    with pytest.raises(ValueError, match="config"):
        LE.load_matrix(str(p))
