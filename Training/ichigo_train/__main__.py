"""``python -m ichigo_train`` CLI (docs/spec/02-training.md §10). M0 provides doctor,
init-model, export, inspect, and fixture generation; train/label/build-data arrive in M1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

EXIT_OK, EXIT_CONFIG, EXIT_INPUT, EXIT_TRAINING = 0, 2, 3, 4


def cmd_doctor(args) -> int:
    import numpy
    import torch

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "cudaVersion": torch.version.cuda,
        "gpuCount": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "nvidiaSmi": None,
        "diskFreeBytes": shutil.disk_usage(os.getcwd()).free,
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            report["gpus"].append({"index": i, "name": p.name, "totalMemoryBytes": p.total_memory})
    if shutil.which("nvidia-smi"):
        try:
            report["nvidiaSmi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            report["nvidiaSmi"] = f"error: {e}"
    text = json.dumps(report, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return EXIT_OK


def cmd_init_model(args) -> int:
    from .export import save_checkpoint
    from .model import LogicNet
    from .wiring import ModelSpec

    try:
        model = LogicNet(ModelSpec.from_profile(args.profile, args.seed))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    save_checkpoint(model, args.out, {"boardSizes": args.board_sizes})
    print(json.dumps({"checkpoint": args.out, "profile": args.profile, "seed": args.seed}))
    return EXIT_OK


def cmd_export(args) -> int:
    from .export import export_model, load_checkpoint
    from .model_format import ModelFormatError

    try:
        try:
            model = load_checkpoint(args.checkpoint)
        except KeyError:
            from .checkpoint import load_checkpoint as load_train_ck, model_from_checkpoint
            model = model_from_checkpoint(load_train_ck(args.checkpoint))
    except (OSError, KeyError, ValueError) as e:
        print(f"error: cannot load checkpoint: {e}", file=sys.stderr)
        return EXIT_INPUT
    calibration = None
    if args.calibration:
        try:
            with open(args.calibration) as f:
                calibration = json.load(f)
        except (OSError, ValueError) as e:
            print(f"error: cannot read --calibration {args.calibration}: {e}", file=sys.stderr)
            return EXIT_INPUT
    try:
        manifest = export_model(model, args.out, args.board_sizes, {"checkpoint": os.path.basename(args.checkpoint)},
                                overwrite=args.overwrite, calibration=calibration)
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except ModelFormatError as e:
        print(f"error: export verification failed: {e}", file=sys.stderr)
        return EXIT_INPUT
    print(json.dumps({"out": args.out, "files": manifest["files"], "calibrationTemperature": manifest["calibrationTemperature"]}, indent=2))
    return EXIT_OK


def cmd_sgf_results(args) -> int:
    from .sgf_results import build_results

    if not os.path.isdir(args.sgf_dir) or not os.path.isfile(args.positions):
        print("error: --sgf-dir/--positions not found", file=sys.stderr)
        return EXIT_INPUT
    report = build_results(args.sgf_dir, args.positions, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["gamesWritten"] > 0 else EXIT_INPUT


def cmd_calibrate(args) -> int:
    from .calibrate import run_calibration

    if not os.path.isfile(args.results):
        print("error: --results not found", file=sys.stderr)
        return EXIT_INPUT
    report = run_calibration(args.checkpoint, args.data, args.results, args.out, resamples=args.resamples, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["verified"]:
        print("warning: no real-result validation positions found; temperature fit kept at T=1 (unverified)", file=sys.stderr)
    if report["test"]["insufficientSamples"]:
        print(f"warning: only {report['test']['games']} complete test games with results (< {report['test']['minCompleteGames']}); "
              "calibration sample is insufficient", file=sys.stderr)
    return EXIT_OK


def cmd_inspect(args) -> int:
    from .model_format import ModelFormatError, read_model

    try:
        m = read_model(args.model)
    except ModelFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    summary = {k: v for k, v in m.manifest.items() if k != "headTensors"}
    summary["gateHistogram"] = {str(g): int(c) for g, c in enumerate(__import__("numpy").bincount(m.gates.reshape(-1), minlength=16))}
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def cmd_make_fixtures(args) -> int:
    from .fixtures import write_parity_case
    from .symmetry import write_permutation_table

    os.makedirs(args.out, exist_ok=True)
    write_parity_case(os.path.join(args.out, "parity"), "tiny-9", 9, 4, 20260908)
    write_parity_case(os.path.join(args.out, "parity"), "tiny-19", 19, 2, 20260909)
    write_parity_case(os.path.join(args.out, "parity"), "tiny-9-headv1", 9, 2, 20260910, head_version=1)
    write_parity_case(os.path.join(args.out, "parity"), "tiny-9-headv3", 9, 3, 20260911, head_version=3)
    write_parity_case(os.path.join(args.out, "parity"), "tiny-19-headv3", 19, 2, 20260912, head_version=3)
    os.makedirs(os.path.join(args.out, "symmetry"), exist_ok=True)
    for s in (9, 19):
        write_permutation_table(s, os.path.join(args.out, "symmetry", f"perm-{s}.json"))
    print(f"fixtures written under {args.out}")
    return EXIT_OK


def _teacher_id(binary: str, model: str) -> str:
    import hashlib
    h = hashlib.sha256(open(model, "rb").read()).hexdigest()
    return f"katago:{os.path.basename(binary)}:{os.path.basename(model)}:{h[:16]}"


def cmd_label(args) -> int:
    import hashlib
    import time
    from .teacher import TeacherConfig, label_positions

    if not os.path.isfile(args.teacher_model) or not os.path.isfile(args.positions):
        print("error: positions/teacher model not found", file=sys.stderr)
        return EXIT_INPUT
    cmd = [args.teacher_bin, "analysis", "-config", args.teacher_config, "-model", args.teacher_model]
    cfg = TeacherConfig(command=cmd, teacher_id=_teacher_id(args.teacher_bin, args.teacher_model), perspective=args.perspective,
                        visits=args.visits, timeout_seconds=args.timeout, retries=2, queue_depth=args.queue_depth,
                        stderr_log=args.out + ".teacher-stderr.log")
    t0 = time.monotonic()

    def progress(n, stats):
        el = time.monotonic() - t0
        print(f"[label] positions read {n} labels {stats.labels} rejected {stats.rejected} elapsed {el:.0f}s rate {stats.labels / max(el, 1e-9):.1f}/s", file=sys.stderr, flush=True)

    stats = label_positions(args.positions, args.out, cfg, max_positions=args.max_positions, progress=progress)
    elapsed = time.monotonic() - t0
    summary = {"positions": args.positions, "out": args.out, "labels": stats.labels, "rejected": stats.rejected, "rejectReasons": stats.reject_reasons,
               "queries": stats.queries, "retries": stats.retries, "duplicates": stats.duplicates, "elapsedSeconds": round(elapsed, 1),
               "labelsPerSecond": round(stats.labels / max(elapsed, 1e-9), 2), "teacherId": cfg.teacher_id, "visits": args.visits,
               "perspective": args.perspective, "teacherConfigSha256": hashlib.sha256(open(args.teacher_config, "rb").read()).hexdigest()}
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return EXIT_OK if stats.labels > 0 else EXIT_INPUT


def cmd_build_data(args) -> int:
    from .dataset import build_dataset

    wdl = None
    if args.results:
        wdl = {}
        with open(args.results) as f:
            for line in f:
                r = json.loads(line)
                wdl[r["gameId"]] = r["result"]
    try:
        manifest = build_dataset(args.positions, args.labels, args.out, {"positions": os.path.basename(args.positions), "labels": os.path.basename(args.labels), "labelsSummary": _maybe_json(args.labels + ".summary.json")}, shard_size=args.shard_size, wdl_from_results=wdl)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT
    print(json.dumps({"out": args.out, "counts": manifest["counts"], "shards": len(manifest["shards"]), "report": manifest["provenance"]["buildReport"]}, indent=2))
    return EXIT_OK


def _maybe_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cmd_inventory_ringo(args) -> int:
    from .ringo_import import inventory

    report = inventory(args.shards)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return EXIT_OK


def cmd_import_ringo(args) -> int:
    from .ringo_import import import_targets

    os.makedirs(args.out, exist_ok=True)
    shards = [os.path.join(args.shards, n) for n in sorted(os.listdir(args.shards)) if n.endswith(".nngd")] if os.path.isdir(args.shards) else [args.shards]
    report = import_targets(shards, args.positions, args.mapping, os.path.join(args.out, "labels-imported.jsonl"), args.source_run_id)
    with open(os.path.join(args.out, "import-report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return EXIT_OK


def cmd_train(args) -> int:
    from .train import main as train_main
    return train_main(args.config, args.resume)


def cmd_evaluate(args) -> int:
    import torch
    from .checkpoint import load_checkpoint, model_from_checkpoint
    from .data_loader import load_split_arrays
    from .metrics import evaluate_split, gate_statistics, write_json

    ck = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(ck)
    fixture = os.path.exists(os.path.join(args.data, "fixture.npz")) and not os.path.exists(os.path.join(args.data, "manifest.json"))
    arrays = load_split_arrays(args.data, args.split, fixture)
    size = arrays["spatial"].shape[1]
    res = evaluate_split(model, arrays, size, args.mode, 1.0 if args.mode == "hard" else args.tau, ck["frozenPrefix"], torch.device("cpu"))
    res.update({"checkpoint": args.checkpoint, "step": ck["step"], "split": args.split, "gates": gate_statistics(model)})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        write_json(args.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "gates"}, indent=2))
    return EXIT_OK


def cmd_match(args) -> int:
    from .match import MatchRunner, parse_engine_spec

    if args.time_main_seconds is not None and (args.visits_a is not None or args.visits_b is not None):
        print("error: --time-main-seconds cannot be combined with --visits-a/--visits-b", file=sys.stderr)
        return EXIT_CONFIG
    try:
        engine_a = parse_engine_spec(args.engine_a)
        engine_b = parse_engine_spec(args.engine_b)
        openings_path = None if args.openings in (None, "none") else args.openings
        runner = MatchRunner(
            engine_a=engine_a, engine_b=engine_b, games=args.games, size=args.size, komi=args.komi,
            out_dir=args.out, seed=args.seed, openings_path=openings_path, visits_a=args.visits_a,
            visits_b=args.visits_b, time_main_seconds=args.time_main_seconds, max_moves=args.max_moves,
            command_timeout=args.command_timeout, genmove_timeout=args.genmove_timeout,
            resamples=args.resamples, log=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    report = runner.run()
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ichigo_train", description="IchiGo training tools (M0 subset).")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Report Python/torch/CUDA environment as JSON.")
    d.add_argument("--out", default=None)
    d.set_defaults(func=cmd_doctor)
    i = sub.add_parser("init-model", help="Create an untrained checkpoint with fixed wiring.")
    i.add_argument("--profile", required=True, choices=["tiny", "small", "base"])
    i.add_argument("--seed", type=int, default=20260908)
    i.add_argument("--board-sizes", type=int, nargs="+", default=[9])
    i.add_argument("--out", required=True)
    i.set_defaults(func=cmd_init_model)
    e = sub.add_parser("export", help="Export a checkpoint to a .ichigo directory.")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--board-sizes", type=int, nargs="+", default=[9])
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--calibration", default=None, help="calibrate report.json (T29); sets manifest.calibrationTemperature "
                                                        "and trainingProvenance.calibration (default: unverified, T=1.0)")
    e.set_defaults(func=cmd_export)
    n = sub.add_parser("inspect", help="Validate a .ichigo directory and print its manifest.")
    n.add_argument("--model", required=True)
    n.set_defaults(func=cmd_inspect)
    l = sub.add_parser("label", help="Label positions JSONL with an external KataGo analysis engine.")
    l.add_argument("--positions", required=True)
    l.add_argument("--teacher-bin", required=True)
    l.add_argument("--teacher-model", required=True)
    l.add_argument("--teacher-config", required=True)
    l.add_argument("--perspective", default="sidetomove", choices=["black", "white", "sidetomove"], help="must equal reportAnalysisWinratesAs in the teacher config")
    l.add_argument("--visits", type=int, default=128)
    l.add_argument("--timeout", type=float, default=60.0)
    l.add_argument("--queue-depth", type=int, default=16)
    l.add_argument("--max-positions", type=int, default=None)
    l.add_argument("--out", required=True)
    l.set_defaults(func=cmd_label)
    b = sub.add_parser("build-data", help="Join positions + labels into shard v1 dataset.")
    b.add_argument("--positions", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--results", default=None, help="optional JSONL {gameId,result:B|W|draw} for on-board finished games (wdl targets)")
    b.add_argument("--shard-size", type=int, default=4096)
    b.set_defaults(func=cmd_build_data)
    iv = sub.add_parser("inventory-ringo", help="Read-only inventory of RinGo .nngd v2 shards.")
    iv.add_argument("--shards", nargs="+", required=True)
    iv.add_argument("--out", default=None)
    iv.set_defaults(func=cmd_inventory_ringo)
    ir = sub.add_parser("import-ringo", help="Import RinGo v2 targets via an explicit sample mapping.")
    ir.add_argument("--shards", required=True)
    ir.add_argument("--positions", required=True)
    ir.add_argument("--mapping", required=True)
    ir.add_argument("--source-run-id", default=None)
    ir.add_argument("--out", required=True)
    ir.set_defaults(func=cmd_import_ringo)
    t = sub.add_parser("train", help="Train from a config JSON (docs/spec/02-training.md §10).")
    t.add_argument("--config", required=True)
    t.add_argument("--resume", default=None)
    t.set_defaults(func=cmd_train)
    ev = sub.add_parser("evaluate", help="Evaluate a checkpoint on a dataset split (soft or hard).")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--split", default="validation")
    ev.add_argument("--mode", default="hard", choices=["soft", "hard"])
    ev.add_argument("--tau", type=float, default=1.0)
    ev.add_argument("--out", default=None)
    ev.set_defaults(func=cmd_evaluate)
    f = sub.add_parser("make-fixtures", help="Regenerate Swift/Python parity and symmetry fixtures.")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_make_fixtures)
    m = sub.add_parser("match", help="Run a GTP engine-vs-engine (or vs the uniform-legal baseline) match (docs/spec/04-tasks.md T31).")
    m.add_argument("--engine-a", required=True, help="GTP argv: shell-quoted string or a JSON array of strings, e.g. '.build/release/ichigo gtp --model-9 m.ichigo --visits 100'")
    m.add_argument("--engine-b", required=True, help="GTP argv (as --engine-a), or the literal 'uniform' for the built-in uniform-legal baseline")
    m.add_argument("--games", type=int, required=True)
    m.add_argument("--size", type=int, default=9, choices=[9, 19])
    m.add_argument("--komi", type=float, default=7.0)
    m.add_argument("--openings", default="none", help="JSONL of {'id','moves':[gtp,...]} openings, or 'none' for the empty board")
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--visits-a", type=int, default=None, help="appended to engine A's argv as '--visits N' (conflicts with --time-main-seconds)")
    m.add_argument("--visits-b", type=int, default=None, help="appended to engine B's argv as '--visits N' (conflicts with --time-main-seconds)")
    m.add_argument("--time-main-seconds", type=float, default=None, help="sends 'time_settings T 0 0' (sudden death) instead of a fixed visit count")
    m.add_argument("--max-moves", type=int, default=None, help="default 4*size*size; exceeding it without two passes ends a game as 'truncated'")
    m.add_argument("--command-timeout", type=float, default=30.0, help="per-command timeout (seconds) for boardsize/clear_board/komi/play/final_score")
    m.add_argument("--genmove-timeout", type=float, default=None, help="per-genmove timeout (seconds); default max(120, 4*time-main-seconds)")
    m.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples for report.json's confidence intervals")
    m.set_defaults(func=cmd_match)
    sr = sub.add_parser("sgf-results", help="Parse each SGF's RE tag into a real game-result JSONL (docs/spec/05-validation.md §5, T29).")
    sr.add_argument("--sgf-dir", required=True)
    sr.add_argument("--positions", required=True, help="positions JSONL (the `features` CLI's output) for the sourceFile->gameId map")
    sr.add_argument("--out", required=True)
    sr.set_defaults(func=cmd_sgf_results)
    cal = sub.add_parser("calibrate", help="Fit a WDL calibration temperature on real-result positions and report Brier/ECE (docs/spec/05-validation.md §5, T29).")
    cal.add_argument("--checkpoint", required=True)
    cal.add_argument("--data", required=True, help="shard v1 dataset directory (ideally built with `build-data --results`)")
    cal.add_argument("--results", required=True, help="sgf-results JSONL ({gameId,result,kind})")
    cal.add_argument("--out", required=True)
    cal.add_argument("--resamples", type=int, default=10000)
    cal.add_argument("--seed", type=int, default=20260908)
    cal.set_defaults(func=cmd_calibrate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
