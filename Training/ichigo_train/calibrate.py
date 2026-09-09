"""Win-probability calibration of the raw root NN evaluation (docs/spec/05-validation.md §5,
docs/spec/03-engine.md §9, T29).

``python -m ichigo_train calibrate --checkpoint CK --data DATASET --results RESULTS --out OUT``:

1. Reads ``DATASET`` (a shard v1 dataset, ``dataset.py``/T13) and ``RESULTS`` (one JSONL row per
   game, ``sgf_results.build_results``'s ``{"gameId","result","kind"}`` output, normally fed into
   the dataset via ``build-data --results`` so the dataset's own ``wdl`` target already holds the
   real, to-move-perspective outcome one-hot on positions with ``target_mask[:,4]==1``).
2. On the validation split, runs the checkpoint's exact ``forward_hard`` to get raw wdl logits,
   and fits a single positive temperature ``T`` in ``[0.25,4]`` minimising the expected-score BCE
   (NLL) of ``e(T) = softmax(logits/T)[win] + 0.5*softmax(logits/T)[draw]`` against the outcome
   ``o`` (win 1 / draw 0.5 / loss 0, to-move perspective) -- a deterministic scipy-free 1-D search
   (coarse log-spaced grid, then golden-section refinement).
3. On the held-out test split, reports Brier and ECE at both ``T=1`` and the fitted ``T``, with
   game-level bootstrap confidence intervals for Brier (positions within one game are correlated,
   so the resampling unit is the game, not the position -- docs/spec/05-validation.md §5).

This calibrates the *raw root NN* evaluation only: no search runs here. Root-raw is the only
mode this module ever measures; search-derived winrates are a separate metric that must be
calibrated (or verified) separately, per docs/spec/05-validation.md §5 "root rawとsearch別に
指標を記録し、NNのTだけを調整する". If validation has zero positions with a real result (the
dataset was not built with ``--results``, or ``RESULTS`` shares no games with it), the fit falls
back to ``T=1`` and the report is marked ``"verified": false`` -- ``export --calibration`` must
then keep the manifest's default unverified ``T=1`` rather than trust an unfounded fit
(docs/spec/05-validation.md §5 "教師予測だけしかない場合は校正未検証、T=1を保持する").
"""

from __future__ import annotations

import collections
import json
import math
import os

import numpy as np
import torch

from . import dataset as D
from .stats import bootstrap_mean_ci

T_MIN, T_MAX = 0.25, 4.0
ECE_BINS = 10
MIN_COMPLETE_GAMES = 100
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260908


# ----- results file -----

def load_results(path: str) -> dict[str, dict]:
    """``{gameId: {"result": "B"|"W"|"draw", "kind": "resign"|"score"|"draw"}}`` from a
    ``sgf_results.build_results`` JSONL file."""
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["gameId"]] = {"result": r["result"], "kind": r["kind"]}
    return out


# ----- checkpoint loading (mirrors __main__.cmd_export's fallback) -----

def load_checkpoint_model(path: str):
    from .export import load_checkpoint as load_plain_ck
    try:
        return load_plain_ck(path)
    except KeyError:
        from .checkpoint import load_checkpoint as load_train_ck
        from .checkpoint import model_from_checkpoint
        return model_from_checkpoint(load_train_ck(path))


def checkpoint_step(path: str) -> int | None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    step = ck.get("step")
    return int(step) if step is not None else None


# ----- collecting real-result positions from the dataset -----

def _hex_ids(rows: np.ndarray) -> np.ndarray:
    return np.array([bytes(row).hex() for row in rows])


def collect_split(data_dir: str, split: str, results: dict[str, dict]) -> dict[str, np.ndarray]:
    """Rows of ``split`` with a baked-in real-result wdl target (``target_mask[:,4]==1``, written
    by ``build-data --results``) whose ``gameId`` is also present in ``results`` (defends against
    a dataset built from a stale/different results file). Returns ``spatial``/``global`` (model
    inputs), ``outcome`` (the dataset's to-move wdl one-hot), ``gameId`` (hex, one per row) and
    ``weight`` (the dataset's ``sample_weight``), concatenated across shards."""
    spatials, globals_, outcomes, game_ids, weights = [], [], [], [], []
    for _, arrays in D.iter_split(data_dir, split, verify=True):
        m = arrays["target_mask"][:, 4] == 1
        if not m.any():
            continue
        idx_all = np.nonzero(m)[0]
        gids = _hex_ids(arrays["game_id"][idx_all])
        keep = np.array([g in results for g in gids]) if len(gids) else np.zeros(0, dtype=bool)
        if not keep.any():
            continue
        idx = idx_all[keep]
        spatials.append(arrays["spatial"][idx])
        globals_.append(arrays["global"][idx])
        outcomes.append(arrays["wdl"][idx])
        game_ids.append(gids[keep])
        weights.append(arrays["sample_weight"][idx])
    if not spatials:
        return {"spatial": np.zeros((0, 1, 1, 32), np.uint8), "global": np.zeros((0, 4), np.float32),
                "outcome": np.zeros((0, 3), np.float32), "gameId": np.zeros((0,), dtype="<U64"),
                "weight": np.zeros((0,), np.float32)}
    return {
        "spatial": np.concatenate(spatials, axis=0), "global": np.concatenate(globals_, axis=0),
        "outcome": np.concatenate(outcomes, axis=0), "gameId": np.concatenate(game_ids, axis=0),
        "weight": np.concatenate(weights, axis=0),
    }


@torch.no_grad()
def wdl_logits_for(model, spatial: np.ndarray, global_: np.ndarray, batch: int = 256) -> np.ndarray:
    """Raw wdl logits from ``forward_hard`` (the exact-gate numeric reference)."""
    if spatial.shape[0] == 0:
        return np.zeros((0, 3), np.float32)
    model.eval()
    out = []
    for i in range(0, spatial.shape[0], batch):
        sp = torch.from_numpy(np.ascontiguousarray(spatial[i:i + batch]))
        gl = torch.from_numpy(np.ascontiguousarray(global_[i:i + batch]).astype(np.float32))
        r = model.forward_hard(sp, gl)
        out.append(r["wdl_logits"].numpy())
    return np.concatenate(out, axis=0)


# ----- e/o, temperature fit, and metrics -----

def expected_result(wdl_logits: np.ndarray, temperature: float) -> np.ndarray:
    """``e(T) = softmax(wdl_logits / T)[win] + 0.5 * softmax(wdl_logits / T)[draw]`` (hard
    forward, to-move perspective) -- the same formula ``Postprocess.evaluate`` applies in Swift
    and ``model.postprocess`` applies in Python (docs/spec/03-engine.md §9)."""
    if wdl_logits.shape[0] == 0:
        return np.zeros((0,), np.float64)
    z = wdl_logits.astype(np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    q = e / e.sum(axis=1, keepdims=True)
    return q[:, 0] + 0.5 * q[:, 1]


def outcome_of(onehot_wdl: np.ndarray) -> np.ndarray:
    """win 1 / draw .5 / loss 0, to-move perspective, from a win/draw/loss one-hot."""
    if onehot_wdl.shape[0] == 0:
        return np.zeros((0,), np.float64)
    return onehot_wdl[:, 0].astype(np.float64) + 0.5 * onehot_wdl[:, 1].astype(np.float64)


def _bce_nll(e: np.ndarray, o: np.ndarray, w: np.ndarray) -> float | None:
    """Expected-score BCE / NLL: ``o`` is a soft label in {0,0.5,1}, ``e`` a probability
    prediction in (0,1) -- minimised at ``e=o`` including the draw case ``o=0.5``. ``None``
    (rather than NaN, which is not valid JSON) when there is no weighted sample to score."""
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    eps = 1e-12
    e = np.clip(e, eps, 1 - eps)
    return float(np.sum(w * -(o * np.log(e) + (1 - o) * np.log(1 - e))) / wsum)


def brier(e: np.ndarray, o: np.ndarray, w: np.ndarray) -> float | None:
    """``None`` (not NaN -- not valid JSON) when there is no weighted sample to score."""
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    return float(np.sum(w * (e - o) ** 2) / wsum)


def ece(e: np.ndarray, o: np.ndarray, w: np.ndarray, bins: int = ECE_BINS) -> float | None:
    """docs/spec/05-validation.md §5: 10 equal-width bins of the prediction ``e``, sample-weighted
    ``|mean(e) - mean(o)|`` per bin, summed with bin-weight (bin-count / total); empty bins are
    skipped entirely (neither in the sum nor the weight). ``None`` (not NaN) when empty."""
    total = float(w.sum())
    if total <= 0:
        return None
    idx = np.clip((e * bins).astype(np.int64), 0, bins - 1)
    total_gap = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        wb = float(w[m].sum())
        pred = float(np.average(e[m], weights=w[m]))
        actual = float(np.average(o[m], weights=w[m]))
        total_gap += (wb / total) * abs(pred - actual)
    return float(total_gap)


def fit_temperature(wdl_logits: np.ndarray, o: np.ndarray, w: np.ndarray, lo: float = T_MIN, hi: float = T_MAX,
                    grid_points: int = 65, golden_iters: int = 60) -> float:
    """Deterministic, scipy-free 1-D search for the ``T in [lo,hi]`` minimising ``_bce_nll(e(T),
    o, w)``: a coarse log-spaced grid locates the neighbourhood of the minimum, then golden-section
    search refines within the grid cell either side of the best point. No randomness anywhere, so
    the same inputs always return the same ``T``."""
    if wdl_logits.shape[0] == 0:
        return 1.0

    def obj(t: float) -> float:
        return _bce_nll(expected_result(wdl_logits, t), o, w)

    log_lo, log_hi = math.log(lo), math.log(hi)
    grid = np.linspace(log_lo, log_hi, grid_points)
    vals = [obj(math.exp(g)) for g in grid]
    best = int(np.argmin(vals))
    a = float(grid[max(best - 1, 0)])
    b = float(grid[min(best + 1, grid_points - 1)])
    if not a < b:
        a, b = log_lo, log_hi
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    inv_phi2 = (3.0 - math.sqrt(5.0)) / 2.0
    c = a + inv_phi2 * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = obj(math.exp(c)), obj(math.exp(d))
    for _ in range(golden_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + inv_phi2 * (b - a)
            fc = obj(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = obj(math.exp(d))
    t = math.exp((a + b) / 2.0)
    return float(min(max(t, lo), hi))


def per_game_brier(e: np.ndarray, o: np.ndarray, w: np.ndarray, game_ids: np.ndarray) -> dict[str, float]:
    return {g: brier(e[game_ids == g], o[game_ids == g], w[game_ids == g]) for g in np.unique(game_ids)}


# ----- report -----

def run_calibration(checkpoint: str, data_dir: str, results_path: str, out_path: str | None = None,
                    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    results = load_results(results_path)
    kind_counts = collections.Counter(r["kind"] for r in results.values())
    result_counts = collections.Counter(r["result"] for r in results.values())

    model = load_checkpoint_model(checkpoint)
    step = checkpoint_step(checkpoint)

    val = collect_split(data_dir, "validation", results)
    val_logits = wdl_logits_for(model, val["spatial"], val["global"])
    val_o = outcome_of(val["outcome"])
    verified = val_logits.shape[0] > 0
    fitted_t = fit_temperature(val_logits, val_o, val["weight"]) if verified else 1.0
    val_games = sorted(set(val["gameId"].tolist()))

    test = collect_split(data_dir, "test", results)
    test_logits = wdl_logits_for(model, test["spatial"], test["global"])
    test_o = outcome_of(test["outcome"])
    test_games = sorted(set(test["gameId"].tolist()))
    insufficient = len(test_games) < MIN_COMPLETE_GAMES

    def metrics_at(t: float) -> dict:
        e = expected_result(test_logits, t)
        per_game = per_game_brier(e, test_o, test["weight"], test["gameId"])
        ci = bootstrap_mean_ci(list(per_game.values()), resamples=resamples, seed=seed)
        return {"temperature": t, "brier": brier(e, test_o, test["weight"]), "ece": ece(e, test_o, test["weight"]),
                "brierGameBootstrapCI": ci}

    report = {
        "schemaVersion": 1,
        "checkpoint": os.path.basename(checkpoint),
        "checkpointStep": step,
        "dataDir": data_dir,
        "datasetHash": D.manifest_hash(data_dir),
        "resultsFile": os.path.basename(results_path),
        "resultsCounts": {"games": len(results), "byKind": dict(kind_counts), "byResult": dict(result_counts)},
        "scope": ("raw root NN calibration only: e = softmax(forward_hard(spatial,global).wdl_logits / T). "
                  "No search runs in this command; search-derived winrates are a separate metric and are not "
                  "calibrated by this report (docs/spec/05-validation.md §5)."),
        "temperatureRange": [T_MIN, T_MAX],
        "objective": "expected-score BCE / NLL of e=win+0.5*draw against outcome o (win 1 / draw .5 / loss 0)",
        "verified": verified,
        "fittedTemperature": fitted_t,
        "validation": {"positions": int(val_logits.shape[0]), "games": len(val_games), "fittedTemperature": fitted_t,
                       "nll": _bce_nll(expected_result(val_logits, fitted_t), val_o, val["weight"]) if verified else None},
        "test": {
            "positions": int(test_logits.shape[0]), "games": len(test_games),
            "minCompleteGames": MIN_COMPLETE_GAMES, "insufficientSamples": insufficient,
            "T1": metrics_at(1.0), "fitted": metrics_at(fitted_t),
        },
    }
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
    return report
