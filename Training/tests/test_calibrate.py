"""docs/spec/05-validation.md §5 / docs/spec/03-engine.md §9 (T29): WDL calibration."""

import json
import os

import numpy as np
import pytest

from ichigo_train import calibrate as C
from ichigo_train import dataset as D
from ichigo_train import model_format as MF
from ichigo_train.__main__ import main
from ichigo_train.export import export_model, save_checkpoint
from ichigo_train.model import build_model
from ichigo_train.teacher import gtp_to_index

S = 9


def _game(gid_seed, moves):
    """Each game gets a distinct komi so no two games' turn-0 (empty board) position collides
    under dataset dedupe (see test_sgf_results.py's ``_game`` for the same note)."""
    rows = []
    komi = 7 + (gid_seed % 1000) * 0.01
    gid = D.canonical_hash(S, komi, [], "B", moves)
    for t in range(len(moves) + 1):
        pid = D.canonical_hash(S, komi, [], "B", moves[:t])
        legal = [1] * (S * S + 1)
        for _, m in moves[:t]:
            legal[gtp_to_index(m, S)] = 0
        spatial = [0] * (S * S * 32)
        rows.append({"positionId": pid, "gameId": gid, "boardSize": S, "komi": komi, "rulesId": D.RULES_ID,
                     "initialStones": [], "initialPlayer": "B", "moves": moves[:t], "turnNumber": t,
                     "toMove": "B" if t % 2 == 0 else "W", "spatial": spatial, "global": [0, S / 19, t / 162, 0],
                     "legal": legal, "sourceFile": f"{gid_seed}.sgf"})
    return gid, rows


def _label(row):
    pol = [0.0] * (S * S + 1)
    legal_idx = [i for i, l in enumerate(row["legal"]) if l]
    pol[legal_idx[0]] = 1.0
    return {"positionId": row["positionId"], "gameId": row["gameId"], "turnNumber": row["turnNumber"], "toMove": row["toMove"],
            "boardSize": S, "policy": pol, "expectedResult": 0.5, "score": 0.0, "ownership": [0.0] * (S * S),
            "teacherVisits": 1, "sourceType": "teacher", "teacherId": "t"}


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---- ECE: hand-computed tiny case ----

def test_ece_hand_computed():
    # bin 0 ([0,.1)): e=.05 twice, o={0,1} -> mean pred .05, mean actual .5, gap .45, weight 2/3
    # bin 9 ([.9,1]): e=.95 once, o=1        -> mean pred .95, mean actual 1,  gap .05, weight 1/3
    # all other bins empty and skipped.
    e = np.array([0.05, 0.05, 0.95])
    o = np.array([0.0, 1.0, 1.0])
    w = np.ones(3)
    expected = (2 / 3) * 0.45 + (1 / 3) * 0.05
    assert C.ece(e, o, w) == pytest.approx(expected, abs=1e-12)


def test_ece_perfect_calibration_is_zero():
    e = np.linspace(0.05, 0.95, 10)
    w = np.ones(10)
    assert C.ece(e, e.copy(), w) == pytest.approx(0.0, abs=1e-12)


def test_ece_skips_empty_bins():
    # only two of ten bins populated; the weighting must be relative to populated weight, not 10 bins.
    e = np.array([0.15, 0.85])
    o = np.array([0.15, 0.85])
    assert C.ece(e, o, np.ones(2)) == pytest.approx(0.0, abs=1e-12)


def test_brier():
    e = np.array([0.9, 0.5, 0.1])
    o = np.array([1.0, 0.5, 0.0])
    assert C.brier(e, o, np.ones(3)) == pytest.approx(((0.1 ** 2) + 0 + (0.1 ** 2)) / 3)


# ---- temperature fit: synthetic overconfident model recovers T>1 ----

def test_fit_temperature_recovers_overconfident_model():
    """Every position's raw logits predict a near-certain win ([1.5,0,-1.5], e(T=1)~=.873), but
    the true win rate is only 75%: the model is overconfident and T=1 should not be the minimiser.
    NLL/BCE for a constant prediction is minimised exactly at the empirical mean, so fitting
    should push T up until e(T) == .75 -- this also pins down the *exact* recovered T, not just
    its sign, without depending on scipy or randomness (search is deterministic)."""
    n = 1000
    logits = np.tile(np.array([1.5, 0.0, -1.5], dtype=np.float32), (n, 1))
    o = np.concatenate([np.ones(750), np.zeros(250)])
    w = np.ones(n)

    t_fit = C.fit_temperature(logits, o, w)
    assert 1.3 < t_fit < 2.2  # comfortably > 1 (softened) and well inside [0.25,4]

    e_fit = C.expected_result(logits, t_fit)
    assert e_fit.mean() == pytest.approx(0.75, abs=1e-3)
    assert C._bce_nll(e_fit, o, w) < C._bce_nll(C.expected_result(logits, 1.0), o, w)


def test_fit_temperature_stays_in_bounds_and_is_deterministic():
    n = 200
    logits = np.tile(np.array([6.0, 0.0, -6.0], dtype=np.float32), (n, 1))  # extremely overconfident
    o = np.full(n, 0.5)  # true outcome is a coin flip: needs a very soft (large T) prediction
    w = np.ones(n)
    t1 = C.fit_temperature(logits, o, w)
    t2 = C.fit_temperature(logits, o, w)
    assert t1 == t2  # deterministic (no RNG anywhere in the search)
    assert C.T_MIN <= t1 <= C.T_MAX
    assert t1 == pytest.approx(C.T_MAX, abs=1e-6)  # clamped: no finite T fully flattens this to .5


def test_fit_temperature_with_no_positions_defaults_to_one():
    assert C.fit_temperature(np.zeros((0, 3)), np.zeros(0), np.zeros(0)) == 1.0


# ---- run_calibration: insufficient-samples flag + full report shape ----

def _small_calibration_fixture(tmp_path, n_games=60, n_results=2):
    rng = np.random.default_rng(9)
    pts = [f"{c}{r}" for c in "ABCDEFGHJ" for r in range(1, 10)]
    rows, game_ids = [], []
    for g in range(n_games):
        seq = list(rng.permutation(pts)[:5])
        moves = [["B" if i % 2 == 0 else "W", m] for i, m in enumerate(seq)]
        gid, grows = _game(g, moves)
        game_ids.append(gid)
        rows += grows
    labels = [_label(r) for r in rows]

    results_rows = []
    for i in range(n_results):
        result, kind = ("B", "resign") if i % 2 == 0 else ("W", "score")
        results_rows.append({"gameId": game_ids[i], "result": result, "kind": kind})

    pos, lab, res = tmp_path / "positions.jsonl", tmp_path / "labels.jsonl", tmp_path / "results.jsonl"
    _write(pos, rows)
    _write(lab, labels)
    _write(res, results_rows)

    wdl_from_results = {r["gameId"]: r["result"] for r in results_rows}
    ds_dir = tmp_path / "ds"
    D.build_dataset(str(pos), str(lab), str(ds_dir), {}, wdl_from_results=wdl_from_results)

    model = build_model("tiny", seed=1)
    ck = tmp_path / "ck.pt"
    save_checkpoint(model, str(ck))
    return ck, ds_dir, res


def test_insufficient_samples_flag_and_report_shape(tmp_path):
    ck, ds_dir, res = _small_calibration_fixture(tmp_path, n_games=60, n_results=2)
    out = tmp_path / "report.json"
    report = C.run_calibration(str(ck), str(ds_dir), str(res), str(out))

    assert report["resultsCounts"]["games"] == 2
    assert report["test"]["games"] < C.MIN_COMPLETE_GAMES
    assert report["test"]["insufficientSamples"] is True
    assert report["scope"].startswith("raw root NN calibration only")
    assert report["temperatureRange"] == [C.T_MIN, C.T_MAX]
    assert set(report["test"]["T1"]) == {"temperature", "brier", "ece", "brierGameBootstrapCI"}
    assert set(report["test"]["fitted"]) == {"temperature", "brier", "ece", "brierGameBootstrapCI"}
    assert report["test"]["T1"]["brierGameBootstrapCI"]["resamples"] > 0
    assert os.path.isfile(out)
    with open(out) as f:
        assert json.load(f) == report


def test_run_calibration_unverified_when_no_real_results(tmp_path):
    """A dataset built without ``--results`` (no wdl_from_results) has zero real-result validation
    positions: the fit must fall back to T=1 and mark the report unverified, per
    docs/spec/05-validation.md §5 "教師予測だけしかない場合は校正未検証、T=1を保持する"."""
    rng = np.random.default_rng(1)
    pts = [f"{c}{r}" for c in "ABCDEFGHJ" for r in range(1, 10)]
    rows, game_ids = [], []
    for g in range(60):
        seq = list(rng.permutation(pts)[:5])
        moves = [["B" if i % 2 == 0 else "W", m] for i, m in enumerate(seq)]
        gid, grows = _game(g, moves)
        game_ids.append(gid)
        rows += grows
    labels = [_label(r) for r in rows]
    pos, lab = tmp_path / "positions.jsonl", tmp_path / "labels.jsonl"
    _write(pos, rows)
    _write(lab, labels)
    D.build_dataset(str(pos), str(lab), str(tmp_path / "ds"), {})  # no wdl_from_results

    # a results file exists (some game finished) but the dataset was never rebuilt with it
    res = tmp_path / "results.jsonl"
    _write(res, [{"gameId": game_ids[0], "result": "B", "kind": "resign"}])

    model = build_model("tiny", seed=2)
    ck = tmp_path / "ck.pt"
    save_checkpoint(model, str(ck))

    report = C.run_calibration(str(ck), str(tmp_path / "ds"), str(res), str(tmp_path / "report.json"))
    assert report["verified"] is False
    assert report["fittedTemperature"] == 1.0
    assert report["validation"]["positions"] == 0


# ---- CLI wiring ----

def test_cli_sgf_results_and_calibrate_and_export(tmp_path):
    ck, ds_dir, res = _small_calibration_fixture(tmp_path, n_games=60, n_results=2)
    report_path = tmp_path / "report.json"
    rc = main(["calibrate", "--checkpoint", str(ck), "--data", str(ds_dir), "--results", str(res), "--out", str(report_path)])
    assert rc == 0
    assert os.path.isfile(report_path)

    model_out = tmp_path / "m.ichigo"
    rc = main(["export", "--checkpoint", str(ck), "--out", str(model_out), "--calibration", str(report_path)])
    assert rc == 0
    loaded = MF.read_model(str(model_out))
    with open(report_path) as f:
        report = json.load(f)
    assert loaded.manifest["calibrationTemperature"] == pytest.approx(report["fittedTemperature"])
    assert loaded.manifest["trainingProvenance"]["calibration"]["status"] in ("fitted", "unverified")


# ---- export writes T (docs/spec/03-engine.md §9) ----

def test_export_default_is_unverified_temperature_one(tmp_path):
    model = build_model("tiny", seed=2)
    out = tmp_path / "m.ichigo"
    manifest = export_model(model, str(out), [9])
    assert manifest["calibrationTemperature"] == 1.0
    assert manifest["trainingProvenance"]["calibration"] == {"status": "unverified"}


def test_export_writes_fitted_calibration_temperature_and_provenance(tmp_path):
    model = build_model("tiny", seed=2)
    calib = {
        "verified": True, "fittedTemperature": 1.83,
        "test": {"games": 120, "positions": 900, "insufficientSamples": False,
                 "T1": {"brier": 0.21, "ece": 0.09},
                 "fitted": {"brier": 0.14, "ece": 0.03, "brierGameBootstrapCI": {"low": 0.11, "high": 0.17}}},
    }
    out = tmp_path / "m.ichigo"
    manifest = export_model(model, str(out), [9], calibration=calib)
    assert manifest["calibrationTemperature"] == pytest.approx(1.83)
    prov = manifest["trainingProvenance"]["calibration"]
    assert prov["status"] == "fitted"
    assert prov["temperature"] == pytest.approx(1.83)
    assert prov["testBrierT1"] == 0.21 and prov["testBrierFitted"] == 0.14
    assert prov["testECET1"] == 0.09 and prov["testECEFitted"] == 0.03
    assert prov["testGames"] == 120
    assert prov["testInsufficientSamples"] is False

    loaded = MF.read_model(str(out))
    assert loaded.manifest["calibrationTemperature"] == pytest.approx(1.83)


def test_export_keeps_default_when_report_is_unverified(tmp_path):
    model = build_model("tiny", seed=2)
    calib = {"verified": False, "fittedTemperature": 3.0, "test": {}}
    out = tmp_path / "m.ichigo"
    manifest = export_model(model, str(out), [9], calibration=calib)
    assert manifest["calibrationTemperature"] == 1.0
    assert manifest["trainingProvenance"]["calibration"]["status"] == "unverified"
