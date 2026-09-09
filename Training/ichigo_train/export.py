"""Export a LogicNet (checkpoint or in-memory) to a ``.ichigo`` directory (T09).

Atomic: write to a temporary sibling directory, re-read and verify it, then rename.
Refuses to overwrite an existing output unless ``overwrite=True``.
"""

from __future__ import annotations

import datetime as _dt
import os
import shutil
import tempfile

import numpy as np
import torch

from . import model_format as MF
from .model import LogicNet
from .wiring import ModelSpec, Wiring


def save_checkpoint(model: LogicNet, path: str, extra: dict | None = None) -> None:
    """Minimal M0 checkpoint: spec + wiring + parameters. Training-resume checkpoints (T15)
    extend this dict; Swift never reads it."""
    torch.save({
        "schemaVersion": 1,
        "spec": {"channels": model.spec.channels, "dilations": model.spec.dilations, "seed": model.spec.seed, "profile": model.spec.profile},
        "headVersion": model.head_version,
        "wiring": model.wiring_numpy(),
        "theta": model.theta.detach().cpu().numpy(),
        "heads": model.head_numpy(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str) -> LogicNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    spec = ModelSpec(**ck["spec"])
    model = LogicNet(spec, Wiring(wiring=ck["wiring"], theta=ck["theta"], dilations=spec.dilations), heads=ck["heads"], head_version=ck.get("headVersion", 1))
    return model


def _calibration_provenance(calibration: dict | None) -> tuple[float, dict]:
    """docs/spec/03-engine.md §9 / docs/spec/05-validation.md §5 (T29): resolve a loaded
    ``calibrate`` report (``ichigo_train.calibrate.run_calibration``'s JSON) into the manifest's
    ``calibrationTemperature`` plus a ``trainingProvenance.calibration`` record.

    Absent, or present but fit on zero real-result validation positions (``verified`` false --
    "教師予測だけしかない場合は校正未検証、T=1を保持する"), keeps the safe default: T=1.0,
    status "unverified". A verified report contributes its fitted T and the held-out test
    Brier/ECE at both T=1 and the fitted T; this is NN (root raw) calibration only -- search
    winrates are a separate metric that this temperature does not itself measure."""
    if calibration is None:
        return 1.0, {"status": "unverified"}
    if not calibration.get("verified", False):
        return 1.0, {"status": "unverified", "reason": "calibration report had no real-result validation positions"}
    t = float(calibration["fittedTemperature"])
    test = calibration.get("test", {})
    t1, fitted = test.get("T1", {}), test.get("fitted", {})
    return t, {
        "status": "fitted",
        "temperature": t,
        "scope": "raw root NN calibration only (forward_hard -> softmax(wdl_logits/T)); search-derived winrates are a separate metric (docs/spec/05-validation.md §5)",
        "testGames": test.get("games"),
        "testPositions": test.get("positions"),
        "testInsufficientSamples": test.get("insufficientSamples"),
        "testBrierT1": t1.get("brier"),
        "testBrierFitted": fitted.get("brier"),
        "testECET1": t1.get("ece"),
        "testECEFitted": fitted.get("ece"),
        "testBrierFittedCI": fitted.get("brierGameBootstrapCI"),
    }


def export_model(model: LogicNet, out: str, board_sizes: list[int], provenance: dict | None = None, overwrite: bool = False,
                 calibration: dict | None = None) -> dict:
    """Write ``out`` (a directory). Returns the manifest.

    Learned wiring is resolved to its argmax fixed wiring (with the A!=B fallback) before the
    normal ``.ichigo`` serializer runs; the on-disk format remains unchanged.

    ``calibration`` (optional) is a loaded ``calibrate`` report dict (T29); see
    ``_calibration_provenance`` for how it maps to ``calibrationTemperature`` and
    ``trainingProvenance.calibration``.
    """
    out = os.path.normpath(out)
    if os.path.lexists(out):
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite=True / --overwrite")
        if os.path.islink(out) or not os.path.isdir(out):
            raise FileExistsError(f"{out} exists and is not a model directory")
    prov = dict(provenance or {})
    prov.setdefault("generatedAt", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    prov.setdefault("seed", model.spec.seed)
    prov.setdefault("profile", model.spec.profile)
    temperature, calib_prov = _calibration_provenance(calibration)
    prov["calibration"] = calib_prov
    files = MF.serialize_model(model.wiring_numpy(), model.hard_gates(), model.head_numpy(), model.dilations, board_sizes, prov,
                               calibration_temperature=temperature, head_version=model.head_version)
    parent = os.path.dirname(out) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".export-", dir=parent)
    tmpdir = os.path.join(tmp, "model")
    try:
        MF.write_model_dir(tmpdir, files)
        loaded = MF.read_model(tmpdir)  # re-read + verify before rename
        _verify_roundtrip(model, loaded)
        # Keep the old model until the replacement is in place: move it aside, promote the new
        # one, and only then drop the backup. If the promotion fails, put the old one back.
        backup = None
        if os.path.lexists(out):
            backup = os.path.join(tmp, "backup")
            os.rename(out, backup)
        try:
            os.rename(tmpdir, out)
        except BaseException:
            if backup is not None and not os.path.lexists(out):
                os.rename(backup, out)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return loaded.manifest


def _verify_roundtrip(model: LogicNet, loaded: MF.LoadedModel) -> None:
    if not np.array_equal(loaded.wiring, model.wiring_numpy()):
        raise MF.ModelFormatError("re-read wiring differs")
    if not np.array_equal(loaded.gates, model.hard_gates()):
        raise MF.ModelFormatError("re-read gates differ")
    for n, arr in model.head_numpy().items():
        if not np.array_equal(loaded.heads[n], arr):
            raise MF.ModelFormatError(f"re-read head {n} differs")


def model_from_loaded(loaded: MF.LoadedModel) -> LogicNet:
    """Rebuild a LogicNet whose argmax gates equal the file's gates (theta = one-hot*3).

    Sets a ``calibration_temperature`` attribute (docs/spec/03-engine.md §9) from the manifest,
    for evaluate paths that read a loaded ``.ichigo`` model to pass into ``model.postprocess``."""
    m = loaded.manifest
    spec = ModelSpec(channels=m["channels"], dilations=list(m["dilations"]), seed=0)
    L, C = loaded.gates.shape
    theta = np.zeros((L, C, 16), dtype=np.float32)
    theta[np.arange(L)[:, None], np.arange(C)[None, :], loaded.gates.astype(np.int64)] = 3.0
    model = LogicNet(spec, Wiring(wiring=loaded.wiring, theta=theta, dilations=spec.dilations), heads=loaded.heads, head_version=int(m["headVersion"]))
    model.calibration_temperature = float(m.get("calibrationTemperature", 1.0))
    return model
