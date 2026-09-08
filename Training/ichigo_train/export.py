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


def export_model(model: LogicNet, out: str, board_sizes: list[int], provenance: dict | None = None, overwrite: bool = False) -> dict:
    """Write ``out`` (a directory). Returns the manifest. Gates are the argmax of theta."""
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
    files = MF.serialize_model(model.wiring_numpy(), model.hard_gates(), model.head_numpy(), model.dilations, board_sizes, prov, head_version=model.head_version)
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
    """Rebuild a LogicNet whose argmax gates equal the file's gates (theta = one-hot*3)."""
    m = loaded.manifest
    spec = ModelSpec(channels=m["channels"], dilations=list(m["dilations"]), seed=0)
    L, C = loaded.gates.shape
    theta = np.zeros((L, C, 16), dtype=np.float32)
    theta[np.arange(L)[:, None], np.arange(C)[None, :], loaded.gates.astype(np.int64)] = 3.0
    return LogicNet(spec, Wiring(wiring=loaded.wiring, theta=theta, dilations=spec.dilations), heads=loaded.heads, head_version=int(m["headVersion"]))
