"""Training config schema (docs/spec/02-training.md §10). Unknown keys are errors."""

from __future__ import annotations

import json
import os

DEFAULTS = {
    "schemaVersion": 1,
    "profile": "small",
    "seed": 20260908,
    "device": "cpu",
    "microBatch": 8,
    "effectiveBatch": 128,
    "maxSteps": 20000,
    "validationInterval": 500,
    "checkpointInterval": 1000,
    "gateLearningRate": 0.01,
    "headLearningRate": 0.001,
    "headWeightDecay": 0.0001,
    "gradClipNorm": 1.0,
    "discretization": "prefix-60-30-10",
    "augmentation": "d4",
    "precision": "fp32",
    # IchiGo-specific optional keys (documented in docs/implementation-status.md)
    "tauFinal": 0.2,
    "maxValidationPositions": None,   # None = whole holdout
    "fixtureMode": False,             # True: data dir holds fixture.npz; train == validation, no split
    "freezeWarningThreshold": 0.2,
    "throughputReference": None,      # path to a 1-GPU run-summary.json to compute DDP throughput ratio/scaling efficiency against (T26); None = skip
}
DEFAULTS.update({
    # T33 capacity/wiring/CNN-baseline experiment keys
    "modelType": "logic", "channels": None, "dilations": None, "bank1Ratio": 0.1, "wiringSeed": None, "headVersion": 2,
})
REQUIRED = ["data", "out", "boardSize", "runId"]
KNOWN = set(DEFAULTS) | set(REQUIRED)


class ConfigError(ValueError):
    pass


def load_config(path: str) -> tuple[dict, dict]:
    """Returns (raw, resolved). Relative paths resolve against the current working directory."""
    with open(path) as f:
        raw = json.load(f)
    unknown = set(raw) - KNOWN
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"missing required keys: {missing}")
    cfg = dict(DEFAULTS)
    cfg.update(raw)
    if cfg["schemaVersion"] != 1:
        raise ConfigError("schemaVersion must be 1")
    if cfg["profile"] not in ("tiny", "small", "base"):
        raise ConfigError("profile must be tiny/small/base")
    if cfg["modelType"] not in ("logic", "cnn-baseline"):
        raise ConfigError("modelType must be logic or cnn-baseline")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ConfigError("device must be cpu or cuda")
    if cfg["precision"] != "fp32":
        raise ConfigError("precision must be fp32 (v1)")
    if cfg["discretization"] not in ("prefix-60-30-10",):
        raise ConfigError("discretization must be prefix-60-30-10 (gumbel-ste-90-10 is not implemented)")
    if cfg["augmentation"] not in ("d4", "none"):
        raise ConfigError("augmentation must be d4 or none")
    if cfg["boardSize"] not in (9, 19):
        raise ConfigError("boardSize must be 9 or 19")
    for k in ("microBatch", "effectiveBatch", "maxSteps", "validationInterval", "checkpointInterval"):
        if not isinstance(cfg[k], int) or cfg[k] <= 0:
            raise ConfigError(f"{k} must be a positive integer")
    if cfg["effectiveBatch"] % cfg["microBatch"] != 0:
        raise ConfigError("effectiveBatch must be divisible by microBatch")
    cfg["data"] = os.path.abspath(cfg["data"])
    cfg["out"] = os.path.abspath(cfg["out"])
    if cfg["throughputReference"] is not None:
        cfg["throughputReference"] = os.path.abspath(cfg["throughputReference"])
    return raw, cfg
