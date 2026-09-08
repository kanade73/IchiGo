"""Model construction from a training config dict (docs/spec/04-tasks.md T33).

``build_model_from_config(cfg)`` builds either the production LogicNet (docs/spec/01-network.md
§3-4) or the diagnostic BaselineCNN (docs/spec/04-tasks.md T33; baseline_cnn.py), selected by
``cfg.get("modelType")`` (``"logic"``, the default, or ``"cnn-baseline"``). This lets
configs/experiments/*.json (the T33 capacity/wiring/CNN-baseline experiment matrix) request either
model with the same config schema.

Not wired into train.py yet -- train.py currently always builds ``LogicNet(ModelSpec.from_profile(
cfg["profile"], cfg["seed"]))`` directly. train.py, config.py, data_loader.py, losses.py,
distributed.py and checkpoint.py are being edited concurrently by another change; the exact diffs
this module expects on top of them (config.py: accept + validate the new keys below; train.py: call
build_model_from_config and model_spec_from_config instead of constructing LogicNet/ModelSpec
directly, and skip the wiring-matches-checkpoint resume check for cnn-baseline, which has no
wiring) are in the report accompanying this change, not applied here.

Config keys this module reads (all optional; defaults match wiring.py/model.py, and config.py's
DEFAULTS once it accepts these keys per the report's diff):
  modelType   "logic" (default) or "cnn-baseline"
  channels    override the profile's channel count (logic) / the CNN trunk width (cnn-baseline);
              default: the profile's channel count / baseline_cnn.DEFAULT_CHANNELS
  dilations   override the profile's per-layer dilation schedule (logic only); default: the
              profile's dilation list. Ignored for cnn-baseline (no gate layers).
  bank1Ratio  wiring.ModelSpec.bank1_ratio (logic only); default 0.1. Ignored for cnn-baseline.
  wiringSeed  wiring.ModelSpec.wiring_seed (logic only); default None (falls back to cfg["seed"]).
              Ignored for cnn-baseline.
  headVersion model.LogicNet/baseline_cnn.BaselineCNN head_version; default model.HEAD_VERSION (2).
"""

from __future__ import annotations

from .baseline_cnn import DEFAULT_CHANNELS, BaselineCNN
from .model import HEAD_VERSION, LogicNet
from .wiring import PROFILES, ModelSpec

MODEL_TYPES = ("logic", "cnn-baseline")


def model_spec_from_config(cfg: dict) -> ModelSpec:
    """Build the ``ModelSpec`` a ``modelType: "logic"`` config describes. Shared by
    ``build_model_from_config`` and (per the report's diff) train.py's resume-time wiring check,
    so both agree on what a config's channels/dilations/bank1Ratio/wiringSeed overrides mean."""
    profile = cfg["profile"]
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    base = PROFILES[profile]
    channels = cfg.get("channels") or base["channels"]
    dilations = cfg.get("dilations") or base["dilations"]
    return ModelSpec.custom(channels=channels, dilations=list(dilations), seed=cfg["seed"], profile=profile,
                            bank1_ratio=cfg.get("bank1Ratio", 0.1), wiring_seed=cfg.get("wiringSeed"))


def build_model_from_config(cfg: dict):
    """Returns a LogicNet or a BaselineCNN per ``cfg.get("modelType", "logic")``. ``cfg`` is a
    resolved training config dict (config.load_config's second return value, or an equivalent
    plain dict in tests) -- see the module docstring for which keys it reads."""
    model_type = cfg.get("modelType", "logic")
    head_version = cfg.get("headVersion", HEAD_VERSION)
    if model_type == "cnn-baseline":
        channels = cfg.get("channels") or DEFAULT_CHANNELS
        return BaselineCNN(channels=channels, seed=cfg["seed"], profile=cfg.get("profile"), head_version=head_version)
    if model_type != "logic":
        raise ValueError(f"unknown modelType {model_type!r}; expected one of {MODEL_TYPES}")
    spec = model_spec_from_config(cfg)
    return LogicNet(
        spec,
        head_version=head_version,
        wiring_mode=cfg.get("wiringMode", "fixed"),
        wiring_candidates=cfg.get("wiringCandidates", 8),
        wiring_tau=cfg.get("wiringTau", 1.0),
    )
