"""Resume checkpoints (T15). Holds model/optimizer/scheduler/step/freeze state/all RNG states/
sampler position/config/dataset hash/wiring. Not readable by Swift (torch format only)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .model import LogicNet
from .wiring import ModelSpec, Wiring

CHECKPOINT_VERSION = 1


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(st: dict):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path: str, model: LogicNet, optimizer, scheduler, step: int, frozen_prefix: int, sampler_state: dict,
                    aug_rng: np.random.Generator, config: dict, dataset_hash: str, feature_version: int, extra: dict | None = None):
    payload = {
        "checkpointVersion": CHECKPOINT_VERSION,
        "spec": {"channels": model.spec.channels, "dilations": model.spec.dilations, "seed": model.spec.seed, "profile": model.spec.profile},
        "headVersion": model.head_version,
        "wiring": model.wiring_numpy(),
        "theta": model.theta.detach().cpu().numpy(),
        "heads": model.head_numpy(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "frozenPrefix": frozen_prefix,
        "sampler": sampler_state,
        "augRng": aug_rng.bit_generator.state,
        "rng": rng_state(),
        "config": config,
        "datasetHash": dataset_hash,
        "featureVersion": feature_version,
        "stateDict": model.state_dict(),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("checkpointVersion") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    return ck


def model_from_checkpoint(ck: dict):
    cfg = ck.get("config") or {}
    if cfg.get("modelType", "logic") == "cnn-baseline":
        from .model_factory import build_model_from_config
        model = build_model_from_config(cfg)
        model.load_state_dict(ck["stateDict"])
        return model
    spec = ModelSpec(**ck["spec"])
    return LogicNet(spec, Wiring(wiring=ck["wiring"], theta=ck["theta"], dilations=spec.dilations), heads=ck["heads"], head_version=ck.get("headVersion", 1))
