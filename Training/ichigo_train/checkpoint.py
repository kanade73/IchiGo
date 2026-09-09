"""Resume checkpoints (T15). Holds model/optimizer/scheduler/step/freeze state/all RNG states/
sampler position/config/dataset hash/wiring. Not readable by Swift (torch format only)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .model import LogicNet
from .wiring import ModelSpec, Wiring

CHECKPOINT_VERSION = 2


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
                    aug_rng: np.random.Generator, config: dict, dataset_hash: str, feature_version: int, extra: dict | None = None,
                    gumbel_rng: torch.Generator | None = None):
    mode = getattr(model, "wiring_mode", "fixed")
    effective_wiring = model.wiring_numpy()
    base_wiring = model.wiring.detach().cpu().numpy().astype(np.int32) if hasattr(model, "wiring") else effective_wiring
    candidate_table = model.candidate_wiring_numpy() if hasattr(model, "candidate_wiring_numpy") else None
    payload = {
        "checkpointVersion": CHECKPOINT_VERSION,
        "spec": {"channels": model.spec.channels, "dilations": model.spec.dilations, "seed": model.spec.seed, "profile": model.spec.profile,
                 "bank1_ratio": model.spec.bank1_ratio, "wiring_seed": model.spec.wiring_seed,
                 "gate_arity": model.spec.gate_arity},
        "headVersion": model.head_version,
        "wiring": effective_wiring,
        "baseWiring": base_wiring,
        "wiringMode": mode,
        "wiringCandidates": candidate_table,
        "wiringTau": getattr(model, "wiring_tau", 1.0),
        "phi": model.phi.detach().cpu().numpy() if mode == "learned-k" else None,
        "theta": model.theta.detach().cpu().numpy(),
        "heads": model.head_numpy(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "frozenPrefix": frozen_prefix,
        "sampler": sampler_state,
        "augRng": aug_rng.bit_generator.state,
        "rng": rng_state(),
        "gumbelRng": gumbel_rng.get_state() if gumbel_rng is not None else None,
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
    if ck.get("checkpointVersion") not in (1, CHECKPOINT_VERSION):
        raise ValueError("unsupported checkpoint version")
    return ck


def model_from_checkpoint(ck: dict):
    cfg = ck.get("config") or {}
    if cfg.get("modelType", "logic") == "cnn-baseline":
        from .model_factory import build_model_from_config
        model = build_model_from_config(cfg)
        model.load_state_dict(ck["stateDict"])
        return model
    spec_data = dict(ck["spec"])
    spec_data.setdefault("bank1_ratio", 0.1)
    spec_data.setdefault("wiring_seed", None)
    spec = ModelSpec(**spec_data)
    mode = ck.get("wiringMode", cfg.get("wiringMode", "fixed"))
    candidates = ck.get("wiringCandidates")
    candidate_count = int(candidates.shape[3]) if candidates is not None else int(cfg.get("wiringCandidates", 8))
    wiring = ck.get("baseWiring", ck["wiring"])
    model = LogicNet(
        spec,
        Wiring(wiring=wiring, theta=ck["theta"], dilations=spec.dilations),
        heads=ck["heads"],
        head_version=ck.get("headVersion", 1),
        wiring_mode=mode,
        wiring_candidates=candidate_count,
        candidate_wiring=candidates,
        wiring_tau=ck.get("wiringTau", cfg.get("wiringTau", 1.0)),
    )
    # stateDict is authoritative for torch buffers/parameters. The explicit arrays above keep
    # checkpoints inspectable and permit loading older minimal payloads.
    if "stateDict" in ck:
        model.load_state_dict(ck["stateDict"])
    return model
