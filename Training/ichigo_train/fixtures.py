"""Parity fixture generation (T08/T10, `make parity-cpu`).

Layout of ``<out>/<name>/``:
  model.ichigo/           exported hard model
  inputs.json             {"boardSize","batch","spatialFile","globalFile","legalFile", ...}
  spatial.u8              uint8 [B,S,S,32]
  global.f32              float32 LE [B,4]
  legal.u8                uint8 [B,S*S+1]
  layers.u8               uint8 [L,B,S,S,C]  expected output bits of every logic layer
  expected.json           raw head outputs (policyLogits, wdlLogits, scoreMean, ownership) and
                          post-processed values, as JSON floats (float32 rounded)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import model_format as MF
from .export import export_model
from .model import LogicNet, postprocess
from .wiring import ModelSpec


def random_inputs(rng: np.random.Generator, size: int, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial = rng.integers(0, 2, size=(batch, size, size, 32), dtype=np.uint8)
    # channel 28 (constant 1) and 27 (outer ring) follow their definitions so at least those are meaningful
    spatial[..., 28] = 1
    ring = np.zeros((size, size), dtype=np.uint8)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = 1
    spatial[..., 27] = ring
    glob = np.zeros((batch, 4), dtype=np.float32)
    glob[:, 0] = rng.uniform(-0.1, 0.1, size=batch)
    glob[:, 1] = size / 19
    glob[:, 2] = rng.uniform(0, 1, size=batch)
    glob[:, 3] = rng.integers(0, 3, size=batch) / 2
    legal = rng.integers(0, 2, size=(batch, size * size + 1), dtype=np.uint8)
    legal[:, -1] = 1  # pass always legal here
    return spatial, glob, legal


def write_parity_case(out_dir: str, name: str, size: int, batch: int, seed: int, profile: str = "tiny", head_version: int = 2) -> None:
    case = os.path.join(out_dir, name)
    os.makedirs(case, exist_ok=True)
    model = LogicNet(ModelSpec.from_profile(profile, seed), head_version=head_version)
    # perturb theta with a fixed RNG so the argmax gates are diverse (not just identity/near-ties)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.theta.add_(torch.randn(model.theta.shape, generator=g) * 2.0)
    export_model(model, os.path.join(case, "model.ichigo"), [size], {"runId": f"fixture-{name}", "purpose": "parity fixture", "generatedAt": "fixed-for-reproducibility"}, overwrite=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    spatial, glob, legal = random_inputs(rng, size, batch)
    sp_t = torch.from_numpy(spatial)
    g_t = torch.from_numpy(glob)
    layers = model.hard_layers(sp_t)
    with torch.no_grad():
        raw = model.heads_forward(layers[-1].to(torch.float32), g_t)
    layer_bits = np.stack([l.numpy() for l in layers], axis=0).astype(np.uint8)
    spatial.tofile(os.path.join(case, "spatial.u8"))
    glob.astype("<f4").tofile(os.path.join(case, "global.f32"))
    legal.tofile(os.path.join(case, "legal.u8"))
    layer_bits.tofile(os.path.join(case, "layers.u8"))
    pl = raw["policy_logits"].numpy().astype(np.float32)
    wdl = raw["wdl_logits"].numpy().astype(np.float32)
    post = postprocess(pl, legal, wdl)
    expected = {
        "policyLogits": pl.tolist(),
        "wdlLogits": wdl.tolist(),
        "scoreMean": raw["score_mean"].numpy().astype(np.float32).tolist(),
        "ownership": raw["ownership"].numpy().astype(np.float32).tolist(),
        "policy": post["policy"].tolist(),
        "wdl": post["wdl"].tolist(),
        "expectedResult": post["expected_result"].tolist(),
    }
    with open(os.path.join(case, "expected.json"), "w") as f:
        json.dump(expected, f)
    with open(os.path.join(case, "inputs.json"), "w") as f:
        json.dump({
            "boardSize": size, "batch": batch, "channels": model.channels, "layers": len(model.dilations),
            "seed": seed, "profile": profile, "headVersion": head_version,
            "spatialFile": "spatial.u8", "globalFile": "global.f32", "legalFile": "legal.u8",
            "layersFile": "layers.u8", "layersLayout": "[L,B,S,S,C] uint8",
        }, f, indent=2)


def write_eval_position(path: str, size: int, spatial: np.ndarray, glob: np.ndarray, legal: np.ndarray) -> None:
    """Single-position JSON in the docs/spec/02-training.md §2 shape (only the tensors)."""
    with open(path, "w") as f:
        json.dump({"schemaVersion": 1, "boardSize": size, "spatial": spatial.reshape(-1).tolist(),
                   "global": glob.reshape(-1).tolist(), "legal": legal.reshape(-1).tolist()}, f)
