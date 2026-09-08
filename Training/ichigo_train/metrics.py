"""Validation metrics (docs/spec/05-validation.md §4) and CSV logging (T15)."""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
import torch

from . import losses as L
from .data_loader import to_tensors
from .model import LogicNet

CSV_FIELDS = ["step", "phase", "mode", "tau", "tauWire", "frozenPrefix", "headsOnly", "lrGates", "lrHeads", "gradNorm", "total", "gateEntropyLoss", "policy",
              "expected_result", "wdl", "score", "ownership", "policyTop1", "expectedMAE", "expectedBrier", "scoreMAEPoints",
              "ownershipMSE", "softHardPolicyCEDiff", "elapsedSeconds"]


class MetricsCSV:
    def __init__(self, path: str, resume: bool):
        exists = os.path.exists(path)
        self.f = open(path, "a" if resume else "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not (resume and exists):
            self.w.writeheader()

    def write(self, row: dict):
        self.w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        self.f.flush()

    def close(self):
        self.f.close()


@torch.no_grad()
def evaluate_split(model: LogicNet, arrays: dict[str, np.ndarray], board_size: int, mode: str, tau: float, frozen_prefix: int,
                   device, batch: int = 256, max_positions: int | None = None) -> dict:
    """mode 'soft' uses the relaxation (with current tau/frozen prefix); 'hard' uses argmax gates on 0/1 bits."""
    n = arrays["spatial"].shape[0] if max_positions is None else min(max_positions, arrays["spatial"].shape[0])
    sums = {k: 0.0 for k in ["total", "policy", "expected_result", "wdl", "score", "ownership"]}
    wsum = {k: 0.0 for k in ["policy", "expected_result", "wdl", "score", "ownership"]}
    top1 = mae = brier = smae = omse = 0.0
    n_pol = n_exp = n_score = n_own = 0
    logp_hard = []
    model.eval()
    for i in range(0, n, batch):
        b = to_tensors({k: v[i:i + batch] for k, v in arrays.items()}, device)
        if mode == "hard":
            out = model.forward_hard(b["spatial"], b["global"])
        else:
            out = model(b["spatial"], b["global"], tau=tau, frozen_prefix=frozen_prefix)
        r = L.compute_losses(out, b, board_size)
        for k in wsum:
            w = r["w_" + k].item()
            sums[k] += r[k].item() * w
            wsum[k] += w
        m = b["target_mask"]
        pm = m[:, 0] > 0
        if pm.any():
            neg = torch.finfo(torch.float32).min
            pred = torch.where(b["legal"] > 0, out["policy_logits"], torch.full_like(out["policy_logits"], neg)).argmax(-1)
            top1 += (pred[pm] == b["policy"][pm].argmax(-1)).float().sum().item(); n_pol += int(pm.sum())
        em = m[:, 1] > 0
        if em.any():
            q = torch.softmax(out["wdl_logits"], -1); e = q[:, 0] + 0.5 * q[:, 1]
            d = (e[em] - b["expected_result"][em])
            mae += d.abs().sum().item(); brier += (d ** 2).sum().item(); n_exp += int(em.sum())
        sm = m[:, 2] > 0
        if sm.any():
            smae += (out["score_mean"][sm] - b["score"][sm]).abs().sum().item(); n_score += int(sm.sum())
        om = m[:, 3] > 0
        if om.any():
            omse += ((out["ownership"][om] - b["ownership"][om].reshape(int(om.sum()), -1)) ** 2).mean(-1).sum().item(); n_own += int(om.sum())
    model.train()
    res = {k: (sums[k] / wsum[k] if wsum[k] > 0 else 0.0) for k in wsum}
    res["total"] = res["policy"] + res["expected_result"] + 0.5 * res["wdl"] + 0.25 * res["score"] + 0.25 * res["ownership"]
    res["policyTop1"] = top1 / n_pol if n_pol else 0.0
    res["expectedMAE"] = mae / n_exp if n_exp else 0.0
    res["expectedBrier"] = brier / n_exp if n_exp else 0.0
    res["scoreMAEPoints"] = smae / n_score if n_score else 0.0
    res["ownershipMSE"] = omse / n_own if n_own else 0.0
    res["positions"] = n
    res["mode"] = mode
    return res


@torch.no_grad()
def gate_statistics(model: LogicNet, sample_spatial: torch.Tensor | None = None) -> dict:
    theta = model.theta.detach()
    p = torch.softmax(theta, -1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)   # [L,C]
    g = model.hard_gates()
    per_layer = []
    hard_outs = None
    if sample_spatial is not None:
        hard_outs = model.hard_layers(sample_spatial)
    for l in range(theta.shape[0]):
        gl = g[l]
        d = {
            "layer": l,
            "gateEntropy": float(ent[l].mean()),
            "constantRate": float(np.mean((gl == 0) | (gl == 15))),
            "identityRate": float(np.mean((gl == 12) | (gl == 10))),
            "notRate": float(np.mean((gl == 3) | (gl == 5))),
            "thetaGradNorm": float(theta.new_tensor(0.0)) if model.theta.grad is None else float(model.theta.grad[l].norm()),
        }
        if hard_outs is not None:
            h = hard_outs[l].to(torch.float32)
            d["featureVariance"] = float(h.var(dim=(0, 1, 2)).mean())
            d["featureMean"] = float(h.mean())
        per_layer.append(d)
    return {"layers": per_layer, "gateHistogram": np.bincount(g.reshape(-1), minlength=16).tolist()}


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)
