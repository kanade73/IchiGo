"""Experimental aggregation layers placed after chosen gate layers (phase 20).

Question under test: does adding many-input aggregation to the gate network improve it? Each
layer replaces the first ``channels``/``nodes`` channels of a gate layer's output, so channel
counts, wiring and heads stay unchanged.

``component-or``
    OR of a channel over a 4-connected component of same-state points (a chain of stones, or an
    empty region). Components come from input channels 0/1. ``scope`` picks which components are
    pooled; points outside it pass through unchanged:
      ``chains``      stone chains only (default)
      ``chains+eyes`` stone chains and empty regions of at most ``maxRegion`` points
      ``all``         every component (large opening regions saturate: avoid for training)
    ``relaxation`` is the soft (training) form: ``prob`` = 1 - prod(1 - x) (the OR gate's
    relaxation; saturates on large components) or ``max`` (gradient to the largest input). Hard
    evaluation is the exact OR for both.

``threshold``
    BNN-style threshold nodes. Node ``j`` reads ``fanIn`` fixed references (channel, dx, dy) in the
    3x3 neighbourhood of the layer output (zero outside the board), each optionally negated, and
    fires when at least ``threshold_j`` of them are 1. ``weights``: ``tanh`` uses tanh(w) in the
    soft forward (hard uses sign(w), so the two can disagree); ``ste`` uses sign(w) in the forward
    and tanh's gradient in the backward (straight-through), so soft and hard differ only in the
    output sigmoid. ``outputSte`` additionally makes the forward output a hard step (sigmoid
    gradient in the backward).

Neither is in the ``.ichigo`` format yet: export refuses models that use them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TYPES = ("component-or", "threshold")
SCOPES = ("chains", "chains+eyes", "all")
RELAXATIONS = ("prob", "max")
WEIGHT_MODES = ("tanh", "ste")


def _int(v, lo, hi):
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def normalize(cfg: dict | None, layers: int, channels: int) -> dict | None:
    """Validates an ``aggregation`` config entry; returns it with defaults filled (or None)."""
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError("aggregation must be an object")
    kind = cfg.get("type")
    if kind not in TYPES:
        raise ValueError(f"aggregation.type must be one of {TYPES}")
    after = cfg.get("afterLayers")
    if not isinstance(after, list) or not after or len(set(after)) != len(after) or not all(_int(l, 0, layers - 1) for l in after):
        raise ValueError(f"aggregation.afterLayers must be distinct layer indices in [0, {layers})")
    out = {"type": kind, "afterLayers": sorted(after)}
    if kind == "component-or":
        out.update(channels=cfg.get("channels", 32), scope=cfg.get("scope", "chains"), maxRegion=cfg.get("maxRegion", 8),
                   relaxation=cfg.get("relaxation", "prob"))
        if not _int(out["channels"], 1, channels):
            raise ValueError(f"aggregation.channels must be in [1, {channels}]")
        if out["scope"] not in SCOPES:
            raise ValueError(f"aggregation.scope must be one of {SCOPES}")
        if not _int(out["maxRegion"], 1, 361):
            raise ValueError("aggregation.maxRegion must be in [1, 361]")
        if out["relaxation"] not in RELAXATIONS:
            raise ValueError(f"aggregation.relaxation must be one of {RELAXATIONS}")
    else:
        out.update(nodes=cfg.get("nodes", 32), fanIn=cfg.get("fanIn", 16), seed=cfg.get("seed", 20260924),
                   weights=cfg.get("weights", "ste"), outputSte=cfg.get("outputSte", False))
        if not _int(out["nodes"], 1, channels):
            raise ValueError(f"aggregation.nodes must be in [1, {channels}]")
        if not _int(out["fanIn"], 1, 9 * channels):
            raise ValueError(f"aggregation.fanIn must be in [1, {9 * channels}]")
        if not _int(out["seed"], 0, 2 ** 62):
            raise ValueError("aggregation.seed must be a non-negative integer")
        if out["weights"] not in WEIGHT_MODES:
            raise ValueError(f"aggregation.weights must be one of {WEIGHT_MODES}")
        if not isinstance(out["outputSte"], bool):
            raise ValueError("aggregation.outputSte must be a boolean")
    extra = set(cfg) - set(out)
    if extra:
        raise ValueError(f"unknown aggregation keys: {sorted(extra)}")
    return out


# ---------------------------------------------------------------- components
def component_labels(spatial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (labels [B,P] int64: the smallest point index of each point's component,
    sizes [B,P]: that component's size, state [B,P]: +1 own stone, -1 opponent stone, 0 empty).
    Min-label propagation over same-state neighbours, O(P x component diameter)."""
    B, S = spatial.shape[0], spatial.shape[1]
    P = S * S
    dev = spatial.device
    state = (spatial[..., 0].to(torch.int8) - spatial[..., 1].to(torch.int8)).reshape(B, S, S)
    same_r = state[:, :, :-1] == state[:, :, 1:]   # (y,x) ~ (y,x+1)
    same_d = state[:, :-1, :] == state[:, 1:, :]   # (y,x) ~ (y+1,x)
    lab = torch.arange(P, device=dev).reshape(1, S, S).expand(B, S, S).clone()
    big = torch.iinfo(torch.int64).max
    while True:
        new = lab.clone()
        new[:, :, :-1] = torch.minimum(new[:, :, :-1], torch.where(same_r, lab[:, :, 1:], big))
        new[:, :, 1:] = torch.minimum(new[:, :, 1:], torch.where(same_r, lab[:, :, :-1], big))
        new[:, :-1, :] = torch.minimum(new[:, :-1, :], torch.where(same_d, lab[:, 1:, :], big))
        new[:, 1:, :] = torch.minimum(new[:, 1:, :], torch.where(same_d, lab[:, :-1, :], big))
        if torch.equal(new, lab):
            break
        lab = new
    lab = lab.reshape(B, P)
    sizes = torch.zeros(B, P, dtype=torch.int64, device=dev).scatter_add_(1, lab, torch.ones_like(lab))
    return lab, sizes.gather(1, lab), state.reshape(B, P).to(torch.int64)


def pooled_mask(labels, sizes, state, scope: str, max_region: int) -> torch.Tensor:
    if scope == "all":
        return torch.ones_like(labels, dtype=torch.bool)
    stones = state != 0
    if scope == "chains":
        return stones
    return stones | (sizes <= max_region)


def component_or(x: torch.Tensor, labels: torch.Tensor, member: torch.Tensor, hard: bool, relaxation: str = "prob") -> torch.Tensor:
    """``x`` [B,S,S,k] in [0,1]. Points with ``member`` get the OR over their component; the rest
    keep ``x``."""
    B, S, _, k = x.shape
    P = S * S
    flat = x.reshape(B, P, k).to(torch.float32)
    idx = labels.unsqueeze(-1).expand(B, P, k)
    if hard or relaxation == "max":
        red = torch.full((B, P, k), -1.0, device=x.device).scatter_reduce(1, idx, flat, reduce="amax", include_self=True)
        pooled = red.gather(1, idx)
    else:
        logs = torch.log(torch.clamp(1.0 - flat, min=1e-6))
        pooled = 1.0 - torch.exp(torch.zeros(B, P, k, device=x.device).scatter_add(1, idx, logs).gather(1, idx))
    out = torch.where(member.unsqueeze(-1), pooled, flat)
    return out.reshape(B, S, S, k)


# ---------------------------------------------------------------- threshold nodes
class ThresholdLayer(nn.Module):
    def __init__(self, channels: int, nodes: int, fan_in: int, seed: int, weights: str = "ste", output_ste: bool = False):
        super().__init__()
        rng = np.random.default_rng(seed)
        ch = rng.integers(0, channels, size=(nodes, fan_in))
        off = rng.integers(0, 9, size=(nodes, fan_in))
        refs = np.stack([ch, off % 3 - 1, off // 3 - 1], axis=-1).astype(np.int64)
        self.register_buffer("refs", torch.from_numpy(refs), persistent=True)  # [nodes, fanIn, 3]
        self.weight = nn.Parameter(torch.from_numpy(rng.normal(0.0, 1.0, size=(nodes, fan_in)).astype(np.float32)))
        # half-integer start: no ties between the integer count and the threshold
        self.threshold = nn.Parameter(torch.full((nodes,), fan_in // 2 + 0.5, dtype=torch.float32))
        self.channels, self.nodes, self.fan_in = channels, nodes, fan_in
        self.weights_mode, self.output_ste = weights, output_ste
        self._index: dict[tuple[int, str], torch.Tensor] = {}

    def _gather(self, y: torch.Tensor) -> torch.Tensor:
        B, S, _, C = y.shape
        key = (S, str(y.device))
        if key not in self._index:
            refs = self.refs.cpu().numpy()
            P = S + 2
            ys = np.arange(S)[:, None, None, None] + 1
            xs = np.arange(S)[None, :, None, None] + 1
            idx = ((ys + refs[None, None, :, :, 2]) * P + (xs + refs[None, None, :, :, 1])) * C + refs[None, None, :, :, 0]
            self._index[key] = torch.from_numpy(idx.reshape(-1)).to(y.device)
        flat = torch.nn.functional.pad(y, (0, 0, 1, 1, 1, 1)).reshape(B, -1)
        return flat[:, self._index[key]].reshape(B, S, S, self.nodes, self.fan_in)

    def signs(self) -> torch.Tensor:
        t = torch.tanh(self.weight)
        if self.weights_mode == "tanh":
            return t
        hard = torch.where(self.weight > 0, 1.0, -1.0)
        return hard + t - t.detach()

    def forward(self, y: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
        g = self._gather(y.to(torch.float32))
        if hard:
            sign = (self.weight > 0).view(1, 1, 1, self.nodes, self.fan_in)
            total = torch.where(sign, g, 1.0 - g).sum(-1)
            return (total >= self.threshold.detach().view(1, 1, 1, -1)).to(torch.float32)
        s = self.signs().view(1, 1, 1, self.nodes, self.fan_in)
        total = (0.5 + s * (g - 0.5)).sum(-1)
        z = total - self.threshold.view(1, 1, 1, -1)
        soft = torch.sigmoid(z / tau)
        if not self.output_ste:
            return soft
        return (z >= 0).to(soft.dtype) + soft - soft.detach()

    def hard_parameters(self) -> dict[str, np.ndarray]:
        """What an inference backend needs: refs, negate flags, integer thresholds."""
        return {"refs": self.refs.cpu().numpy().astype(np.int32),
                "negate": (self.weight.detach().cpu().numpy() <= 0),
                "threshold": np.ceil(self.threshold.detach().cpu().numpy()).astype(np.int32)}


class Aggregation(nn.Module):
    def __init__(self, cfg: dict, channels: int):
        super().__init__()
        self.cfg = cfg
        self.after = set(cfg["afterLayers"])
        self.capture: dict | None = None  # diagnostics: set to {} to record inputs/outputs per layer
        self.layers = nn.ModuleDict()
        if cfg["type"] == "threshold":
            for i, l in enumerate(cfg["afterLayers"]):
                self.layers[str(l)] = ThresholdLayer(channels, cfg["nodes"], cfg["fanIn"], cfg["seed"] + 1000 * i,
                                                     cfg["weights"], cfg["outputSte"])

    def apply(self, layer: int, y: torch.Tensor, spatial: torch.Tensor, cache: dict, tau: float, hard: bool) -> torch.Tensor:
        """``y`` is the gate layer output (float 0/1 when ``hard``); returns the replaced output."""
        if layer not in self.after:
            return y
        if self.capture is not None and y.requires_grad:
            y.retain_grad()
        if self.cfg["type"] == "component-or":
            if "labels" not in cache:
                labels, sizes, state = component_labels(spatial)
                cache["labels"] = labels
                cache["member"] = pooled_mask(labels, sizes, state, self.cfg["scope"], self.cfg["maxRegion"])
            k = self.cfg["channels"]
            pooled = component_or(y[..., :k], cache["labels"], cache["member"], hard, self.cfg["relaxation"])
            if self.capture is not None:
                self.capture[layer] = {"in": y, "out": pooled, "labels": cache["labels"], "member": cache["member"]}
            return torch.cat([pooled.to(y.dtype), y[..., k:]], dim=-1)
        t = self.layers[str(layer)](y, tau, hard)
        if self.capture is not None:
            self.capture[layer] = {"in": y, "out": t}
        n = self.cfg["nodes"]
        return torch.cat([t.to(y.dtype), y[..., n:]], dim=-1)
