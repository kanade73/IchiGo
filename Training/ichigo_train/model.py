"""LogicNet: spatially shared logic-gate layers + FP32 heads (docs/spec/01-network.md §3–4).

Tensor layout: spatial input ``[B, S, S, 32]`` (NHWC, uint8 0/1 for hard, float for soft),
``global`` ``[B, 4]`` float32. Layer outputs are ``[B, S, S, C]``.

Outputs (all "to-move" perspective):
  policy_logits [B, S*S+1] (board points row-major y*S+x, then pass), wdl_logits [B, 3]
  (win, draw, loss), score_mean [B] (points, komi included), ownership [B, S*S] in [-1, 1].
``forward_hard`` evaluates argmax gates on exact 0/1 bits with the truth table; it is the
numerical reference the Swift backends must match bit-for-bit (layer outputs) and within
tolerance (heads).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from . import gates as G
from .wiring import INPUT_CHANNELS, ModelSpec, Wiring, generate_wiring, generate_wiring_candidates, validate_wiring

HEAD_LOCAL = 64
HEAD_GLOBAL = 128
GLOBAL_FEATURES = 4
HEAD_TENSOR_NAMES = [
    "Wlocal", "blocal", "Wpolicy", "bpolicy", "Wowner", "bowner",
    "Wglobal", "bglobal", "Wpass", "bpass", "Wwdl", "bwdl", "Wscore", "bscore",
]


HEAD_VERSION = 2
SUPPORTED_HEAD_VERSIONS = (1, 2)


def global_input_size(channels: int, head_version: int = HEAD_VERSION) -> int:
    """headVersion 1: concat(m, v, global) = 2C+4; headVersion 2: concat(m, v, zbar[64], ownMean[1], global) = 2C+69."""
    if head_version == 1:
        return 2 * channels + GLOBAL_FEATURES
    if head_version == 2:
        return 2 * channels + HEAD_LOCAL + 1 + GLOBAL_FEATURES
    raise ValueError(f"unsupported headVersion {head_version}")


def head_shapes(channels: int, head_version: int = HEAD_VERSION) -> dict[str, tuple[int, ...]]:
    C = channels
    G = global_input_size(C, head_version)
    return {
        "Wlocal": (3 * C + 4, HEAD_LOCAL), "blocal": (HEAD_LOCAL,),
        "Wpolicy": (HEAD_LOCAL, 1), "bpolicy": (1,),
        "Wowner": (HEAD_LOCAL, 1), "bowner": (1,),
        "Wglobal": (G, HEAD_GLOBAL), "bglobal": (HEAD_GLOBAL,),
        "Wpass": (HEAD_GLOBAL, 1), "bpass": (1,),
        "Wwdl": (HEAD_GLOBAL, 3), "bwdl": (3,),
        "Wscore": (HEAD_GLOBAL, 1), "bscore": (1,),
    }


def xavier_uniform(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    fan_in, fan_out = shape
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def init_heads(channels: int, rng: np.random.Generator, head_version: int = HEAD_VERSION) -> dict[str, np.ndarray]:
    """Xavier-uniform weights, zero biases, drawn in HEAD_TENSOR_NAMES order."""
    out = {}
    for name, shape in head_shapes(channels, head_version).items():
        if name.startswith("W"):
            out[name] = xavier_uniform(rng, shape)
        else:
            out[name] = np.zeros(shape, dtype=np.float32)
    return out


class LayerGather:
    """Precomputed flat gather indices for one layer and one board size.

    For a sample, bank tensors are zero-padded by ``pad`` on each side and flattened as
    ``[(S+2p)*(S+2p)*Cin]``; bank 1 (the 32 input channels) is appended after bank 0. Index
    ``idx[y, x, c, k]`` then points at reference ``k`` (A/B) of output channel ``c`` at (x, y).
    """

    def __init__(self, wiring_layer: np.ndarray, layer: int, size: int, channels: int, pad: int):
        self.size = size
        self.pad = pad
        self.c0 = INPUT_CHANNELS if layer == 0 else channels
        self.c1 = INPUT_CHANNELS
        P = size + 2 * pad
        self.padded = P
        bank0_len = P * P * self.c0
        C = wiring_layer.shape[0]
        if wiring_layer.ndim == 3:
            refs = wiring_layer[:, :, None, :]
        elif wiring_layer.ndim == 4:
            refs = wiring_layer
        else:
            raise ValueError(f"wiring layer must be [C,2,4] or [C,2,K,4], got {wiring_layer.shape}")
        K = refs.shape[2]
        idx = np.empty((size, size, C, 2, K), dtype=np.int64)
        ys = np.arange(size)[:, None] + pad
        xs = np.arange(size)[None, :] + pad
        for c in range(C):
            for k in range(2):
                for q in range(K):
                    bank, ch, dx, dy = (int(v) for v in refs[c, k, q])
                    cin = self.c0 if bank == 0 else self.c1
                    base = 0 if bank == 0 else bank0_len
                    idx[:, :, c, k, q] = base + (((ys + dy) * P + (xs + dx)) * cin + ch)
        self.k = K
        self.index = torch.from_numpy(idx)
        self.uses_bank1 = bool((refs[:, :, :, 0] == 1).any())

    def gather_candidates(self, prev: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Return candidate activations [B,S,S,C,2,K] without materialising full gate mixtures."""
        B = prev.shape[0]
        p = self.pad
        flat0 = torch.nn.functional.pad(prev, (0, 0, p, p, p, p)).reshape(B, -1)
        if self.uses_bank1:
            flat1 = torch.nn.functional.pad(inputs, (0, 0, p, p, p, p)).reshape(B, -1)
            flat = torch.cat([flat0, flat1], dim=1)
        else:
            flat = flat0
        index = self.index.to(prev.device).reshape(-1)
        return flat[:, index].reshape(B, self.size, self.size, -1, 2, self.k)

    def gather(self, prev: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather fixed references; a K=1 candidate table is reduced to A/B tensors."""
        g = self.gather_candidates(prev, inputs)
        if self.k != 1:
            raise ValueError("gather() is only valid for a fixed or K=1 wiring table")
        g = g[..., 0]
        return g[..., 0], g[..., 1]


class LogicNet(nn.Module):
    def __init__(self, spec: ModelSpec, wiring: Wiring | None = None, heads: dict[str, np.ndarray] | None = None,
                 head_version: int = HEAD_VERSION, wiring_mode: str = "fixed", wiring_candidates: int = 8,
                 candidate_wiring: np.ndarray | None = None, wiring_tau: float = 1.0):
        super().__init__()
        if head_version not in SUPPORTED_HEAD_VERSIONS:
            raise ValueError(f"unsupported headVersion {head_version}")
        if wiring_mode not in ("fixed", "learned-k"):
            raise ValueError("wiring_mode must be fixed or learned-k")
        self.head_version = head_version
        self.wiring_mode = wiring_mode
        self.wiring_tau = float(wiring_tau)
        self.spec = spec
        if wiring_mode == "learned-k" and candidate_wiring is None:
            generated = generate_wiring_candidates(spec, wiring_candidates)
            candidate_wiring = generated.candidates
            if wiring is None:
                wiring = Wiring(wiring=generated.wiring, theta=generated.theta, dilations=generated.dilations)
        if wiring is None:
            wiring = generate_wiring(spec)
        validate_wiring(wiring.wiring, spec.dilations)
        self.channels = spec.channels
        self.dilations = list(spec.dilations)
        self.register_buffer("wiring", torch.from_numpy(wiring.wiring.astype(np.int32)), persistent=True)
        if wiring_mode == "learned-k":
            candidate_wiring = np.asarray(candidate_wiring, dtype=np.int32)
            if candidate_wiring.ndim != 5 or candidate_wiring.shape[:3] != (spec.layers, spec.channels, 2) or candidate_wiring.shape[4] != 4:
                raise ValueError(f"candidate wiring must be [L,C,2,K,4], got {candidate_wiring.shape}")
            if candidate_wiring.shape[3] != wiring_candidates:
                wiring_candidates = int(candidate_wiring.shape[3])
            for k in range(candidate_wiring.shape[3]):
                validate_wiring(candidate_wiring[:, :, :, k, :], spec.dilations)
            self.register_buffer("candidate_wiring", torch.from_numpy(candidate_wiring), persistent=True)
            self.phi = nn.Parameter(torch.zeros(spec.layers, spec.channels, 2, candidate_wiring.shape[3], dtype=torch.float32))
        self.theta = nn.Parameter(torch.from_numpy(wiring.theta.astype(np.float32)))
        if heads is None:
            rng = np.random.Generator(np.random.PCG64(spec.seed + 1))
            heads = init_heads(spec.channels, rng, head_version)
        expected = head_shapes(spec.channels, head_version)
        for n in HEAD_TENSOR_NAMES:
            if tuple(np.shape(heads[n])) != expected[n]:
                raise ValueError(f"head tensor {n} has shape {np.shape(heads[n])}, expected {expected[n]} for headVersion {head_version}")
        self.heads = nn.ParameterDict({n: nn.Parameter(torch.from_numpy(np.array(heads[n], dtype=np.float32))) for n in HEAD_TENSOR_NAMES})
        self._gathers: dict[tuple[int, int, str], LayerGather] = {}

    # ----- wiring helpers -----
    def wiring_numpy(self) -> np.ndarray:
        return self.hard_wiring_numpy()

    def candidate_wiring_numpy(self) -> np.ndarray | None:
        if self.wiring_mode != "learned-k":
            return None
        return self.candidate_wiring.detach().cpu().numpy().astype(np.int32)

    def hard_wiring_numpy(self) -> np.ndarray:
        """Return the ordinary fixed wiring used by hard forward and export."""
        if self.wiring_mode != "learned-k":
            return self.wiring.detach().cpu().numpy().astype(np.int32)
        candidates = self.candidate_wiring_numpy()
        logits = self.phi.detach().cpu().numpy()
        out = np.empty(candidates.shape[:3] + (4,), dtype=np.int32)
        adjusted = 0
        for l in range(candidates.shape[0]):
            for c in range(candidates.shape[1]):
                ai = int(np.argmax(logits[l, c, 0]))
                a = candidates[l, c, 0, ai]
                order = np.argsort(-logits[l, c, 1], kind="stable")
                bi = next((int(i) for i in order if not np.array_equal(candidates[l, c, 1, i], a)), None)
                if bi is None:
                    raise ValueError(f"layer {l} channel {c}: no B candidate distinct from selected A")
                if np.array_equal(candidates[l, c, 1, int(order[0])], a):
                    adjusted += 1
                out[l, c, 0] = a
                out[l, c, 1] = candidates[l, c, 1, bi]
        if adjusted:
            import logging
            logging.getLogger(__name__).warning("learned wiring adjusted %d A==B argmax selections", adjusted)
        validate_wiring(out, self.dilations)
        return out

    def hard_gates(self) -> np.ndarray:
        """``uint8 [L, C]`` argmax gate ids (smallest id on ties)."""
        return G.argmax_gate(self.theta)

    def head_numpy(self) -> dict[str, np.ndarray]:
        return {n: self.heads[n].detach().cpu().numpy().astype(np.float32) for n in HEAD_TENSOR_NAMES}

    def gather_for(self, layer: int, size: int, hard: bool = False, hard_wiring: np.ndarray | None = None) -> LayerGather:
        if hard and self.wiring_mode == "learned-k":
            refs = self.hard_wiring_numpy() if hard_wiring is None else hard_wiring
            return LayerGather(refs[layer], layer, size, self.channels, self.dilations[layer])
        kind = "soft" if self.wiring_mode == "learned-k" else "fixed"
        key = (layer, size, kind)
        if key not in self._gathers:
            refs = self.candidate_wiring_numpy()[layer] if kind == "soft" else self.wiring.detach().cpu().numpy()[layer]
            self._gathers[key] = LayerGather(refs, layer, size, self.channels, self.dilations[layer])
        return self._gathers[key]

    @staticmethod
    def _check_inputs(spatial: torch.Tensor, glob: torch.Tensor) -> int:
        if spatial.ndim != 4 or spatial.shape[1] != spatial.shape[2] or spatial.shape[3] != INPUT_CHANNELS:
            raise ValueError(f"spatial must be [B,S,S,32], got {tuple(spatial.shape)}")
        if glob.ndim != 2 or glob.shape[1] != GLOBAL_FEATURES or glob.shape[0] != spatial.shape[0]:
            raise ValueError(f"global must be [B,4], got {tuple(glob.shape)}")
        return spatial.shape[1]

    # ----- gate layers -----
    def soft_layers(self, spatial: torch.Tensor, tau: float, frozen_prefix: int = 0,
                    gumbel_noise: torch.Tensor | None = None, tau_wire: float | None = None) -> list[torch.Tensor]:
        """Returns all layer outputs (float). Layers < frozen_prefix use hard argmax gates on the
        (already 0/1) activations; the rest use the softmax relaxation."""
        size = spatial.shape[1]
        x = spatial.to(torch.float32)
        outs = []
        hard = None
        hard_wiring = None
        if frozen_prefix > 0:
            hard = torch.from_numpy(self.hard_gates()).to(spatial.device)
            if self.wiring_mode == "learned-k":
                hard_wiring = self.hard_wiring_numpy()
        if gumbel_noise is not None:
            if tuple(gumbel_noise.shape) != tuple(self.theta.shape):
                raise ValueError(f"gumbel_noise must have shape {tuple(self.theta.shape)}, got {tuple(gumbel_noise.shape)}")
            gumbel_noise = gumbel_noise.to(self.theta.device, dtype=self.theta.dtype)
        wire_tau = self.wiring_tau if tau_wire is None else tau_wire
        if wire_tau <= 0:
            raise ValueError("tau_wire must be positive")
        for l in range(len(self.dilations)):
            is_hard_layer = l < frozen_prefix
            gath = self.gather_for(l, size, hard=is_hard_layer, hard_wiring=hard_wiring)
            if self.wiring_mode == "learned-k" and not is_hard_layer:
                candidates = gath.gather_candidates(x, spatial.to(torch.float32))
                weights = torch.softmax(self.phi[l] / wire_tau, dim=-1).view(1, 1, 1, self.channels, 2, -1)
                mixed = (candidates * weights).sum(-1)
                a, b = mixed[..., 0], mixed[..., 1]
            else:
                a, b = gath.gather(x, spatial.to(torch.float32))
            if is_hard_layer:
                g = hard[l].to(torch.int64)  # [C]
                row = (2 * a.to(torch.int64) + b.to(torch.int64))
                y = ((g.view(1, 1, 1, -1) >> row) & 1).to(torch.float32)
            else:
                if gumbel_noise is None:
                    t = G.reduce_theta(self.theta[l], tau)  # [C,4]
                else:
                    p = G.gate_probabilities(self.theta[l] + gumbel_noise[l], tau)
                    h = torch.nn.functional.one_hot(p.argmax(dim=-1), num_classes=G.NUM_GATES).to(p.dtype)
                    p = h - p.detach() + p
                    t = p @ G.truth_table_tensor(dtype=p.dtype, device=p.device)
                y = G.soft_gate_reduced(t.view(1, 1, 1, -1, 4), a, b)
            outs.append(y)
            x = y
        return outs

    def hard_layers(self, spatial: torch.Tensor) -> list[torch.Tensor]:
        """Exact 0/1 evaluation with argmax gates. ``spatial`` uint8 [B,S,S,32]."""
        if spatial.dtype != torch.uint8:
            raise TypeError("hard forward requires uint8 spatial input")
        vals = spatial.unique()
        if not bool(((vals == 0) | (vals == 1)).all()):
            raise ValueError("hard forward requires spatial values in {0,1}")
        size = spatial.shape[1]
        gates = torch.from_numpy(self.hard_gates()).to(spatial.device)
        hard_wiring = self.hard_wiring_numpy() if self.wiring_mode == "learned-k" else None
        x = spatial
        outs = []
        for l in range(len(self.dilations)):
            gath = self.gather_for(l, size, hard=True, hard_wiring=hard_wiring)
            a, b = gath.gather(x, spatial)
            row = 2 * a.to(torch.int32) + b.to(torch.int32)
            y = ((gates[l].to(torch.int32).view(1, 1, 1, -1) >> row) & 1).to(torch.uint8)
            outs.append(y)
            x = y
        return outs

    # ----- heads -----
    def heads_forward(self, h: torch.Tensor, glob: torch.Tensor) -> dict[str, torch.Tensor]:
        """``h`` float [B,S,S,C] (last logic layer), ``glob`` [B,4]."""
        H = self.heads
        B, S, _, C = h.shape
        m = h.mean(dim=(1, 2))  # [B,C]
        v = h.amax(dim=(1, 2))  # [B,C]
        local_stats = torch.cat([m, v, glob], dim=1)  # [B, 2C+4]
        u = torch.cat([h, local_stats.view(B, 1, 1, -1).expand(B, S, S, 2 * C + 4)], dim=3)  # [B,S,S,3C+4]
        z = torch.relu(u @ H["Wlocal"] + H["blocal"])
        policy_xy = (z @ H["Wpolicy"] + H["bpolicy"]).reshape(B, S * S)
        ownership = torch.tanh(z @ H["Wowner"] + H["bowner"]).reshape(B, S * S)
        if self.head_version == 1:
            g_stats = local_stats
        else:
            zbar = z.mean(dim=(1, 2))                       # [B,64]
            own_mean = ownership.mean(dim=1, keepdim=True)  # [B,1]
            g_stats = torch.cat([m, v, zbar, own_mean, glob], dim=1)  # [B, 2C+69]
        zg = torch.relu(g_stats @ H["Wglobal"] + H["bglobal"])
        pass_logit = zg @ H["Wpass"] + H["bpass"]  # [B,1]
        wdl = zg @ H["Wwdl"] + H["bwdl"]
        score = (zg @ H["Wscore"] + H["bscore"]).reshape(B)
        return {
            "policy_logits": torch.cat([policy_xy, pass_logit], dim=1),
            "wdl_logits": wdl,
            "score_mean": score,
            "ownership": ownership,
        }

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor, tau: float = 1.0, frozen_prefix: int = 0,
                gumbel_noise: torch.Tensor | None = None, tau_wire: float | None = None) -> dict[str, torch.Tensor]:
        self._check_inputs(spatial, glob)
        outs = self.soft_layers(spatial, tau, frozen_prefix, gumbel_noise=gumbel_noise, tau_wire=tau_wire)
        return self.heads_forward(outs[-1], glob.to(torch.float32))

    @torch.no_grad()
    def forward_hard(self, spatial: torch.Tensor, glob: torch.Tensor) -> dict[str, torch.Tensor]:
        self._check_inputs(spatial, glob)
        outs = self.hard_layers(spatial)
        return self.heads_forward(outs[-1].to(torch.float32), glob.to(torch.float32))


def postprocess(policy_logits: np.ndarray, legal: np.ndarray, wdl_logits: np.ndarray) -> dict[str, np.ndarray]:
    """Stable masked softmax (docs/spec/03-engine.md §3). Illegal moves get exactly 0.
    Raises if a row has no legal move or logits are non-finite."""
    if not (np.isfinite(policy_logits).all() and np.isfinite(wdl_logits).all()):
        raise ValueError("non-finite logits")
    legal = legal.astype(bool)
    if not legal.any(axis=1).all():
        raise ValueError("a position has no legal move")
    masked = np.where(legal, policy_logits.astype(np.float64), -np.inf)
    mx = masked.max(axis=1, keepdims=True)
    e = np.where(legal, np.exp(masked - mx), 0.0)
    policy = e / e.sum(axis=1, keepdims=True)
    w = wdl_logits.astype(np.float64)
    w = np.exp(w - w.max(axis=1, keepdims=True))
    wdl = w / w.sum(axis=1, keepdims=True)
    expected = wdl[:, 0] + 0.5 * wdl[:, 1]
    return {"policy": policy.astype(np.float32), "wdl": wdl.astype(np.float32), "expected_result": expected.astype(np.float32)}


def build_model(profile: str, seed: int = 20260908, head_version: int = HEAD_VERSION) -> LogicNet:
    return LogicNet(ModelSpec.from_profile(profile, seed), head_version=head_version)
