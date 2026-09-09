"""Two-input gates and experimental four-input LUT gates (docs/spec/01-network.md §2–2b).

Gate id ``g`` in 0..15 *is* the truth table: for inputs ``(a, b)`` the row index is
``i = 2*a + b`` and the output is ``(g >> i) & 1``. Therefore
false=0, AND=8, XOR=6, OR=14, a=12, b=10, NOT a=3, NAND=7, true=15.

The continuous relaxation for a, b in [0, 1] is

    q = [(1-a)(1-b), (1-a)b, a(1-b), ab]
    f_g(a, b) = sum_i bit(g, i) * q_i
    p = softmax(theta / tau)              (16 probabilities)
    y_soft = sum_g p_g * f_g(a, b) = sum_i t_i * q_i,   t_i = sum_g p_g * bit(g, i)

Both forms are the same polynomial; the reduced ``[.., 4]`` form is what training uses.

For the LUT4 path, ``phi[..., i]`` is an independent logit for table row ``i`` and
``t[..., i] = sigmoid(phi[..., i] / tau)``. Rows use MSB-first input order, so the shared
reduced evaluator builds ``q_i`` for ``i = sum_k a_k * 2**(n-1-k)``.
"""

from __future__ import annotations

import numpy as np
import torch

NUM_GATES = 16
GATE_FALSE, GATE_AND, GATE_XOR, GATE_OR = 0, 8, 6, 14
GATE_A, GATE_B, GATE_NOT_A, GATE_NAND, GATE_TRUE = 12, 10, 3, 7, 15


def truth_table() -> np.ndarray:
    """``[16, 4] uint8``: ``table[g, 2*a+b] = (g >> (2*a+b)) & 1``."""
    g = np.arange(NUM_GATES, dtype=np.uint8)[:, None]
    i = np.arange(4, dtype=np.uint8)[None, :]
    return ((g >> i) & 1).astype(np.uint8)


def hard_gate(g: int, a: int, b: int) -> int:
    """Evaluate gate ``g`` on discrete bits. Inputs must be exactly 0 or 1."""
    if a not in (0, 1) or b not in (0, 1) or not 0 <= g < NUM_GATES:
        raise ValueError(f"hard_gate expects bits and g in 0..15, got g={g}, a={a}, b={b}")
    return (g >> (2 * a + b)) & 1


def hard_gate_array(g: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised ``hard_gate``. ``g`` broadcasts against ``a``/``b`` (all uint8, values 0/1)."""
    g = np.asarray(g, dtype=np.uint8)
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return ((g >> (2 * a + b)) & 1).astype(np.uint8)


def truth_table_tensor(dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
    """``[16, 4]`` float tensor of the truth table (bit(g, i))."""
    return torch.tensor(truth_table(), dtype=dtype, device=device)


def gate_probabilities(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """``softmax(theta / tau)`` over the last (16) axis, computed in the dtype of ``theta``."""
    if theta.shape[-1] != NUM_GATES:
        raise ValueError(f"theta last dim must be 16, got {tuple(theta.shape)}")
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.softmax(theta / tau, dim=-1)


def reduce_theta(theta: torch.Tensor, tau: float) -> torch.Tensor:
    """``t[..., i] = sum_g p_g * bit(g, i)`` -> shape ``[..., 4]``."""
    p = gate_probabilities(theta, tau)
    table = truth_table_tensor(dtype=p.dtype, device=p.device)
    return p @ table


def gate_entropy(theta: torch.Tensor, tau: float, gate_arity: int = 2) -> torch.Tensor:
    """Mean gate entropy; arity 2 keeps softmax entropy, LUT4 uses Bernoulli entries."""
    if theta.numel() == 0:
        return theta.sum() * 0.0
    if gate_arity == 4:
        p = lut_probabilities(theta, tau)
        return -(p * torch.log(p.clamp_min(1e-12)) + (1 - p) * torch.log((1 - p).clamp_min(1e-12))).mean()
    if gate_arity != 2:
        raise ValueError("gate_arity must be 2 or 4")
    p = gate_probabilities(theta, tau)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()


def gate_entropy_loss(theta: torch.Tensor, tau: float, weight: float, gate_arity: int = 2) -> torch.Tensor:
    """Weighted gate entropy regularizer; returns an attached zero when ``weight == 0``."""
    if weight == 0:
        return theta.sum() * 0.0
    return torch.as_tensor(weight, dtype=theta.dtype, device=theta.device) * gate_entropy(theta, tau, gate_arity=gate_arity)


def lut_probabilities(phi: torch.Tensor, tau: float) -> torch.Tensor:
    """Independent LUT table probabilities ``t[..., i] = sigmoid(phi[..., i] / tau)``."""
    if phi.shape[-1] != NUM_GATES:
        raise ValueError(f"LUT logits last dim must be 16, got {tuple(phi.shape)}")
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.sigmoid(phi / tau)


def _normalise_lut_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) not in (2, 4):
        raise ValueError(f"LUT input count must be 2 or 4, got {len(inputs)}")
    return tuple(inputs)


def lut_row_products(*inputs: torch.Tensor) -> torch.Tensor:
    """Return ``q[..., i]`` for all MSB-first rows, with ``2**n`` final terms."""
    inputs = _normalise_lut_inputs(tuple(inputs))
    first = inputs[0]
    one = torch.ones((), dtype=first.dtype, device=first.device)
    rows = []
    n = len(inputs)
    for i in range(1 << n):
        q = one
        for k, a in enumerate(inputs):
            q = q * (a if ((i >> (n - 1 - k)) & 1) else (one - a))
        rows.append(q)
    return torch.stack(rows, dim=-1)


def soft_gate_lut_reduced(t: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate ``sum_i t_i * q_i`` for a 2- or 4-input LUT in MSB-first order."""
    inputs = _normalise_lut_inputs(tuple(inputs))
    expected = 1 << len(inputs)
    if t.shape[-1] != expected:
        raise ValueError(f"LUT table last dim must be {expected}, got {tuple(t.shape)}")
    return (t * lut_row_products(*inputs)).sum(dim=-1)


def soft_lut_direct(phi: torch.Tensor, tau: float, *inputs: torch.Tensor) -> torch.Tensor:
    """Reference LUT expression, retaining all table entries before the row reduction."""
    return (lut_probabilities(phi, tau) * lut_row_products(*inputs)).sum(dim=-1)


def hard_lut(phi: torch.Tensor | np.ndarray) -> np.ndarray:
    """Encode ``[phi_i > 0]`` as a uint16 table word, bit ``i`` = row ``i``."""
    arr = phi.detach().cpu().numpy() if isinstance(phi, torch.Tensor) else np.asarray(phi)
    if arr.shape[-1] != NUM_GATES:
        raise ValueError("LUT logits last dim must be 16")
    if not np.all(np.isfinite(arr)):
        raise ValueError("LUT logits contain non-finite values")
    bits = (arr > 0).astype(np.uint16)
    weights = (np.uint16(1) << np.arange(NUM_GATES, dtype=np.uint16)).reshape((1,) * (arr.ndim - 1) + (NUM_GATES,))
    return np.sum(bits * weights, axis=-1, dtype=np.uint16)


lut4_soft_reduced = soft_gate_lut_reduced
lut4_soft_direct = soft_lut_direct
lut4_hard = hard_lut


def soft_gate_reduced(t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_i t_i * q_i`` with ``t`` of shape ``[..., 4]`` broadcast against ``a``/``b``."""
    one = torch.ones((), dtype=a.dtype, device=a.device)
    q00 = (one - a) * (one - b)
    q01 = (one - a) * b
    q10 = a * (one - b)
    q11 = a * b
    return t[..., 0] * q00 + t[..., 1] * q01 + t[..., 2] * q10 + t[..., 3] * q11


def soft_gate_direct(theta: torch.Tensor, tau: float, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference: explicit 16-term sum ``sum_g p_g f_g(a, b)``. Used only for tests."""
    p = gate_probabilities(theta, tau)  # [..., 16]
    table = truth_table_tensor(dtype=p.dtype, device=p.device)  # [16, 4]
    one = torch.ones((), dtype=a.dtype, device=a.device)
    q = torch.stack([(one - a) * (one - b), (one - a) * b, a * (one - b), a * b], dim=-1)  # [..., 4]
    f = q @ table.T  # [..., 16]  f_g(a,b)
    return (p * f).sum(-1)


def argmax_gate(theta: torch.Tensor | np.ndarray) -> np.ndarray:
    """Hard gate id: argmax over the 16 logits, smallest id on ties. Returns uint8."""
    arr = theta.detach().cpu().numpy() if isinstance(theta, torch.Tensor) else np.asarray(theta)
    if arr.shape[-1] != NUM_GATES:
        raise ValueError("theta last dim must be 16")
    if not np.all(np.isfinite(arr)):
        raise ValueError("theta contains non-finite values")
    # np.argmax returns the first (smallest index) maximum.
    return np.argmax(arr, axis=-1).astype(np.uint8)
