"""16 two-input logic gates (docs/spec/01-network.md §2).

Gate id ``g`` in 0..15 *is* the truth table: for inputs ``(a, b)`` the row index is
``i = 2*a + b`` and the output is ``(g >> i) & 1``. Therefore
false=0, AND=8, XOR=6, OR=14, a=12, b=10, NOT a=3, NAND=7, true=15.

The continuous relaxation for a, b in [0, 1] is

    q = [(1-a)(1-b), (1-a)b, a(1-b), ab]
    f_g(a, b) = sum_i bit(g, i) * q_i
    p = softmax(theta / tau)              (16 probabilities)
    y_soft = sum_g p_g * f_g(a, b) = sum_i t_i * q_i,   t_i = sum_g p_g * bit(g, i)

Both forms are the same polynomial; the reduced ``[.., 4]`` form is what training uses.
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
