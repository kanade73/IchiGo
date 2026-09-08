"""D4 board symmetries (docs/spec/01-network.md §1).

``R(x, y) = (S-1-y, x)`` (rotation), ``F(x, y) = (S-1-x, y)`` (flip).
id 0..3 = R^id, id 4..7 = R^(id-4) ∘ F (flip first, then rotate).
``forward[id][p]`` is the destination index of point ``p = y*S + x``; a transformed plane ``T``
satisfies ``T[forward[p]] = X[p]``. Pass (index S*S) and ``global`` are unchanged.
The inverse table is generated separately (not assumed to equal some other id).
"""

from __future__ import annotations

import json
import numpy as np

NUM_SYMMETRIES = 8


def map_point(x: int, y: int, size: int, sym: int) -> tuple[int, int]:
    if not 0 <= sym < NUM_SYMMETRIES:
        raise ValueError("symmetry id must be 0..7")
    if sym >= 4:
        x, y = size - 1 - x, y
    for _ in range(sym % 4):
        x, y = size - 1 - y, x
    return x, y


def forward_permutation(size: int, sym: int) -> np.ndarray:
    """``int64[S*S]`` where entry ``p`` is the index the point ``p`` moves to."""
    perm = np.empty(size * size, dtype=np.int64)
    for y in range(size):
        for x in range(size):
            nx, ny = map_point(x, y, size, sym)
            perm[y * size + x] = ny * size + nx
    return perm


def inverse_permutation(size: int, sym: int) -> np.ndarray:
    fwd = forward_permutation(size, sym)
    inv = np.empty_like(fwd)
    inv[fwd] = np.arange(size * size, dtype=np.int64)
    return inv


def permutation_table(size: int) -> dict:
    """Shared JSON fixture consumed by Swift and Python tests."""
    return {
        "boardSize": size,
        "forward": [forward_permutation(size, s).tolist() for s in range(NUM_SYMMETRIES)],
        "inverse": [inverse_permutation(size, s).tolist() for s in range(NUM_SYMMETRIES)],
    }


def write_permutation_table(size: int, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(permutation_table(size), f, separators=(",", ":"))


def _apply_plane(values: np.ndarray, size: int, sym: int, inverse: bool) -> np.ndarray:
    perm = inverse_permutation(size, sym) if inverse else forward_permutation(size, sym)
    out = np.empty_like(values)
    out[..., perm, :] = values if values.ndim >= 2 else values
    return out


def transform_spatial(spatial: np.ndarray, sym: int, inverse: bool = False) -> np.ndarray:
    """``[B, S, S, C]`` -> same shape with every channel plane transformed."""
    if spatial.ndim != 4 or spatial.shape[1] != spatial.shape[2]:
        raise ValueError("spatial must be [B,S,S,C]")
    b, s, _, c = spatial.shape
    flat = spatial.reshape(b, s * s, c)
    perm = inverse_permutation(s, sym) if inverse else forward_permutation(s, sym)
    out = np.empty_like(flat)
    out[:, perm, :] = flat
    return out.reshape(b, s, s, c)


def transform_policy(policy: np.ndarray, size: int, sym: int, inverse: bool = False) -> np.ndarray:
    """``[B, S*S+1]``: board entries permuted, pass (last) kept in place."""
    if policy.shape[-1] != size * size + 1:
        raise ValueError("policy last dim must be S*S+1")
    perm = inverse_permutation(size, sym) if inverse else forward_permutation(size, sym)
    out = np.empty_like(policy)
    out[..., perm] = policy[..., : size * size]
    out[..., size * size] = policy[..., size * size]
    return out


def transform_legal(legal: np.ndarray, size: int, sym: int, inverse: bool = False) -> np.ndarray:
    return transform_policy(legal, size, sym, inverse)


def transform_ownership(ownership: np.ndarray, size: int, sym: int, inverse: bool = False) -> np.ndarray:
    """``[B, S*S]`` or ``[B, S, S]``."""
    shape = ownership.shape
    flat = ownership.reshape(shape[0], size * size)
    perm = inverse_permutation(size, sym) if inverse else forward_permutation(size, sym)
    out = np.empty_like(flat)
    out[:, perm] = flat
    return out.reshape(shape)
