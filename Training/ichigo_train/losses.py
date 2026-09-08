"""Training losses (docs/spec/02-training.md §5, T14).

    p = legal-masked softmax(policy logits), q = softmax(wdl), e = q_win + 0.5 q_draw
    Lpolicy   = -Σ target_policy log p
    Lexpected = -t log e - (1-t) log(1-e)
    Lwdl      = -Σ target_wdl log q
    Lscore    = Huber((pred - target)/S, delta=1)
    Lowner    = mean_xy (pred - target)^2
    L = Lpolicy + Lexpected + 0.5 Lwdl + 0.25 Lscore + 0.25 Lowner

Each term is averaged separately over the sum of sample_weight of the samples whose mask is 1;
a term with no valid sample is exactly 0 (never NaN). log inputs are clipped to >= 1e-7.

Gradient-accumulation and DDP normalisation (docs/spec/02-training.md §7): an optimizer step draws
``accumulation`` microbatches (and, under DDP, ``world_size`` ranks each draw their own disjoint
microbatches), and each microbatch gets its own ``backward()`` call. Normalising each microbatch by
*its own* local valid-weight sum and averaging those normalised losses (dividing by accumulation)
is wrong whenever the valid mask differs between microbatches — a term whose only valid sample
lands in a small microbatch would be overweighted relative to one spread over a full microbatch.
The correct denominator is the valid weight sum over the *whole* effective batch (every microbatch
of every rank for this optimizer step). The caller (train.py) therefore: (1) draws all microbatches
of the step up front, (2) sums each term's local weight (this rank, all its microbatches) via
:func:`local_weight_sums`, (3) all-reduces that sum across ranks (see distributed.py) to get the
global denominator, (4) calls :func:`compute_losses` once per microbatch with
``weight_denominators=<global sums>`` and ``numerator_scale=world_size``, and (5) calls
``backward()`` on each microbatch's loss *without* dividing by accumulation — the fixed global
denominator already normalises across microbatches, and DDP's own gradient all-reduce divides the
summed per-microbatch gradients by world_size again, which is exactly what ``numerator_scale``
compensates for (worked through: summing ``world_size * NUM_mb / DEN_global`` over every
microbatch of every rank, then dividing by world_size once for DDP's average, gives
``NUM_global / DEN_global`` — the correct global weighted mean's gradient). Single-process training
is just the world_size=1 special case (no all-reduce, numerator_scale=1); the microbatch-normalisation
fix still applies. Validation (metrics.py) evaluates one batch of the whole holdout at a time and
does not accumulate gradients, so it is unaffected: it aggregates per-batch weighted sums itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7
WEIGHTS = {"policy": 1.0, "expected_result": 1.0, "wdl": 0.5, "score": 0.25, "ownership": 0.25}
TERM_NAMES = ("policy", "expected_result", "score", "ownership", "wdl")


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """log_softmax over legal entries only; illegal entries get log-prob -inf but are never
    multiplied by non-zero targets (targets are 0 there), so we return 0 for them instead."""
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(legal > 0, logits, torch.full_like(logits, neg))
    logp = F.log_softmax(masked, dim=-1)
    return torch.where(legal > 0, logp, torch.zeros_like(logp))


def local_weight_sums(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-term valid weight sum (``sample_weight`` where ``target_mask`` is 1) for one microbatch,
    with no dependency on model outputs. Callers accumulate this across every microbatch of an
    optimizer step (and, under DDP, all-reduce across ranks) to get the global denominator that
    ``compute_losses(weight_denominators=...)`` needs -- see the module docstring."""
    mask = batch["target_mask"].to(torch.float32)
    w = batch["sample_weight"].to(torch.float32)
    return {name: (w * mask[:, i]).sum() for i, name in enumerate(TERM_NAMES)}


def compute_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], board_size: int,
                   weight_denominators: dict[str, torch.Tensor] | None = None, numerator_scale: float = 1.0) -> dict[str, torch.Tensor]:
    """Returns dict with per-term losses, "total", and per-term valid weight sums ("w_<term>")."""
    mask = batch["target_mask"].to(torch.float32)   # [B,5]
    w = batch["sample_weight"].to(torch.float32)     # [B]
    S = board_size
    terms: dict[str, torch.Tensor] = {}
    denoms: dict[str, torch.Tensor] = {}

    # policy
    logp = masked_log_softmax(outputs["policy_logits"], batch["legal"])
    tp = batch["policy"]
    lp = -(tp * logp).sum(-1)                        # [B]
    terms["policy"] = lp
    # expected result
    q = torch.softmax(outputs["wdl_logits"], dim=-1)
    e = (q[:, 0] + 0.5 * q[:, 1]).clamp(EPS, 1 - EPS)
    t = batch["expected_result"]
    terms["expected_result"] = -(t * torch.log(e) + (1 - t) * torch.log(1 - e))
    # wdl
    logq = torch.log(q.clamp_min(EPS))
    terms["wdl"] = -(batch["wdl"] * logq).sum(-1)
    # score
    diff = (outputs["score_mean"] - batch["score"]) / S
    terms["score"] = F.huber_loss(diff, torch.zeros_like(diff), delta=1.0, reduction="none")
    # ownership
    own_t = batch["ownership"].reshape(batch["ownership"].shape[0], -1)
    terms["ownership"] = ((outputs["ownership"] - own_t) ** 2).mean(-1)

    out: dict[str, torch.Tensor] = {}
    total = torch.zeros((), dtype=torch.float32, device=w.device)
    for i, name in enumerate(TERM_NAMES):
        wm = w * mask[:, i]
        local_den = wm.sum()
        den = weight_denominators[name] if weight_denominators is not None else local_den
        num = (terms[name] * wm).sum() * numerator_scale
        val = torch.where(den > 0, num / den.clamp_min(EPS), torch.zeros_like(num))
        out[name] = val
        out["w_" + name] = local_den
        total = total + WEIGHTS[name] * val
    out["total"] = total
    return out


def policy_top1(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Fraction of masked-policy samples whose argmax matches the target argmax (legal-masked)."""
    neg = torch.finfo(outputs["policy_logits"].dtype).min
    pred = torch.where(batch["legal"] > 0, outputs["policy_logits"], torch.full_like(outputs["policy_logits"], neg)).argmax(-1)
    tgt = batch["policy"].argmax(-1)
    m = batch["target_mask"][:, 0] > 0
    if m.sum() == 0:
        return torch.zeros(())
    return (pred[m] == tgt[m]).float().mean()


def expected_result_mae(outputs, batch) -> torch.Tensor:
    q = torch.softmax(outputs["wdl_logits"], dim=-1)
    e = q[:, 0] + 0.5 * q[:, 1]
    m = batch["target_mask"][:, 1] > 0
    if m.sum() == 0:
        return torch.zeros(())
    return (e[m] - batch["expected_result"][m]).abs().mean()
