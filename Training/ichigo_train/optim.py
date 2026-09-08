"""Optimizer and schedule (docs/spec/02-training.md §5, T14).

AdamW with two groups: gate logits (lr=gateLearningRate, wd=0) and heads (lr=headLearningRate,
wd=headWeightDecay); betas (0.9, 0.999), eps 1e-8. Global-norm clip. LR schedule: linear warmup
over the first 5% of optimizer steps, then cosine decay to 10% of the initial lr.
"""

from __future__ import annotations

import math

import torch

from .model import LogicNet


def build_optimizer(model: LogicNet, gate_lr: float, head_lr: float, head_wd: float) -> torch.optim.AdamW:
    gate_params = [model.theta]
    if getattr(model, "wiring_mode", "fixed") == "learned-k":
        gate_params.append(model.phi)
    return torch.optim.AdamW([
        {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0, "name": "gates"},
        {"params": [q for n, q in model.named_parameters() if n not in ("theta", "phi")], "lr": head_lr, "weight_decay": head_wd, "name": "heads"},
    ], betas=(0.9, 0.999), eps=1e-8)


def lr_multiplier(step: int, max_steps: int, warmup_fraction: float = 0.05, floor: float = 0.1) -> float:
    """Multiplier for optimizer step ``step`` (0-based) out of ``max_steps``."""
    warm = max(1, int(round(max_steps * warmup_fraction)))
    if step < warm:
        return (step + 1) / warm
    if max_steps <= warm:
        return 1.0
    progress = (step - warm) / max(1, max_steps - warm)
    progress = min(1.0, progress)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer, max_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, max_steps))


def clip_gradients(model: LogicNet, max_norm: float) -> float:
    params = list(model.parameters())
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
