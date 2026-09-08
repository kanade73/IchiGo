"""Prefix discretisation schedule (docs/spec/02-training.md §6, T16).

For ``max_steps`` optimizer steps and ``L`` layers:
  stage 1  [0, 60%)        tau = 1, all layers soft
  stage 2  [60%, 90%)      tau decreases linearly 1 -> tau_final; the stage is split into L equal
                           intervals and at the end of interval k layer k is frozen (argmax gates,
                           theta updates stopped, output evaluated as hard 0/1)
  stage 3  [90%, 100%]     every gate frozen; only the heads train
``state_at(step)`` returns (tau, frozen_prefix, heads_only) for the optimizer step about to run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduleState:
    tau: float
    frozen_prefix: int
    heads_only: bool
    stage: int


class PrefixSchedule:
    def __init__(self, max_steps: int, layers: int, tau_final: float = 0.2, fractions=(0.6, 0.3, 0.1)):
        self.max_steps = max_steps
        self.layers = layers
        self.tau_final = tau_final
        self.s1_end = int(round(max_steps * fractions[0]))
        self.s2_end = int(round(max_steps * (fractions[0] + fractions[1])))

    def freeze_steps(self) -> list[int]:
        """Step at which layer k becomes frozen (first step that runs with k+1 frozen layers)."""
        span = self.s2_end - self.s1_end
        return [self.s1_end + int(round(span * (k + 1) / self.layers)) for k in range(self.layers)]

    def state_at(self, step: int) -> ScheduleState:
        if step < self.s1_end:
            return ScheduleState(1.0, 0, False, 1)
        if step >= self.s2_end:
            return ScheduleState(self.tau_final, self.layers, True, 3)
        span = max(1, self.s2_end - self.s1_end)
        frac = (step - self.s1_end) / span
        tau = 1.0 + (self.tau_final - 1.0) * frac
        frozen = sum(1 for fs in self.freeze_steps() if step >= fs)
        return ScheduleState(tau, min(frozen, self.layers), frozen >= self.layers, 2)
