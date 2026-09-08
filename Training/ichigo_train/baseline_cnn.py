"""Diagnostic CNN baseline (docs/spec/04-tasks.md T33).

``32ch spatial -> 3x3 stem to C channels (default 64) -> num_blocks (default 4) residual blocks
(two 3x3 convs each) -> the SAME local/global heads as LogicNet (model.py's head_shapes /
heads_forward, headVersion 1 or 2) -> the same five losses (losses.py)``. This is the
"通常CNN（同じ特徴とheads、小型3x3 residual trunk）" baseline from docs/spec/02-training.md §8: a
research diagnostic to tell apart a data/label problem from a representational-capacity problem
in the logic-gate network (see docs/implementation-status.md, "value head の切り分け", where a
64ch/4-block version of this network -- originally prototyped in Scripts/cnn_baseline_pilot.py --
learned value/ownership from the same data the logic-gate small profile could not). It is
explicitly NOT a production model: it ignores 01-network.md §3's fixed sparse wiring and 16-gate
layers entirely and uses a dense float conv trunk instead, so ``forward_hard``/``hard_layers`` are
the SAME dense computation as ``forward``/``soft_layers`` -- there is no separate discrete
evaluation to match, unlike LogicNet's soft/hard split (01-network.md §2 argmax vs. softmax).

Exposes the attribute surface the training loop (train.py), checkpointer (checkpoint.py),
optimizer (optim.py) and metrics (metrics.py) use on a LogicNet -- ``forward``, ``forward_hard``,
``theta``, ``spec``, ``dilations``, ``channels``, ``head_version``, ``heads``, ``wiring_numpy``,
``hard_gates``, ``hard_layers``, ``head_numpy`` -- so it can mostly stand in for one; see
model_factory.py for how a training config selects it (``modelType: "cnn-baseline"``), and the
report accompanying this change for the exact train.py/config.py diff that would wire it in
(not applied here -- train.py/config.py are owned by a concurrent edit).

``theta`` is a zero-element ``nn.Parameter`` of shape ``[0, 0, 16]`` (mirroring LogicNet's
``[L, C, 16]`` with L=C=0) purely so code that reads ``model.theta`` / ``model.theta.grad``
without special-casing the model type does not crash: optim.clip_gradients includes it in the
parameter list (torch's grad-norm clipping simply skips parameters whose ``.grad`` is None, which
``theta`` always is since it never enters the forward graph), optim.build_optimizer's "gates"
param group holds it (AdamW skips param-groups members with no gradient), and
metrics.gate_statistics reads ``theta.shape[0]`` as the layer count (0, so its per-layer loop
never executes) and calls ``hard_gates()`` (below) for the gate histogram, which returns an empty
``[0,0]`` array whose ``np.bincount(..., minlength=16)`` is all zeros -- never a crash, and an
honest "no gates" signal rather than a fabricated histogram.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import HEAD_TENSOR_NAMES, HEAD_VERSION, SUPPORTED_HEAD_VERSIONS, LogicNet, head_shapes, init_heads
from .wiring import INPUT_CHANNELS, ModelSpec

DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 4


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a residual add, ReLU after each conv and after the add (matches
    Scripts/cnn_baseline_pilot.py's ``Block``)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv2(F.relu(self.conv1(x))))


class BaselineCNN(nn.Module):
    """Diagnostic 3x3 residual CNN with LogicNet's heads bolted on (T33). Board-size agnostic
    (all-convolutional trunk); works on 9x9 and 19x19 without reconfiguration."""

    def __init__(self, channels: int = DEFAULT_CHANNELS, num_blocks: int = DEFAULT_BLOCKS,
                 seed: int = 20260908, profile: str | None = None, head_version: int = HEAD_VERSION,
                 heads: dict[str, np.ndarray] | None = None):
        super().__init__()
        if head_version not in SUPPORTED_HEAD_VERSIONS:
            raise ValueError(f"unsupported headVersion {head_version}")
        self.head_version = head_version
        # dilations=[] gives spec.layers == 0 for free (ModelSpec.layers is len(dilations)):
        # there are no gate layers, so nothing here should ever look like it has any.
        self.spec = ModelSpec(channels=channels, dilations=[], seed=seed, profile=profile)
        self.channels = channels
        self.dilations: list[int] = []
        self.stem = nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.theta = nn.Parameter(torch.zeros(0, 0, 16, dtype=torch.float32))
        if heads is None:
            rng = np.random.Generator(np.random.PCG64(seed + 1))
            heads = init_heads(channels, rng, head_version)
        expected = head_shapes(channels, head_version)
        for n in HEAD_TENSOR_NAMES:
            if tuple(np.shape(heads[n])) != expected[n]:
                raise ValueError(f"head tensor {n} has shape {np.shape(heads[n])}, expected {expected[n]} for headVersion {head_version}")
        self.heads = nn.ParameterDict({n: nn.Parameter(torch.from_numpy(np.array(heads[n], dtype=np.float32))) for n in HEAD_TENSOR_NAMES})

    # ----- LogicNet-compatible surface (see module docstring) -----
    def wiring_numpy(self) -> np.ndarray:
        return np.zeros((0, 0, 2, 4), dtype=np.int32)

    def hard_gates(self) -> np.ndarray:
        return np.zeros((0, 0), dtype=np.uint8)

    def hard_layers(self, spatial: torch.Tensor) -> list[torch.Tensor]:
        return []

    def head_numpy(self) -> dict[str, np.ndarray]:
        return {n: self.heads[n].detach().cpu().numpy().astype(np.float32) for n in HEAD_TENSOR_NAMES}

    # ----- forward -----
    def _trunk(self, spatial: torch.Tensor) -> torch.Tensor:
        x = spatial.to(torch.float32).permute(0, 3, 1, 2)         # NHWC -> NCHW
        h = self.blocks(F.relu(self.stem(x)))
        return h.permute(0, 2, 3, 1)                               # NCHW -> NHWC, [B,S,S,C]

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor, tau: float = 1.0, frozen_prefix: int = 0,
                gumbel_noise: torch.Tensor | None = None, tau_wire: float | None = None) -> dict[str, torch.Tensor]:
        # tau/frozen_prefix are LogicNet-only concepts (gate-softmax temperature, discretisation
        # prefix); accepted here only so train.py can call both models identically, and ignored.
        LogicNet._check_inputs(spatial, glob)
        h = self._trunk(spatial)
        return LogicNet.heads_forward(self, h, glob.to(torch.float32))

    def forward_hard(self, spatial: torch.Tensor, glob: torch.Tensor) -> dict[str, torch.Tensor]:
        # There is nothing to discretise (no gates): hard == soft for this model.
        with torch.no_grad():
            return self.forward(spatial, glob)
