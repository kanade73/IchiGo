"""Fixed wiring generation and gate-logit initialisation (docs/spec/01-network.md §3).

Wiring shape ``[L, C, 2, 4] int32`` with the last axis ``(bank, channel, dx, dy)``:
  bank 0 = previous layer (layer 0: the 32 input channels), bank 1 = the 32 input channels
  (only allowed from layer 1 on). Input point is ``(x+dx, y+dy)``; off-board reads 0.
Layer ``l`` offsets are ``dx, dy ∈ {-d_l, 0, d_l}``.

Random-draw order (this order, together with NumPy PCG64 and the effective wiring seed, is what
makes the wiring reproducible; the saved wiring file remains the authoritative copy). The
effective wiring seed is ``spec.wiring_seed`` if set, else ``spec.seed`` -- ``wiring_seed`` lets an
experiment vary the training seed (data order, augmentation, theta/head init downstream of this
generator) while holding the wiring itself fixed, or vice versa:
  rng = PCG64(wiring_seed if wiring_seed is not None else seed)
  for each layer l, for each channel c:
      if A is not fixed: A = draw_reference(l)
      B = draw_reference(l); while B == A: B = draw_reference(l)
  then, for each layer l, for each channel c with c % 4 != 0: theta[l, c, :] = Normal(0, 0.05)^16
where draw_reference = (bank via uniform() < 1 - bank1_ratio ? 0 : 1 [layer 0 always 0, no draw],
                        channel = integers(0, bankChannels), dx = (-d,0,d)[integers(0,3)],
                        dy = (-d,0,d)[integers(0,3)]).
``bank1_ratio`` (default 0.1, matching 01-network.md §3's "90% bank=0 / 10% bank=1") is the
fraction of non-fixed A/B draws (layer >= 1 only) that land on bank 1 (the original 32 input
channels) instead of bank 0 (the previous layer's output).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INPUT_CHANNELS = 32
DEFAULT_SEED = 20260908
IDENTITY_GATE = 12  # gate id "a"
IDENTITY_THETA = 3.0
THETA_STD = 0.05

PROFILES: dict[str, dict] = {
    "tiny": {"channels": 64, "dilations": [1, 1, 2, 1]},
    "small": {"channels": 256, "dilations": [1, 1, 2, 1, 4, 1, 8, 1]},
    "base": {"channels": 512, "dilations": [1, 1, 2, 1, 4, 1, 8, 1, 4, 1, 2, 1]},
}


@dataclass
class ModelSpec:
    channels: int
    dilations: list[int]
    seed: int = DEFAULT_SEED
    profile: str | None = None
    bank1_ratio: float = 0.1
    wiring_seed: int | None = None

    @property
    def layers(self) -> int:
        return len(self.dilations)

    @staticmethod
    def from_profile(name: str, seed: int = DEFAULT_SEED) -> "ModelSpec":
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}")
        p = PROFILES[name]
        return ModelSpec(channels=p["channels"], dilations=list(p["dilations"]), seed=seed, profile=name)

    @classmethod
    def custom(cls, channels: int, dilations: list[int], seed: int = DEFAULT_SEED, profile: str | None = None,
               bank1_ratio: float = 0.1, wiring_seed: int | None = None) -> "ModelSpec":
        """Build a spec with an arbitrary channel/dilation schedule (T33 capacity experiments:
        wide/local variants of a profile, wiring-seed sweeps, bank-1 ratio sweeps). ``profile`` is
        purely a record of which named profile (if any) this was derived from -- it does not
        constrain ``channels``/``dilations``."""
        return cls(channels=channels, dilations=list(dilations), seed=seed, profile=profile,
                   bank1_ratio=bank1_ratio, wiring_seed=wiring_seed)


@dataclass
class Wiring:
    """``wiring`` int32 [L,C,2,4]; ``theta`` float32 [L,C,16] initial logits."""

    wiring: np.ndarray
    theta: np.ndarray
    dilations: list[int] = field(default_factory=list)

    @property
    def layers(self) -> int:
        return self.wiring.shape[0]

    @property
    def channels(self) -> int:
        return self.wiring.shape[1]


def bank_channels(layer: int, bank: int, channels: int) -> int:
    if bank == 0:
        return INPUT_CHANNELS if layer == 0 else channels
    if bank == 1:
        if layer == 0:
            raise ValueError("bank 1 is not allowed in layer 0")
        return INPUT_CHANNELS
    raise ValueError(f"invalid bank {bank}")


def _draw_reference(rng: np.random.Generator, layer: int, channels: int, d: int, bank1_ratio: float = 0.1) -> tuple[int, int, int, int]:
    if layer == 0:
        bank = 0
    else:
        bank = 0 if rng.uniform() < (1.0 - bank1_ratio) else 1
    ch = int(rng.integers(0, bank_channels(layer, bank, channels)))
    offsets = (-d, 0, d)
    dx = offsets[int(rng.integers(0, 3))]
    dy = offsets[int(rng.integers(0, 3))]
    return (bank, ch, dx, dy)


def generate_wiring(spec: ModelSpec) -> Wiring:
    seed = spec.wiring_seed if spec.wiring_seed is not None else spec.seed
    rng = np.random.Generator(np.random.PCG64(seed))
    L, C = spec.layers, spec.channels
    wiring = np.zeros((L, C, 2, 4), dtype=np.int32)
    for l in range(L):
        d = spec.dilations[l]
        for c in range(C):
            if l == 0:
                a = (0, c % INPUT_CHANNELS, 0, 0)
            elif c % 4 == 0:
                a = (0, c, 0, 0)
            else:
                a = _draw_reference(rng, l, C, d, spec.bank1_ratio)
            b = _draw_reference(rng, l, C, d, spec.bank1_ratio)
            while b == a:
                b = _draw_reference(rng, l, C, d, spec.bank1_ratio)
            wiring[l, c, 0] = a
            wiring[l, c, 1] = b
    theta = np.zeros((L, C, 16), dtype=np.float32)
    for l in range(L):
        for c in range(C):
            if c % 4 == 0:
                theta[l, c, IDENTITY_GATE] = IDENTITY_THETA
            else:
                theta[l, c, :] = rng.normal(0.0, THETA_STD, size=16).astype(np.float32)
    return Wiring(wiring=wiring, theta=theta, dilations=list(spec.dilations))


def validate_wiring(wiring: np.ndarray, dilations: list[int]) -> None:
    """Structural checks shared with the Swift loader. Raises ValueError."""
    if wiring.ndim != 4 or wiring.shape[2] != 2 or wiring.shape[3] != 4:
        raise ValueError(f"wiring must be [L,C,2,4], got {wiring.shape}")
    L, C = wiring.shape[:2]
    if L != len(dilations):
        raise ValueError("dilation count must equal layer count")
    for l in range(L):
        d = dilations[l]
        for c in range(C):
            refs = []
            for k in range(2):
                bank, ch, dx, dy = (int(v) for v in wiring[l, c, k])
                if bank not in (0, 1) or (bank == 1 and l == 0):
                    raise ValueError(f"layer {l} channel {c}: invalid bank {bank}")
                if not 0 <= ch < bank_channels(l, bank, C):
                    raise ValueError(f"layer {l} channel {c}: channel {ch} out of range")
                if dx not in (-d, 0, d) or dy not in (-d, 0, d):
                    raise ValueError(f"layer {l} channel {c}: offset ({dx},{dy}) not in dilation {d}")
                refs.append((bank, ch, dx, dy))
            if refs[0] == refs[1]:
                raise ValueError(f"layer {l} channel {c}: A == B")
