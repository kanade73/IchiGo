import json

import numpy as np
import pytest
import torch

from ichigo_train import gates as G
from ichigo_train.config import ConfigError, load_config
from ichigo_train.export import export_model
from ichigo_train.model import LogicNet
from ichigo_train.model_factory import build_model_from_config
from ichigo_train.model_format import ModelFormatError, read_model
from ichigo_train.wiring import ModelSpec, generate_wiring, validate_wiring


def test_lut4_truth_values_match_reduced_form():
    torch.manual_seed(20)
    phi = torch.randn(4, 16, dtype=torch.float64)
    inputs = [torch.rand(7, 4, dtype=torch.float64) for _ in range(4)]
    t = G.lut_probabilities(phi, 0.7)
    q = []
    for i in range(16):
        term = torch.ones_like(inputs[0])
        for k, a in enumerate(inputs):
            term = term * (a if ((i >> (3 - k)) & 1) else (1 - a))
        q.append(term)
    direct = (t * torch.stack(q, dim=-1)).sum(-1)
    reduced = G.soft_gate_lut_reduced(t, *inputs)
    assert torch.allclose(direct, reduced, atol=1e-12)


def test_lut4_gradcheck():
    torch.manual_seed(21)
    phi = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
    inputs = tuple(torch.rand(3, 2, dtype=torch.float64, requires_grad=True) for _ in range(4))
    assert torch.autograd.gradcheck(lambda p, *x: G.soft_lut_direct(p, 0.5, *x), (phi, *inputs), eps=1e-6, atol=1e-5)


def test_lut4_msb_first_soft_and_hard_indexing_agree():
    # Make one row sharply true at a time. The row order is 0000, ..., 1111, with a1 as MSB.
    for row in range(16):
        phi = torch.full((1, 16), -60.0, dtype=torch.float64)
        phi[0, row] = 60.0
        inputs = [torch.tensor([[float((row >> (3 - k)) & 1)]], dtype=torch.float64) for k in range(4)]
        soft = G.soft_gate_lut_reduced(G.lut_probabilities(phi, 1.0).view(1, 16), *inputs).item()
        table = int(G.hard_lut(phi)[0])
        assert abs(soft - 1.0) < 1e-12
        assert ((table >> row) & 1) == 1
        assert table == 1 << row


def test_lut4_identity_initialisation_and_distinct_wiring():
    spec = ModelSpec.custom(channels=64, dilations=[1, 1, 2], seed=20260908, gate_arity=4)
    w = generate_wiring(spec)
    assert w.wiring.shape == (3, 64, 4, 4)
    assert w.theta.shape == (3, 64, 16)
    validate_wiring(w.wiring, spec.dilations, gate_arity=4)
    assert all(len({tuple(ref) for ref in refs}) == 4 for refs in w.wiring.reshape(-1, 4, 4))
    for l in range(1, spec.layers):
        for c in range(0, spec.channels, 4):
            assert w.wiring[l, c, 0].tolist() == [0, c, 0, 0]
    assert np.array_equal(w.theta[0, 0], np.r_[[ -3.0] * 8, [3.0] * 8].astype(np.float32))
    assert np.array_equal(G.hard_lut(w.theta[0, 0]), np.array(0xFF00, dtype=np.uint16))


def test_lut4_model_factory_and_gumbel_config_rejection(tmp_path):
    cfg = {"profile": "tiny", "seed": 4, "gateArity": 4}
    model = build_model_from_config(cfg)
    assert isinstance(model, LogicNet) and model.gate_arity == 4 and hasattr(model, "phi")
    assert model.theta is model.phi
    bad = dict(cfg, discretization="gumbel-ste-90-10")
    path = tmp_path / "invalid-config.json"
    with open(path, "w") as f:
        json.dump(dict(bad, data="d", out="o", boardSize=9, runId="bad"), f)
    with pytest.raises(ConfigError, match="gumbel"):
        load_config(path)


def test_lut4_export_uses_u16_and_roundtrips(tmp_path):
    model = LogicNet(ModelSpec.from_profile("tiny", seed=3, gate_arity=4))
    out = tmp_path / "lut4.ichigo"
    manifest = export_model(model, str(out), [9], {"runId": "lut4-test"})
    assert manifest["gateArity"] == 4
    assert manifest["gateEncoding"] == "lut4-msb-first"
    assert set(manifest["files"]) == {"wiring.i32", "gates.u16", "heads.f32"}
    loaded = read_model(str(out))
    assert loaded.wiring.shape == (4, 64, 4, 4)
    assert loaded.gates.dtype == np.dtype("uint16")
    assert loaded.gates.shape == (4, 64)


def test_lut4_rejects_u8_and_bad_wiring_shapes():
    base = {
        "format": "ichigo.logic", "version": 1, "featureVersion": 1, "headVersion": 2,
        "boardSizes": [9], "rulesId": "cgos-area-psk-v1", "channels": 64, "layers": 4,
        "dilations": [1, 1, 2, 1], "gateArity": 4, "gateEncoding": "lut4-msb-first", "layout": "NHWC",
        "endianness": "little", "valuePerspective": "to-move", "wdlOrder": ["win", "draw", "loss"],
        "calibrationTemperature": 1.0, "files": {"wiring.i32": {"byteLength": 4096, "sha256": "0" * 64},
        "gates.u8": {"byteLength": 256, "sha256": "0" * 64}, "heads.f32": {"byteLength": 0, "sha256": "0" * 64}},
        "headTensors": [], "trainingProvenance": {},
    }
    with pytest.raises(ModelFormatError, match="files"):
        from ichigo_train.model_format import validate_manifest
        validate_manifest(base)
