import numpy as np
import torch

from ichigo_train import gates
from ichigo_train.discretize import GumbelSchedule, PrefixSchedule
from ichigo_train.export import export_model, model_from_loaded
from ichigo_train.model import LogicNet
from ichigo_train.model_factory import build_model_from_config
from ichigo_train.model_format import read_model
from ichigo_train.wiring import ModelSpec, Wiring


def _bits(batch=2, size=9):
    return torch.from_numpy(np.random.default_rng(123).integers(0, 2, (batch, size, size, 32), dtype=np.uint8))


def test_gumbel_ste_forward_is_binary_and_gradient_is_soft_surrogate():
    model = LogicNet(ModelSpec(channels=16, dilations=[1], seed=4))
    spatial = _bits()
    noise = torch.randn(model.theta.shape, generator=torch.Generator().manual_seed(8))
    got = torch.autograd.grad(model.soft_layers(spatial, 1.0, gumbel_noise=noise)[0].sum(), model.theta)[0]
    assert set(torch.unique(model.soft_layers(spatial, 1.0, gumbel_noise=noise)[0]).tolist()) <= {0.0, 1.0}

    a, b = model.gather_for(0, 9).gather(spatial.float(), spatial.float())
    expected_theta = model.theta.detach().clone().requires_grad_()
    t = gates.reduce_theta(expected_theta[0] + noise[0], 1.0)
    expected = gates.soft_gate_reduced(t.view(1, 1, 1, -1, 4), a, b).sum()
    expected_grad = torch.autograd.grad(expected, expected_theta)[0]
    assert torch.allclose(got, expected_grad, atol=1e-6, rtol=1e-5)


def test_schedules_and_entropy_knobs():
    s = PrefixSchedule(100, 4, tau_final=0.1, tau_start=0.5)
    assert s.state_at(0).tau == 0.5
    assert abs(s.state_at(60).tau - 0.5) < 1e-9
    assert GumbelSchedule(100, 4).state_at(89).heads_only is False
    assert GumbelSchedule(100, 4).state_at(90).heads_only is True

    theta = torch.tensor([[2.0, 0.0] + [0.0] * 14], requires_grad=True)
    zero = gates.gate_entropy_loss(theta, 1.0, 0.0)
    positive = gates.gate_entropy_loss(theta, 1.0, 0.1)
    assert zero.item() == 0.0 and positive.item() > 0.0
    positive.backward()
    assert theta.grad[0, 0] < 0 < theta.grad[0, 1]


def test_learned_wiring_k1_and_sharp_phi_match_fixed_and_export(tmp_path):
    cfg = {"profile": "tiny", "seed": 9, "wiringMode": "learned-k", "wiringCandidates": 1, "headVersion": 2}
    learned = build_model_from_config(cfg)
    fixed = build_model_from_config({"profile": "tiny", "seed": 9, "headVersion": 2})
    spatial, glob = _bits(), torch.zeros(2, 4)
    assert np.array_equal(learned.candidate_wiring_numpy()[:, :, :, 0], fixed.wiring_numpy())
    assert torch.equal(learned(spatial, glob)["policy_logits"], fixed(spatial, glob)["policy_logits"])

    cfg["wiringCandidates"] = 8
    learned = build_model_from_config(cfg)
    with torch.no_grad():
        learned.phi[..., 0] = 100.0
    fixed_selected = LogicNet(learned.spec,
                              wiring=Wiring(learned.hard_wiring_numpy(), learned.theta.detach().cpu().numpy(), learned.dilations),
                              heads=learned.head_numpy(), head_version=2)
    assert torch.allclose(learned(spatial, glob, tau_wire=0.01)["policy_logits"], fixed_selected(spatial, glob)["policy_logits"])

    out = tmp_path / "learned.ichigo"
    export_model(learned, str(out), [9])
    loaded = read_model(str(out))
    assert np.array_equal(loaded.wiring, learned.hard_wiring_numpy())
    assert np.array_equal(loaded.gates, learned.hard_gates())
    exported_model = model_from_loaded(loaded)
    for key, value in learned.forward_hard(spatial, glob).items():
        assert torch.equal(value, exported_model.forward_hard(spatial, glob)[key]), key
