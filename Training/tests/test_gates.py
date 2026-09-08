import numpy as np
import pytest
import torch

from ichigo_train import gates as G


def test_truth_table_named_gates():
    t = G.truth_table()
    assert t.shape == (16, 4)
    # rows are (a,b) = (0,0),(0,1),(1,0),(1,1)
    assert t[G.GATE_FALSE].tolist() == [0, 0, 0, 0]
    assert t[G.GATE_AND].tolist() == [0, 0, 0, 1]
    assert t[G.GATE_XOR].tolist() == [0, 1, 1, 0]
    assert t[G.GATE_OR].tolist() == [0, 1, 1, 1]
    assert t[G.GATE_A].tolist() == [0, 0, 1, 1]
    assert t[G.GATE_B].tolist() == [0, 1, 0, 1]
    assert t[G.GATE_NOT_A].tolist() == [1, 1, 0, 0]
    assert t[G.GATE_NAND].tolist() == [1, 1, 1, 0]
    assert t[G.GATE_TRUE].tolist() == [1, 1, 1, 1]


def test_all_64_truth_values():
    for g in range(16):
        for a in (0, 1):
            for b in (0, 1):
                assert G.hard_gate(g, a, b) == (g >> (2 * a + b)) & 1
    g = np.arange(16, dtype=np.uint8)[:, None]
    a = np.array([0, 0, 1, 1], dtype=np.uint8)[None, :]
    b = np.array([0, 1, 0, 1], dtype=np.uint8)[None, :]
    assert np.array_equal(G.hard_gate_array(g, a, b), G.truth_table())


def test_ab_swap_is_not_symmetric_for_a_and_b_gates():
    assert G.hard_gate(G.GATE_A, 1, 0) == 1 and G.hard_gate(G.GATE_A, 0, 1) == 0
    assert G.hard_gate(G.GATE_B, 1, 0) == 0 and G.hard_gate(G.GATE_B, 0, 1) == 1


def test_soft_matches_hard_on_bits():
    theta = torch.zeros(16, 16, dtype=torch.float64)
    theta[torch.arange(16), torch.arange(16)] = 50.0  # one-hot-ish
    t = G.reduce_theta(theta, 1.0)
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            y = G.soft_gate_reduced(t, torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))
            for g in range(16):
                assert abs(y[g].item() - G.hard_gate(g, int(a), int(b))) < 1e-12


def test_direct_and_reduced_agree_forward_and_gradient():
    torch.manual_seed(0)
    theta = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    a = torch.rand(5, 7, dtype=torch.float64, requires_grad=True)
    b = torch.rand(5, 7, dtype=torch.float64, requires_grad=True)
    tau = 0.7
    y1 = G.soft_gate_direct(theta, tau, a, b)
    y2 = G.soft_gate_reduced(G.reduce_theta(theta, tau), a, b)
    assert torch.allclose(y1, y2, atol=1e-12)
    g1 = torch.autograd.grad(y1.sum(), (theta, a, b))
    g2 = torch.autograd.grad(y2.sum(), (theta, a, b))
    for x, y in zip(g1, g2):
        assert torch.allclose(x, y, atol=1e-10)


def test_gradcheck_float64():
    torch.manual_seed(1)
    theta = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    a = torch.rand(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.rand(4, 3, dtype=torch.float64, requires_grad=True)

    def f(theta, a, b):
        return G.soft_gate_reduced(G.reduce_theta(theta, 0.5), a, b)

    assert torch.autograd.gradcheck(f, (theta, a, b), eps=1e-6, atol=1e-5)


def test_argmax_tie_smallest_id():
    theta = np.zeros((2, 16), dtype=np.float32)
    theta[0, [3, 9]] = 1.0
    theta[1, :] = 2.0
    assert G.argmax_gate(theta).tolist() == [3, 0]
    with pytest.raises(ValueError):
        G.argmax_gate(np.array([[np.nan] * 16]))
