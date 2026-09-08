import json
import os

import numpy as np
import pytest

from ichigo_train import symmetry as S

FIX = os.path.join(os.path.dirname(__file__), "..", "..", "Tests", "Fixtures", "symmetry")


@pytest.mark.parametrize("size", [9, 19])
def test_inverse_roundtrip_all_eight(size):
    rng = np.random.default_rng(size)
    spatial = rng.integers(0, 2, size=(2, size, size, 32), dtype=np.uint8)
    policy = rng.random((2, size * size + 1)).astype(np.float32)
    own = rng.random((2, size, size)).astype(np.float32)
    for sym in range(8):
        assert np.array_equal(S.transform_spatial(S.transform_spatial(spatial, sym), sym, inverse=True), spatial)
        p2 = S.transform_policy(policy, size, sym)
        assert p2[:, -1].tolist() == policy[:, -1].tolist()
        assert np.array_equal(S.transform_policy(p2, size, sym, inverse=True), policy)
        assert np.array_equal(S.transform_ownership(S.transform_ownership(own, size, sym), size, sym, inverse=True), own)
        fwd = S.forward_permutation(size, sym)
        assert sorted(fwd.tolist()) == list(range(size * size))


def test_rotation_and_flip_definitions():
    # R(x,y) = (S-1-y, x); F(x,y) = (S-1-x, y)
    assert S.map_point(0, 0, 9, 1) == (8, 0)
    assert S.map_point(2, 5, 9, 1) == (3, 2)
    assert S.map_point(2, 5, 9, 4) == (6, 5)
    assert S.map_point(2, 5, 9, 2) == S.map_point(*S.map_point(2, 5, 9, 1), 9, 1)
    assert S.map_point(2, 5, 9, 5) == S.map_point(*S.map_point(2, 5, 9, 4), 9, 1)


@pytest.mark.parametrize("size", [9, 19])
def test_shared_fixture_matches(size):
    path = os.path.join(FIX, f"perm-{size}.json")
    with open(path) as f:
        table = json.load(f)
    assert table == S.permutation_table(size)
