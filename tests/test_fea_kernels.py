from __future__ import annotations

import numpy as np

from src.fea_kernels import (
    ball_neighbor_mean,
    element_damage_features,
    knn_max_gradient,
    smooth_edges_from_tree,
)


def _pts(n: int = 6):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(n, 2))


def test_ball_neighbor_mean_constant_field():
    pts = _pts()
    v = np.ones(len(pts))
    out = ball_neighbor_mean(pts, v, radius=0.2)
    assert np.allclose(out, 1.0)


def test_knn_max_gradient_zero_for_constant():
    pts = _pts()
    g = knn_max_gradient(pts, np.ones(len(pts)), k=3)
    assert np.allclose(g, 0.0)


def test_element_damage_features_shape_and_bounds():
    n = 6
    pts = _pts(n)
    rng = np.random.default_rng(1)
    D = rng.uniform(0.0, 0.6, n)
    strains = rng.uniform(-1e-3, 3e-3, (n, 3))
    stresses = rng.uniform(-5.0, 5.0, (n, 3))
    gD = knn_max_gradient(pts, D, k=3)
    F = element_damage_features(D, strains, stresses, nu=0.25, eps0=2e-4, l_c=0.5, gD=gD)
    assert F.shape == (n, 5)
    assert np.all(np.isfinite(F))
    # tanh features are bounded; D passes through on axis 0
    assert np.all(F[:, 0] == D)


def test_smooth_edges_from_tree_pairs():
    pts = _pts(6)
    edges = smooth_edges_from_tree(pts, k=3)
    assert len(edges) > 0
    for i, j in edges:
        assert 0 <= i < 6 and 0 <= j < 6
