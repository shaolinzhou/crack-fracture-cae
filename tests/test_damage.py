from __future__ import annotations

import numpy as np

from src.damage_models import mazars_damage_target, compute_damage_parameters


def test_mazars_below_threshold():
    """Damage is zero when eps_eq <= eps0."""
    D = mazars_damage_target(np.array([1e-5, 5e-5, 1e-4]), 1e-4, 1000.0, 1e-2)
    assert np.all(D == 0.0)


def test_mazars_above_threshold():
    """Damage is positive when eps_eq > eps0."""
    D = mazars_damage_target(np.array([2e-4, 5e-4]), 1e-4, 1000.0, 1e-2)
    assert np.all(D > 0.0)
    assert np.all(D < 1.0)


def test_mazars_monotonic():
    """Damage increases monotonically with equivalent strain."""
    eps_range = np.logspace(-4, -1, 10)
    D = mazars_damage_target(eps_range, 1e-4, 1000.0, 1.0)
    diffs = np.diff(D)
    assert np.all(diffs >= -1e-12)


def test_damage_parameters():
    """compute_damage_parameters should return reasonable values."""
    eps0, beta = compute_damage_parameters(30000.0, 0.25, 6.0, 31.62, 0.75)
    assert eps0 == 6.0 / 30000.0  # 2e-4
    assert beta > 0
    assert np.isfinite(beta)
