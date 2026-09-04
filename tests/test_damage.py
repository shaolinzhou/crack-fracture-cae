from __future__ import annotations

import numpy as np

from src.damage_models import (
    compute_damage_base,
    compute_damage_parameters,
    mazars_damage_target,
)


def _grid_geometry():
    """3x3 full active grid: 9 elements, radius-1 nonlocal neighborhood."""
    active = list(range(9))
    elem_ji = {e: (e // 3, e % 3) for e in active}
    return active, elem_ji


def _active_strains() -> np.ndarray:
    # tensile-dominated strains well above eps0 = 2e-4 -> all elements drive
    s = np.full((9, 3), 1e-3)
    s[:, 0] += np.linspace(0, 5e-4, 9)
    return s


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


def test_compute_damage_base_phases_and_nonlocal():
    """Exercise the full compute_damage_base path (nonlocal + adaptive damping)."""
    active, elem_ji = _grid_geometry()
    strains = _active_strains()
    D = np.zeros(9)
    eps_eq_grid = np.zeros((3, 3))

    def call(phase):
        return compute_damage_base(
            strains, D, 2e-4, 300.0, 4e-2, 50.0,
            0.3, 0.5, 0.7, phase,
            eps_eq_grid=eps_eq_grid, nonlocal_radius=1,
            active_elem_indices=active, elem_ji=elem_ji, Ny=3, Nx=3,
        )

    dd_w, eq_w = call("warmup")
    dd_c, eq_c = call("coupled")
    assert dd_w.shape == (9,)
    assert eq_w.shape == (9,)
    # warmup uses uniform damping 0.3; coupled uses fast damping 0.7 here
    # (all driving > 0.1), so the increments scale accordingly
    assert np.allclose(dd_c, dd_w * (0.7 / 0.3), rtol=1e-6)
    assert np.all(dd_c > 0.0)

    # pure-local branch (no nonlocal grid) must not raise and returns zeros-driven
    dd_local, _ = compute_damage_base(
        strains, np.ones(9) * 0.99, 2e-4, 300.0, 4e-2, 50.0,
        0.3, 0.5, 0.7, "warmup",
        eps_eq_grid=None, nonlocal_radius=0,
        active_elem_indices=active, elem_ji=elem_ji, Ny=3, Nx=3,
    )
    assert np.all(dd_local == 0.0)
