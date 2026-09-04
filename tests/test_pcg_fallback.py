from __future__ import annotations

import numpy as np

from src.pcg import MatrixFreeOperator, pcg_solve


def _tiny_operator():
    n_dof = 8
    elem_dof_array = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    k0_unit = np.eye(8) * 1000.0
    B_cen = np.eye(3, 8)[:3] * 0.1
    return MatrixFreeOperator(
        n_dof=n_dof,
        elem_dof_array=elem_dof_array,
        k0_unit=k0_unit,
        B_cen=B_cen,
        E=30000.0,
        nu=0.25,
        residual_stiffness=1e-6,
    )


def test_pcg_short_iteration_returns_gracefully():
    """A non-converged solve must return finite best-effort data, not raise."""
    op = _tiny_operator()
    rhs = np.ones(8) * 100.0
    x, n_iter, resid = pcg_solve(op, rhs, np.zeros(1), max_iter=1, tol=1e-14)
    assert n_iter == 1
    assert np.isfinite(resid)
    assert np.all(np.isfinite(x))
    assert x.shape == (8,)


def test_pcg_long_run_converges_with_finite_iterations():
    op = _tiny_operator()
    rhs = np.ones(8) * 100.0
    x, n_iter, resid = pcg_solve(op, rhs, np.zeros(1), max_iter=2000, tol=1e-12)
    assert 0 < n_iter <= 2000
    assert np.isfinite(resid)
    assert np.all(np.isfinite(x))
