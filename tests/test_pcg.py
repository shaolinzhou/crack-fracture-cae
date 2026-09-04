from __future__ import annotations

import numpy as np
import pytest

from src.pcg import MatrixFreeOperator, pcg_solve


@pytest.fixture
def tiny_operator():
    """A minimal 2×2 grid → 8 DOFs, 1 active element operator for testing."""
    n_dof = 8
    elem_dof_array = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    k0_unit = np.eye(8) * 1000.0
    B_cen = np.eye(3, 8)[:3] * 0.1
    op = MatrixFreeOperator(
        n_dof=n_dof,
        elem_dof_array=elem_dof_array,
        k0_unit=k0_unit,
        B_cen=B_cen,
        E=30000.0,
        nu=0.25,
        residual_stiffness=1e-6,
    )
    return op, 1, np.zeros(1)


def test_operator_shape(tiny_operator):
    op, *_ = tiny_operator
    u = np.random.randn(8)
    D = np.zeros(1)
    f = op.apply(u, D)
    assert f.shape == (8,)


def test_pcg_solve_converges(tiny_operator):
    op, n_dof, _ = tiny_operator
    D = np.zeros(1)
    rhs = np.random.randn(8) * 100.0
    x, n_iter, resid = pcg_solve(op, rhs, D, max_iter=200, tol=1e-6)
    assert np.isfinite(resid)
    assert n_iter > 0


def test_pcg_vs_direct(tiny_operator):
    """PCG result should match direct solve for a well-conditioned system."""
    op, n_dof, _ = tiny_operator
    D = np.array([0.0])
    rhs = np.ones(8) * 100.0

    x_cg, n_iter, resid = pcg_solve(op, rhs, D, max_iter=500, tol=1e-12)
    assert n_iter > 0

    K_dense = np.zeros((8, 8))
    for i in range(8):
        e = np.eye(8)[i]
        K_dense[:, i] = op.apply(e, D)

    x_direct = np.linalg.solve(K_dense, rhs)

    err = np.linalg.norm(x_cg - x_direct) / max(np.linalg.norm(x_direct), 1e-30)
    assert err < 1e-6, f"PCG relative error too large: {err:.2e}"
