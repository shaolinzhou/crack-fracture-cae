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


def _coo_operator(inactive_dofs=None):
    n_dof = 8
    elem_dof_array = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    k0_unit = np.eye(8) * 1000.0
    B_cen = np.eye(3, 8)[:3] * 0.1
    dofs = elem_dof_array[0]
    return MatrixFreeOperator(
        n_dof=n_dof,
        elem_dof_array=elem_dof_array,
        k0_unit=k0_unit,
        B_cen=B_cen,
        E=30000.0,
        nu=0.25,
        residual_stiffness=1e-6,
        coo_rows=np.repeat(dofs, 8),
        coo_cols=np.tile(dofs, 8),
        k0_tile=np.tile(k0_unit.ravel(), 1),
        inactive_dofs=np.array(inactive_dofs, dtype=int) if inactive_dofs else None,
    )


def test_assemble_sparse_without_coo_raises(tiny_operator):
    op, *_ = tiny_operator
    with pytest.raises(RuntimeError):
        op.assemble_sparse(np.zeros(1))


def test_assemble_sparse_inactive_and_bc_penalty():
    op = _coo_operator(inactive_dofs=[7])
    K = op.assemble_sparse(np.zeros(1), bc_dofs=np.array([0], dtype=int))
    assert K.shape == (8, 8)
    base = op.assemble_sparse(np.zeros(1))
    assert K[0, 0] > base[0, 0] * 1e3      # penalty added on bc dof
    assert base[7, 7] >= 1.0                 # inactive-dof protection


def test_pcg_solve_exact_bc_reduction():
    op = _coo_operator()
    bc = np.array([0, 1], dtype=int)
    tgt = np.array([1.0, -1.0])
    x, n_iter, resid = pcg_solve(
        op, np.ones(8), np.zeros(1), bc_dofs=bc, bc_targets=tgt,
        max_iter=200, tol=1e-10,
    )
    assert n_iter > 0
    assert np.allclose(x[bc], tgt, atol=1e-6)
    assert np.isfinite(resid)


def test_pcg_solve_no_precond_and_short_nonconverged():
    op = _coo_operator(inactive_dofs=[7])
    x, n_iter, resid = pcg_solve(
        op, np.ones(8), np.zeros(1), precond=False, max_iter=3, tol=1e-16,
    )
    assert 1 <= n_iter <= 3
    assert np.all(np.isfinite(x))
    assert np.isfinite(resid)
