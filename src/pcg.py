from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator


class MatrixFreeOperator:
    """Matrix-free representation of the tangent stiffness operator K·u.

    Instead of assembling the full sparse matrix K, this class computes
    the matrix-vector product K·u element-by-element on the fly, reducing
    memory from O(N²) to O(N).
    """

    def __init__(
        self,
        n_dof: int,
        elem_dof_array: np.ndarray,
        k0_unit: np.ndarray,
        B_cen: np.ndarray,
        E: float,
        nu: float,
        residual_stiffness: float = 1e-6,
        coo_rows: np.ndarray | None = None,
        coo_cols: np.ndarray | None = None,
        k0_tile: np.ndarray | None = None,
        inactive_dofs: np.ndarray | None = None,
    ):
        self.n_dof = n_dof
        self.elem_dof_array = elem_dof_array
        self.k0_unit = k0_unit
        self.B_cen = B_cen
        self.E = E
        self.nu = nu
        self.residual_stiffness = residual_stiffness
        self.n_active = elem_dof_array.shape[0]

        self._coo_rows = coo_rows
        self._coo_cols = coo_cols
        self._k0_tile = k0_tile
        self._inactive_dofs = inactive_dofs if inactive_dofs is not None else np.array([], dtype=int)
        self._use_sparse_fallback = coo_rows is not None

        C_full = self._plane_strain_C(E, nu)
        self._C_unit = C_full / E

    @staticmethod
    def _plane_strain_C(E: float, nu: float) -> np.ndarray:
        f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return np.array([
            [f * (1.0 - nu), f * nu, 0.0],
            [f * nu, f * (1.0 - nu), 0.0],
            [0.0, 0.0, f * (1.0 - 2.0 * nu) / 2.0],
        ])

    def apply(self, u: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Compute K(D) · u element-by-element (Term A only).

        Uses the precomputed k0_unit (2×2 Gauss) scaled by (1-D+res)*E,
        guaranteeing bitwise consistency with the sparse assembly.
        """
        f_int = np.zeros(self.n_dof, dtype=float)

        u_elem = u[self.elem_dof_array]
        D_clipped = np.clip(D, 0.0, 1.0 - self.residual_stiffness)
        scale = (1.0 - D_clipped + self.residual_stiffness) * self.E

        f_elem = (self.k0_unit @ u_elem[:, :, np.newaxis]).squeeze(-1) * scale[:, np.newaxis]

        for idx in range(self.n_active):
            dofs = self.elem_dof_array[idx]
            f_int[dofs] += f_elem[idx]

        if len(self._inactive_dofs):
            f_int[self._inactive_dofs] = 0.0

        return f_int

    def to_scipy_linear_operator(self, D_getter):
        """Wrap as a SciPy LinearOperator for use with scipy.sparse.linalg.cg."""

        def matvec(u):
            return self.apply(u, D_getter())

        return LinearOperator((self.n_dof, self.n_dof), matvec=matvec, dtype=float)

    def assemble_sparse(self, D: np.ndarray, bc_dofs: np.ndarray | None = None) -> object:
        """Assemble sparse matrix using precomputed COO indices + inactive DOF protection."""
        from scipy.sparse import csr_matrix

        if self._k0_tile is None:
            raise RuntimeError("COO cache not available; cannot assemble sparse matrix")

        D_clipped = np.clip(D, 0.0, 1.0 - self.residual_stiffness)
        scale = np.repeat((1.0 - D_clipped + self.residual_stiffness) * self.E, 64)
        elem_vals = scale * self._k0_tile

        r, c, v = list(self._coo_rows), list(self._coo_cols), list(elem_vals)

        for d in self._inactive_dofs:
            r.append(d)
            c.append(d)
            v.append(1.0)

        if bc_dofs is not None and len(bc_dofs):
            pen = 1e10 * self.E
            for d in bc_dofs:
                r.append(int(d))
                c.append(int(d))
                v.append(pen)

        return csr_matrix((v, (r, c)), shape=(self.n_dof, self.n_dof))


def pcg_solve(
    operator: MatrixFreeOperator,
    rhs: np.ndarray,
    D: np.ndarray,
    bc_dofs: np.ndarray | None = None,
    bc_targets: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
    precond: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Preconditioned Conjugate Gradient solver (matrix-free, reduced system).

    Eliminates prescribed DOFs and solves only for free DOFs.
    BCs are enforced exactly (not via penalty), avoiding ill-conditioning.

    Args:
        operator: MatrixFreeOperator instance.
        rhs: External force vector applied at free DOFs (zero at BC DOFs).
        D: Current damage field.
        bc_dofs: DOF indices with prescribed values.
        bc_targets: Target values at bc_dofs.
        max_iter: Maximum CG iterations.
        tol: Relative residual tolerance.
        precond: Whether to use Jacobi preconditioning.

    Returns:
        (full_solution, iterations_used, final_residual_norm)
    """
    n = len(rhs)
    free = np.ones(n, dtype=bool)
    if bc_dofs is not None:
        free[bc_dofs] = False
    if len(operator._inactive_dofs):
        free[operator._inactive_dofs] = False
    free_idx = np.where(free)[0]

    if len(free_idx) == 0:
        return np.zeros(n, dtype=float), 0, 0.0

    x = np.zeros(n, dtype=float)
    if bc_dofs is not None and bc_targets is not None:
        x[bc_dofs] = bc_targets

    u_bc = np.zeros(n, dtype=float)
    if bc_dofs is not None:
        u_bc[bc_dofs] = bc_targets if bc_targets is not None else 0.0

    f_bc = operator.apply(u_bc, D)

    rhs_eff = rhs[free_idx] - f_bc[free_idx]

    residual_norm0 = np.linalg.norm(rhs_eff)
    if residual_norm0 < 1e-30:
        return x, 0, 0.0

    if precond:
        diag = _build_free_precond(operator, D, free_idx)
        z = rhs_eff / diag
    else:
        z = rhs_eff.copy()

    u_free = np.zeros(len(free_idx), dtype=float)
    p = z.copy()
    rz = np.dot(rhs_eff, z)

    for i in range(max_iter):
        u_full = np.zeros(n, dtype=float)
        u_full[free_idx] = p
        Ap = operator.apply(u_full, D)[free_idx]
        alpha = rz / max(np.dot(p, Ap), 1e-30)
        u_free = u_free + alpha * p
        rhs_eff = rhs_eff - alpha * Ap
        residual_norm = np.linalg.norm(rhs_eff)
        if residual_norm / residual_norm0 < tol:
            x[free_idx] = u_free
            return x, i + 1, residual_norm

        if precond:
            z = rhs_eff / diag
        else:
            z = rhs_eff.copy()
        rz_new = np.dot(rhs_eff, z)
        beta = rz_new / max(rz, 1e-30)
        p = z + beta * p
        rz = rz_new

    x[free_idx] = u_free
    return x, max_iter, np.linalg.norm(rhs_eff)


def _build_free_precond(
    operator: MatrixFreeOperator, D: np.ndarray, free_idx: np.ndarray,
) -> np.ndarray:
    """Build Jacobi preconditioner diagonal for the reduced system."""
    d = np.clip(D, 0.0, 1.0 - operator.residual_stiffness)
    diag_full = np.ones(operator.n_dof, dtype=float)
    for idx in range(operator.n_active):
        dofs = operator.elem_dof_array[idx]
        k_local = (1.0 - d[idx] + operator.residual_stiffness) * operator.k0_unit
        for ii in range(4):
            for jj in range(8):
                diag_full[dofs[2 * ii]] += abs(k_local[2 * ii, jj])
                diag_full[dofs[2 * ii + 1]] += abs(k_local[2 * ii + 1, jj])
    return np.maximum(diag_full[free_idx], 1e-12)
