from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.config import SolverConfig
from src.fem_utils import (
    plane_strain_C,
    rect_q4_stiffness_template,
    rect_b_matrix_center,
    mazars_equivalent_strain,
    von_mises_stress,
)
from src.damage_models import compute_damage_base
from src.networks import PhysicsScaleNetSolid, compute_features, compute_germano_signal, compute_loss
from src.pcg import MatrixFreeOperator, pcg_solve

logger = logging.getLogger(__name__)


class BaseCrackSolver:
    """Unified crack solver with optional NN-coupled scale correction.

    Supports both rectangular-grid (via active_mask) and general Q4 meshes
    (via elem_dof_array).  The core solve loop is identical regardless of
    mesh structure.
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.nn_active = False

        # filled by subclasses
        self.n_nodes: int = 0
        self.n_active: int = 0
        self.N_dof: int = 0
        self.E: float = 0.0
        self.nu: float = 0.0
        self.sigma_t: float = 0.0
        self.K_Ic: float = 0.0
        self.eps0: float = 0.0
        self.beta_soft: float = 0.0
        self.eps_eq_cap: float = 0.0
        self.char_len: float = 0.0

        self.elem_dof_array: np.ndarray = np.array([], dtype=int)
        self.C: np.ndarray = np.array([])
        self.k0_unit: np.ndarray = np.array([])
        self.B_cen: np.ndarray = np.array([])
        self._coo_rows: np.ndarray = np.array([], dtype=int)
        self._coo_cols: np.ndarray = np.array([], dtype=int)
        self._k0_tile: np.ndarray = np.array([], dtype=float)

        self.U: np.ndarray = np.array([])
        self.D: np.ndarray = np.array([])
        self.strains: np.ndarray = np.array([])
        self.stresses: np.ndarray = np.array([])
        self.d_field: np.ndarray = np.array([])

        self.fixed_dofs: np.ndarray = np.array([], dtype=int)
        self.top_disp_dofs: np.ndarray = np.array([], dtype=int)

        self.nn: PhysicsScaleNetSolid | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.mf_operator: MatrixFreeOperator | None = None

        self.history: dict[str, list] = {
            "load_disp": [],
            "max_damage": [],
            "loss_total": [],
            "mean_d": [],
        }

    # ── hooks for subclasses ──────────────────────────────────────────

    def on_solver_init(self) -> None:
        """Subclass hook: called after __init__ sets basic fields."""

    def get_bc_rhs(self, load_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Return (bc_dofs, bc_vals, F_rhs, bc_dofs_list)."""
        raise NotImplementedError

    def on_step_end(self, step_idx: int, load_factor: float, is_warmup: bool) -> None:
        """Subclass hook: called at the end of each step (visualization, I/O)."""

    # ── core solve ────────────────────────────────────────────────────

    def solve_elasticity(self, load_factor: float, use_pcg: bool = False) -> float:
        """Solve K·u = F for the current damage field.

        Args:
            load_factor: Scale factor applied to prescribed boundary displacements.
            use_pcg: If True, use matrix-free PCG; otherwise use sparse direct.

        Returns:
            Total reaction force magnitude.
        """
        bc_dofs, bc_vals, F, _ = self.get_bc_rhs(load_factor)
        K_pen = 1e10 * self.E

        F_with_bc = F.copy()
        for dof, val in zip(bc_dofs, bc_vals):
            F_with_bc[int(dof)] += K_pen * float(val)

        if use_pcg and self.mf_operator is not None:

            def D_getter():
                return self.D

            def matvec(u):
                f = self.mf_operator.apply(u, D_getter())
                for dof in bc_dofs:
                    f[int(dof)] += K_pen * u[int(dof)]
                return f

            from scipy.sparse.linalg import LinearOperator
            op = LinearOperator((self.N_dof, self.N_dof), matvec=matvec, dtype=float)
            self.U, success = _pcg_fixed(op, F_with_bc, self.N_dof, bc_dofs, tol=1e-8)
        else:
            if self.mf_operator is not None:
                K = self.mf_operator.assemble_sparse(self.D, bc_dofs)
            else:
                K = self._assemble_sparse(bc_dofs)
            from scipy.sparse.linalg import spsolve
            self.U = spsolve(K, F_with_bc)

        total_reaction = 0.0
        target_vals = load_factor * bc_vals[len(self.fixed_dofs):]
        for dof, val in zip(self.top_disp_dofs, target_vals):
            total_reaction += K_pen * (val - self.U[int(dof)])
        return abs(float(total_reaction))

    def _assemble_sparse(self, bc_dofs: list[int]):
        from scipy.sparse import csr_matrix
        D_clipped = np.clip(self.D, 0.0, 1.0 - self.config.residual_stiffness)
        scale = np.repeat((1.0 - D_clipped + self.config.residual_stiffness) * self.E, 64)
        elem_vals = scale * self._k0_tile
        bc_arr = np.array(bc_dofs, dtype=int)
        rows = np.concatenate([self._coo_rows, bc_arr])
        cols = np.concatenate([self._coo_cols, bc_arr])
        vals = np.concatenate([elem_vals, np.full(len(bc_dofs), 1e10 * self.E)])
        return csr_matrix((vals, (rows, cols)), shape=(self.N_dof, self.N_dof))

    def compute_strains_stresses(self) -> None:
        """Compute element-center strains and stresses from current U."""
        u_elem = self.U[self.elem_dof_array]
        self.strains = u_elem @ self.B_cen.T
        self.stresses = (self.strains @ self.C.T) * (1.0 - self.D)[:, None]

    def update_damage(self, delta_D_base: np.ndarray, d_field: np.ndarray, use_scaling: bool) -> None:
        """Update damage field with optional scale correction."""
        if use_scaling:
            scale = np.clip(self.config.scale_ratio ** d_field, 0.1, 10.0)
            dD = scale * delta_D_base
        else:
            dD = delta_D_base
        self.D = np.clip(self.D + dD, 0.0, 0.99999)

    # ── step ──────────────────────────────────────────────────────────

    def step(self, load_factor: float, step_idx: int, use_pcg: bool = False):
        """Perform one load step.

        Returns: (F_reaction, eps_eq, loss_t, is_warmup)
        """
        is_warmup = step_idx < self.config.n_warmup
        F = self.solve_elasticity(load_factor, use_pcg=use_pcg)
        self.compute_strains_stresses()

        delta_D, eps_eq = compute_damage_base(
            self.strains, self.D, self.eps0, self.beta_soft,
            self.eps_eq_cap, self.config.exp_clip,
            self.config.damping_warmup, self.config.damping_base,
            self.config.damping_fast,
            "warmup" if is_warmup else "coupled",
        )

        if is_warmup:
            self.update_damage(delta_D, self.d_field, use_scaling=False)
            self.on_step_end(step_idx, load_factor, True)
            return F, eps_eq, None, True

        if not self.nn_active:
            self.nn_active = True
            logger.info("Warmup finished at step %d; NN scale correction is active.", step_idx)

        if self.nn is not None:
            feats = self._get_features()
            d_pred = self.nn(feats)
            phi, phi_test = self._get_germano(delta_D)
            loss_t, *_ = self._compute_loss_fn(d_pred, phi, phi_test)
            self.optimizer.zero_grad()
            if torch.isfinite(loss_t):
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
                self.optimizer.step()
            with torch.no_grad():
                d_eval = self.nn(feats).numpy().flatten()
        else:
            d_eval = -0.5 + np.log(1.0 - np.clip(self.D, 0.0, 0.999)) * 0.3
            loss_t = None

        self.d_field = d_eval
        self.update_damage(delta_D, d_eval, use_scaling=True)
        self.on_step_end(step_idx, load_factor, False)
        return F, eps_eq, loss_t, False

    def _get_features(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_germano(self, delta_D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _compute_loss_fn(self, d_pred, phi, phi_test):
        raise NotImplementedError

    def run(self, use_pcg: bool = False) -> None:
        """Run the full simulation loop."""
        total = self.config.total_steps
        logger.info("Starting simulation: %d steps (%d warmup + %d coupled)",
                     total, self.config.n_warmup, self.config.n_coupled)

        for t in range(total):
            load_factor = (t + 1) / total
            F, eps_eq, loss_t, is_warmup = self.step(load_factor, t, use_pcg=use_pcg)
            self.history["load_disp"].append((load_factor, F))
            self.history["max_damage"].append(float(np.max(self.D)))
            if not is_warmup and loss_t is not None:
                self.history["loss_total"].append(
                    loss_t.item() if torch.isfinite(loss_t) else float("inf")
                )
                self.history["mean_d"].append(float(np.mean(self.d_field)))

            if (t + 1) % self.config.output_stride == 0 or t == total - 1 or t == 0:
                tag = "Warmup" if is_warmup else "Coupled"
                extra = ""
                if self.history["loss_total"]:
                    extra = f" | Loss={self.history['loss_total'][-1]:.2e} | <d>={self.history['mean_d'][-1]:.4f}"
                logger.info("[%s] Step %3d/%d | max(D)=%.4f | cracked=%d%s",
                            tag, t + 1, total, np.max(self.D), int(np.sum(self.D > 0.99)), extra)

        logger.info("Simulation finished.")


def _pcg_fixed(op, rhs, n, fixed_dofs, tol=1e-8, max_iter=500):
    """Simple CG with penalty-fixed dofs, returns (x, converged_bool)."""
    x = np.zeros(n, dtype=float)
    r = rhs.copy()
    for dof in fixed_dofs:
        r[dof] = 0.0
    r_norm0 = np.linalg.norm(r)
    if r_norm0 < 1e-30:
        return x, True

    p = r.copy()
    rsold = np.dot(r, r)
    for i in range(max_iter):
        Ap = op(p)
        for dof in fixed_dofs:
            Ap[dof] = p[dof] * 1e10 * 3e4
        alpha = rsold / max(np.dot(p, Ap), 1e-30)
        x += alpha * p
        r -= alpha * Ap
        for dof in fixed_dofs:
            r[dof] = 0.0
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) / r_norm0 < tol:
            return x, True
        p = r + (rsnew / max(rsold, 1e-30)) * p
        rsold = rsnew
    return x, False
