"""
PCG solver demo — verify that matrix-free PCG matches spsolve results
=============================================================
Usage: python -m src.pcg_demo
"""
from __future__ import annotations

import logging
import time

import numpy as np

from src.config import SolverConfig
from src.fem_utils import (
    plane_strain_C,
    rect_q4_stiffness_template,
    rect_b_matrix_center,
)
from src.damage_models import compute_damage_parameters
from src.pcg import MatrixFreeOperator, pcg_solve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_rectangular_mesh(Nx, Ny, L_domain, R, E, nu, sigma_t, K_Ic, loading_half_width):
    """Construct a Brazilian disc rectangular mesh, returning all solver fields."""
    dx = L_domain / (Nx - 1)
    dy = L_domain / (Ny - 1)
    Xc = Yc = L_domain / 2.0

    Ne_x, Ne_y = Nx - 1, Ny - 1
    active = []
    elem_ji = {}
    for j in range(Ne_y):
        for i in range(Ne_x):
            xc = (i + 0.5) * dx
            yc = (j + 0.5) * dy
            if (xc - Xc) ** 2 + (yc - Yc) ** 2 <= R ** 2:
                e = j * Ne_x + i
                active.append(e)
                elem_ji[e] = (j, i)

    n_active = len(active)
    N_dof = 2 * Nx * Ny

    elem_dof_array = np.zeros((n_active, 8), dtype=int)
    for idx, e in enumerate(active):
        j, i = elem_ji[e]
        n0 = j * Nx + i
        n1 = j * Nx + i + 1
        n2 = (j + 1) * Nx + i + 1
        n3 = (j + 1) * Nx + i
        elem_dof_array[idx] = [2 * n0, 2 * n0 + 1, 2 * n1, 2 * n1 + 1,
                               2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1]

    C = plane_strain_C(E, nu)
    k0 = rect_q4_stiffness_template(dx, dy, C)
    k0_unit = rect_q4_stiffness_template(dx, dy, plane_strain_C(1.0, nu))
    B_cen = rect_b_matrix_center(dx, dy)

    k0f = k0_unit.flatten()
    coo_rows = np.zeros(n_active * 64, dtype=int)
    coo_cols = np.zeros(n_active * 64, dtype=int)
    k0_tile = np.tile(k0f, n_active)
    for idx in range(n_active):
        dofs = elem_dof_array[idx]
        coo_rows[idx * 64:(idx + 1) * 64] = np.repeat(dofs, 8)
        coo_cols[idx * 64:(idx + 1) * 64] = np.tile(dofs, 8)

    is_active_dof = np.zeros(N_dof, dtype=bool)
    is_active_dof[elem_dof_array.flatten()] = True

    top_nodes, bottom_nodes = [], []
    top_center, bottom_center = None, None
    min_top = min_bot = 1e9
    for i in range(Nx):
        xn = i * dx
        if abs(xn - Xc) > loading_half_width:
            continue
        j_top, j_bot = -1, Ny
        for j in range(Ny):
            if is_active_dof[2 * (j * Nx + i)]:
                if j < j_bot:
                    j_bot = j
                if j > j_top:
                    j_top = j
        if j_top > 0:
            node = j_top * Nx + i
            top_nodes.append(node)
            if abs(xn - Xc) < min_top:
                min_top = abs(xn - Xc)
                top_center = node
        if j_bot < Ny:
            node = j_bot * Nx + i
            bottom_nodes.append(node)
            if abs(xn - Xc) < min_bot:
                min_bot = abs(xn - Xc)
                bottom_center = node

    return {
        "Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy, "L_domain": L_domain,
        "R": R, "Xc": Xc, "Yc": Yc, "E": E, "nu": nu,
        "N_dof": N_dof, "n_active": n_active,
        "elem_dof_array": elem_dof_array,
        "C": C, "k0": k0, "k0_unit": k0_unit, "B_cen": B_cen,
        "coo_rows": coo_rows, "coo_cols": coo_cols, "k0_tile": k0_tile,
        "top_nodes": top_nodes, "bottom_nodes": bottom_nodes,
        "top_center": top_center, "bottom_center": bottom_center,
        "active": active, "elem_ji": elem_ji,
    }


def main():
    logger.info("=" * 60)
    logger.info("PCG vs spsolve 一致性验证 — 巴西圆盘 (80×80)")
    logger.info("=" * 60)

    mesh = build_rectangular_mesh(
        Nx=80, Ny=80, L_domain=60.0, R=25.0,
        E=30000.0, nu=0.25, sigma_t=6.0, K_Ic=31.62,
        loading_half_width=4.4,
    )

    config = SolverConfig(
        n_warmup=10, n_coupled=0,
        residual_stiffness=1e-6,
        damping_warmup=0.3,
    )

    D = np.zeros(mesh["n_active"])
    eps0, _ = compute_damage_parameters(
        mesh["E"], mesh["nu"], 6.0, 31.62, mesh["dx"]
    )

    is_active_dof = np.zeros(mesh["N_dof"], dtype=bool)
    is_active_dof[mesh["elem_dof_array"].flatten()] = True
    inactive_dofs = np.where(~is_active_dof)[0]

    mf_op = MatrixFreeOperator(
        n_dof=mesh["N_dof"],
        elem_dof_array=mesh["elem_dof_array"],
        k0_unit=mesh["k0_unit"],
        B_cen=mesh["B_cen"],
        E=mesh["E"],
        nu=mesh["nu"],
        residual_stiffness=config.residual_stiffness,
        coo_rows=mesh["coo_rows"],
        coo_cols=mesh["coo_cols"],
        k0_tile=mesh["k0_tile"],
        inactive_dofs=inactive_dofs,
    )

    # ── build BC ──
    fixed_dofs = []
    fixed_targets = []
    for n in mesh["bottom_nodes"]:
        fixed_dofs.append(2 * n + 1)
        fixed_targets.append(0.0)
    if mesh["bottom_center"] is not None:
        fixed_dofs.append(2 * mesh["bottom_center"])
        fixed_targets.append(0.0)
    top_disp = 3e-3
    for n in mesh["top_nodes"]:
        fixed_dofs.append(2 * n + 1)
        fixed_targets.append(-top_disp)

    K_pen = 1e10 * mesh["E"]
    F = np.zeros(mesh["N_dof"])
    for n in mesh["top_nodes"]:
        F[2 * n + 1] += K_pen * (-top_disp)

    # ── spsolve reference ──
    logger.info("Solving with spsolve (sparse direct)...")
    t0 = time.perf_counter()
    K_sparse = mf_op.assemble_sparse(D, fixed_dofs)
    from scipy.sparse.linalg import spsolve
    U_ref = spsolve(K_sparse, F)
    t_spsolve = time.perf_counter() - t0

    # ── PCG (reduced system, exact BCs) ──
    bc_arr = np.array(fixed_dofs, dtype=int)
    bc_targets_arr = np.array(fixed_targets, dtype=float)
    F_no_penalty = np.zeros_like(F)
    logger.info("Solving with matrix-free PCG (reduced system)...")
    t0 = time.perf_counter()
    U_pcg, n_iter, resid = pcg_solve(
        mf_op, F_no_penalty, D,
        bc_dofs=bc_arr, bc_targets=bc_targets_arr,
        max_iter=500, tol=1e-8,
        precond=True,
    )
    t_pcg = time.perf_counter() - t0

    # ── compare ──
    err = np.linalg.norm(U_pcg - U_ref) / max(np.linalg.norm(U_ref), 1e-30)
    logger.info("Results:")
    logger.info("  spsolve:  %.4f s", t_spsolve)
    logger.info("  PCG:      %.4f s  (%d iters, final resid=%.2e)", t_pcg, n_iter, resid)
    logger.info("  rel error: %.2e", err)

    if err < 1e-6:
        logger.info(">>> PASS: PCG and spsolve results match. ✓")
    else:
        logger.warning(">>> WARNING: PCG error > 1e-6. Check implementation.")

    # ── verify operator self-consistency ──
    D_test = np.random.uniform(0, 0.5, mesh["n_active"])
    u_test = np.random.randn(mesh["N_dof"]) * 0.01
    f1 = mf_op.apply(u_test, D_test)
    K_test = mf_op.assemble_sparse(D_test)
    f2 = K_test.dot(u_test)

    all_but_inactive = np.setdiff1d(np.arange(mesh["N_dof"]), inactive_dofs)
    op_err = np.linalg.norm(f1[all_but_inactive] - f2[all_but_inactive]) / max(np.linalg.norm(f2[all_but_inactive]), 1e-30)
    logger.info("Operator self-consistency check:")
    logger.info("  ||K*u (matrix-free) - K*u (assembled)|| / ||K*u|| = %.2e", op_err)
    if op_err < 1e-10:
        logger.info(">>> PASS: Operator is self-consistent. ✓")
    else:
        logger.warning(">>> WARNING: Operator inconsistency detected.")


if __name__ == "__main__":
    main()
