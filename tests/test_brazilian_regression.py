from __future__ import annotations

import numpy as np
import torch

from src.solvers.brazilian_disc_v1 import BrazilianDiscSolver


def test_brazilian_disc_regression_finite_and_damaged():
    """Fixed-seed short run must stay finite and develop damage (no NaN crash)."""
    torch.manual_seed(2026)
    np.random.seed(2026)

    solver = BrazilianDiscSolver(
        Nx=20, Ny=20, L_domain=60.0, R=25.0, flat_height=0.4,
        beta_crack=45.0, a_crack=5.0,
        E=30000.0, nu=0.25, sigma_t=6.0, K_Ic=31.62,
        loading_half_width=4.4, n_warmup=3, n_coupled=3, disp_step=3.0e-3,
        lambda_germano=0.3, lambda_elastic=0.5, lambda_fracture=0.3,
        lambda_damage=0.2, lambda_smooth=0.1, l_c=0.5, l_d=1.0, lr=2e-3,
    )
    assert solver.n_active > 0

    loads = []
    for t in range(6):
        disp = (t + 1) * solver.disp_step
        F, _is_warmup = solver.step(disp, t)
        loads.append(float(F))

    assert np.all(np.isfinite(loads))
    assert loads[0] > 0.0
    assert np.max(solver.D) <= 1.0
    assert np.all(np.isfinite(solver.strains))
    assert np.all(np.isfinite(solver.d_field))

    losses = solver.history["loss_total"]
    assert len(losses) == 3  # one entry per coupled step
    assert all(np.isfinite(x) for x in losses)
    assert np.max(solver.D) >= 0.99  # pre-cracked elements keep D at ~0.999
