from __future__ import annotations

from src.config import SolverConfig


def test_solver_config_defaults_and_totals():
    cfg = SolverConfig()
    assert cfg.n_warmup == 500
    assert cfg.n_coupled == 200
    assert cfg.total_steps == 700
    assert cfg.residual_stiffness == 1e-6
    assert cfg.output_stride == 10
    assert cfg.lam_g == 0.3
