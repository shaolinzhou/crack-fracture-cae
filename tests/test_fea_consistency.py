from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEA = ROOT / "FEA"


def _load_fea():
    sys.path.insert(0, str(FEA))
    import dat_parser  # noqa: F401
    import solver
    return solver


def test_fea_uses_src_elastic_kernels():
    s = _load_fea()
    from src.config import SolverConfig
    from src.fem_utils import plane_strain_C, q4_unit_stiffness

    E, nu = 30000.0, 0.25
    assert np.allclose(s.plane_strain_C(E, nu), plane_strain_C(E, nu))

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    k0, bc, area = s.q4_unit_stiffness(coords, nu)
    k0_ref, bc_ref, area_ref = q4_unit_stiffness(coords, nu)
    assert np.allclose(k0, k0_ref)
    assert np.allclose(bc, bc_ref)
    assert np.allclose(area, area_ref)

    # config is the shared one after the P0-1 convergence
    assert s.SolverConfig is SolverConfig
