from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEA = ROOT / "FEA"


def _load_fea():
    """Load the repo-local FEA solver modules (not part of the installed src pkg)."""
    sys.path.insert(0, str(FEA))
    import dat_parser
    import solver
    return dat_parser, solver


def _multimaterial_dat() -> str:
    # 2x1 = 2 Q4 elements with two different materials.
    return """6 2
coordinates
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
5 2.0 0.0
6 2.0 1.0
end coordinates
element
1 1 2 3 4 1
2 2 5 6 3 2
end element
material properties
1 1000.0 0.2 5.0 1.0
2 3000.0 0.3 8.0 2.0
end material properties
Moment-Load
Node, 1, UX, 0.0, UY, 0.0
end moment-load
presure
Node, 5, -0.05
Node, 6, -0.05
end presure
"""


def test_multimaterial_dat_solver_end_to_end(tmp_path: Path):
    dp, s = _load_fea()
    dat = tmp_path / "mm.dat"
    dat.write_text(_multimaterial_dat(), encoding="utf-8")

    model = dp.read_dat(dat)
    assert len(model.materials) == 2

    cfg = s.SolverConfig(n_warmup=1, n_coupled=0, output_stride=1)
    solver = s.DatCrackSolver(model, output_dir=tmp_path / "out", config=cfg)
    assert len(solver.C_by_mat) == 2

    F, eps_eq, loss, is_warmup = solver.step(1.0, 0)
    assert is_warmup is True
    assert F > 0.0
    assert eps_eq.shape == (model.n_elements,)
    assert solver.strains.shape == (model.n_elements, 3)
    assert np.max(np.abs(solver.strains)) > 0.0  # strain field is populated
    assert solver.D.shape == (model.n_elements,)
