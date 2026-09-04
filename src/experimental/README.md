# src/experimental — research-only demos & prototypes

Status: **DEPRECATED / research lineage only**. These modules are kept in the
repository for provenance and quick experiments but are **excluded from the
installed package** (see `pyproject.toml` `[tool.setuptools.packages.find]
exclude`). They are not covered by the test suite nor the coverage gate.

| Module | What it is | Status |
| --- | --- | --- |
| `brazilian_splitting_solver.py` | earliest 100×100 prototype | legacy |
| `hybrid_cae_solver_v1.py` | matrix-free logic sketch | prototype (not runnable end-to-end) |
| `hybrid_cae_solver_final.py` | hybrid subclass of v1 baseline | demo |
| `hybrid_cae_stable.py` | stabilized hybrid prototype | demo |
| `run_brazilian_demo.py` | random-strain-field visual demo | demo |
| `run_brazilian_real.py` | mesh adapter sketch | prototype |

To run any of them locally (from the repository root):

```bash
python -m src.experimental.hybrid_cae_stable        # example
```

Decision for B4: **documented retention + exclusion from the wheel**; full
removal is deferred unless these modules are revived for a future research
line.
