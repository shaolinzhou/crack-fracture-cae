# Crack-Fracture-CAE

Scale-invariant **hybrid (physics + neural) damage CAE engine** for quasi-brittle
fracture simulation, flagship case: **Brazilian disc splitting** (ISRM) with an
inclined pre-crack (mode I–II).

This repository is the self-contained **fracture/damage line** of the "Project
Crack — AI-CAE" family. It implements a complete pipeline from continuum damage
mechanics to an online self-supervised neural closure:

```
Continuum damage (Mazars) + Q4 FEM (plane strain)
   → damage-degraded secant system  K(D) u = F   (spsolve  or  matrix-free PCG)
   → two-phase load stepping        (pure-FEM warmup → NN-coupled)
   → scale-invariant damage-rate closure   ΔD = (l/L₀)^d(x) · ΔD_Mazars
   → PhysicsScaleNetSolid (5 mechanical invariants → scale exponent d ≤ −1/2)
   → Germano double-filter self-supervision  (no ground-truth labels)
```

| Result figure (docs/figures) | Description |
| --- | --- |
| `brazilian_disc_result.png` | v1.0 6-panel: D, σ_xx, σ_yy, d(x), load–displacement, loss/<d> |
| `brazilian_splitting_analysis.png` | early 100×100 prototype analysis |

## Repository layout

```
├── src/                    installable package `crack_fracture_cae` (import name `src`)
│   ├── solvers/            maintained mainline drivers (thin; kernels live in the package)
│   │   ├── crack.py              v2.0 flagship (python -m src.solvers.crack)
│   │   └── brazilian_disc_v1.py  v1.0 baseline (python -m src.solvers.brazilian_disc_v1)
│   ├── experimental/       DEPRECATED prototypes/demos (research lineage only)
│   │   ├── brazilian_splitting_solver.py   early 100×100 prototype
│   │   ├── hybrid_cae_solver_v1.py         matrix-free logic sketch
│   │   ├── hybrid_cae_solver_final.py      hybrid prototype (subclasses v1)
│   │   ├── hybrid_cae_stable.py            stabilized hybrid prototype
│   │   └── run_brazilian_demo.py / _real.py  demo / mesh adapters
│   ├── fem_utils.py         plane-strain C, Q4 B/k, invariants
│   ├── damage_models.py     Mazars target, adaptive damping, Gf calibration
│   ├── networks.py          PhysicsScaleNetSolid + Germano signal + 5-term loss
│   ├── pcg.py               MatrixFreeOperator + PCG (exact-BC, Jacobi)
│   ├── solver.py            unified BaseCrackSolver loop
│   ├── pcg_demo.py          PCG vs spsolve consistency demo
│   ├── cli.py               console entry points (project.scripts)
│   └── config.py            SolverConfig dataclass
├── tests/              34 unit tests (core coverage gate >= 85%)  [pytest]
├── FEA/                DAT(GiD)-driven general unstructured-Q4 solver (repo-local CLI)
├── data/               input meshes: c1.dat (primary), d1.dat
└── docs/
    ├── THEORETICAL_FOUNDATION.md        ← operator-algebra theory (中文)
    ├── THEORETICAL_FOUNDATION.en.md     ← English backup translation
    └── figures/                         representative outputs
```

## Install

```powershell
pip install numpy scipy matplotlib torch        # runtime deps
pip install -e ".[dev]"                         # install package + dev tools
```

After `pip install -e .` the package and console scripts are available from any
directory:

```powershell
crack-cae            # == python -m src.solvers.crack          (v2.0 flagship)
crack-cae-v1         # == python -m src.solvers.brazilian_disc_v1
crack-pcg-demo       # == python -m src.pcg_demo
```

## Quick start (no install required, from the repository root)

```powershell
# --- flagship v2.0 Brazilian disc (outputs to current dir snapshots\) ---
python -m src.solvers.crack

# --- v1.0 ---
python -m src.solvers.brazilian_disc_v1

# --- PCG vs spsolve consistency check (matrix-free) ---
python -m src.pcg_demo

# --- general DAT-driven solver (any unstructured Q4 mesh) ---
python FEA/run_fea.py data/c1.dat --warmup 10 --coupled 0
```

## Tests

```powershell
python -m pytest          # 34 passed (src-core coverage >= 85%)
```

Coverage includes: elastic C matrix analytic check, Mazars equivalent-strain
identities & monotonic damage, fracture-energy calibration, PCG vs direct solve
(<1e-6), matrix-free operator self-consistency (<1e-10), loss-gradient
finite-difference check, PCG non-convergence guard, DAT-parser regression
(c1/d1 + malformed files), multi-material DAT solver end-to-end, and a
fixed-seed Brazilian-disc regression. `src/` numerical-core coverage is gated
at >= 85% via `--cov-fail-under`.

## Key model & method summary

- **Damage**: isotropic scalar Mazars with equivalent tensile strain
  `ε_eq = sqrt(⟨ε₁⟩₊² + ⟨ε₂⟩₊²)` and exponential softening
  `D = 1 − (ε₀/ε_eq)·exp(−β(ε_eq−ε₀))`;
  `ε₀ = σ_t/E`, `β = σ_t / (Gf/lc − σ_t²/2E)`, `Gf = K_Ic²(1−ν²)/E`.
- **Regularization**: residual stiffness `1e−6`, exp clipping, strain cap,
  adaptive damage damping (0.3/0.5/0.7), saturation `D≤0.99999`, nonlocal ε_eq
  (3×3 box on structured grids / cKDTree ball on unstructured).
- **Linear algebra**: per-step damage-degraded secant stiffness; sparse direct
  `spsolve`, or matrix-free element-wise `K·u` + PCG (exact BC elimination +
  Jacobi preconditioning) — bitwise consistent with the sparse assembly.
- **Neural closure**: local scale exponent `d(x) = −1/2 − softplus(net(x))`
  from 5 invariants `[D, tanh η, tanh θ̄, tanh(ε_eq/ε₀−1), tanh(lc|∇D|)]`;
  5-term loss = Germano consistency + elastic anchor + fracture anchor +
  constitutive prior + smoothness; trained online during the coupled phase.
- **General FEA**: DAT-driven multi-material unstructured Q4, cKDTree-based
  nonlocal/gradient, Wall (pre-crack) initialization, auto UX anchor, GiD
  `.msh`/`.res` export, CPU-only fallbacks when torch/matplotlib are absent.

## Architecture roadmap (extension seams)

- **Single numerical core**: all kernels now live in the installable `src` package;
  the v2/v1 drivers and the DAT-driven FEA solver are thin orchestrators over it.
- Planned seams kept open (see `docs/` theory and `src/solver.py::BaseCrackSolver`):
  matrix-free **Term B** (neural damage gradient) inside the linear operator,
  element-wise `torch.vmap` parallelism, multi-crack / 3D extensions, and batch
  parameterized runs. These are API-compatible next steps rather than rewrites.

## Calibration (C1, reporting)

ISRM (1978) splitting strength: `σ_t = 2 P_peak / (π D t)`. On the intact
D=50 mm disc (σ_t=6 MPa input) the production staggered engine is
mesh-converged at ~0.83 of the ISRM reference (Nx96, C1 run
`benchmarks/run_isrm_calibration.py`). For **reporting** purposes the raw
back-calculation is multiplied by a calibration factor
`k = 1/0.8298 ≈ 1.205` (`src/calibration.py::calibrated_strength`) so that the
reported splitting strength matches the input tensile strength. This is an
engineering reporting factor, not a material constant; its physical origin
(early softening) is tracked as an A-series roadmap item.

## Artifacts & versioning policy

- **Source and small reference data** (`data/*.dat`, `src/`, `solvers/`) are versioned.
- **Regenerable runtime outputs** (`snapshots*/`, `FEA/results/`, `*.res`, `*.msh`) are
  git-ignored by default. The four-angle Brazilian-disc cloud set under
  `snapshots_brazilian/` is a one-off **curated reference artifact** and is versioned;
  it can be regenerated with `python notes/regenerate_brazilian_sweep.py`.
- For large result collections (movies, 100+ MB fields), prefer **GitHub Releases**
  over committing files; keep the repository lean.

## License

Apache-2.0. See [LICENSE](LICENSE).

*The full paper-style math report (`FRACTURE_CAE_MATHEMATICS_AND_IMPLEMENTATION.md`)
and the development retrospective (`DEVELOPMENT_REVIEW.md`) are kept locally under
`notes/`, which is intentionally excluded from this repository.*
