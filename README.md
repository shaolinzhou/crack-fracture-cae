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
├── solvers/            standalone runnable solvers & prototypes
│   ├── crack.py              v2.0: nonlocal ε_eq + adaptive damping + full 5-term loss
│   ├── brazilian_disc_v1.py  v1.0: base BrazilianDiscSolver class + driver
│   ├── brazilian_splitting_solver.py   early 100×100 prototype
│   ├── hybrid_cae_*.py       hybrid prototypes (subclass BrazilianDiscSolver)
│   └── run_brazilian_*.py    demo / mesh adapters
├── src/                shared numerical library (fracture line)
│   ├── fem_utils.py         plane-strain C, Q4 B/k, invariants
│   ├── damage_models.py     Mazars target, adaptive damping, Gf calibration
│   ├── networks.py          PhysicsScaleNetSolid + Germano signal + 5-term loss
│   ├── pcg.py               MatrixFreeOperator + PCG (exact-BC, Jacobi)
│   ├── solver.py            unified BaseCrackSolver loop
│   └── config.py            SolverConfig dataclass
├── tests/              13 unit tests (fracture line)  [pytest]
├── examples/           run_pcg_demo.py: PCG vs spsolve consistency
├── FEA/                DAT(GiD)-driven general unstructured-Q4 solver
├── data/               input meshes: c1.dat (primary), d1.dat
└── docs/
    ├── FRACTURE_CAE_MATHEMATICS_AND_IMPLEMENTATION.md   ← full math report
    ├── THEORETICAL_FOUNDATION.md                        ← operator-algebra theory
    ├── DEVELOPMENT_REVIEW.md                            ← dev retrospective
    └── figures/                                         representative outputs
```

## Install

```powershell
pip install numpy scipy matplotlib torch        # or: pip install -e ".[dev]"
```

## Quick start

```powershell
# --- flagship v2.0 Brazilian disc (outputs to current dir snapshots\) ---
cd solvers
python crack.py

# --- v1.0 (snapshots_brazilian/ + brazilian_disc_result.png) ---
python brazilian_disc_v1.py

# --- PCG vs spsolve consistency check (matrix-free) ---
cd ..
python examples/run_pcg_demo.py

# --- general DAT-driven solver (any unstructured Q4 mesh) ---
python FEA/run_fea.py data/c1.dat --warmup 10 --coupled 0
```

## Tests

```powershell
python -m pytest          # 13 passed
```

Coverage: elastic C matrix analytic check, Mazars equivalent strain identities,
monotonic damage, fracture-energy calibration, PCG vs direct solve (<1e-6),
matrix-free operator self-consistency (<1e-10).

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

## License

Apache-2.0. See [LICENSE](LICENSE).

*Theory document: `docs/FRACTURE_CAE_MATHEMATICS_AND_IMPLEMENTATION.md` contains
the full paper-style derivation and a symbol → code mapping index.*
