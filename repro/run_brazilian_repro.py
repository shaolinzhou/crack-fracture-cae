"""B2 — reproducibility harness for the Brazilian-disc v1 solver.

Unifies seeding and records, for every run, the git HEAD, the full solver
configuration, and outcome metrics in a single JSON so runs can be reproduced
and compared.

Usage (from repository root):
    python repro/run_brazilian_repro.py --seed 2026 --Nx 40 --steps 24
    python repro/run_brazilian_repro.py --seed 7    --Nx 40 --steps 24

Outputs are written to ``results/brazilian_repro/<seed>/`` (git-ignored):
    load_disp.csv, final_fields.png, run_meta.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solvers.brazilian_disc_v1 import BrazilianDiscSolver  # noqa: E402
from src.calibration import (  # noqa: E402
    calibrated_strength,
    splitting_strength,
)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def run_case(seed: int, Nx: int, steps: int, out_dir: Path) -> dict:
    seed_all(seed)
    solver = BrazilianDiscSolver(
        Nx=Nx, Ny=Nx, L_domain=60.0, R=25.0, flat_height=0.4,
        beta_crack=45.0, a_crack=5.0, E=30000.0, nu=0.25,
        sigma_t=6.0, K_Ic=31.62, loading_half_width=4.4,
        n_warmup=steps, n_coupled=0, disp_step=5e-3,
        lambda_germano=0.3, lambda_elastic=0.5, lambda_fracture=0.3,
        lambda_damage=0.2, lambda_smooth=0.1, l_c=0.5, l_d=1.0, lr=2e-3,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    loads = []
    for t in range(steps):
        disp = (t + 1) * solver.disp_step
        F, _is_warmup = solver.step(disp, t)
        loads.append((float(disp), float(F)))

    # load-displacement csv
    arr = np.array(loads)
    np.savetxt(out_dir / "load_disp.csv", arr, delimiter=",",
               header="disp_mm,load", comments="")

    # final fields image
    ne = solver.Nx - 1
    D = np.full((solver.Ny - 1, ne), np.nan)
    S = np.full((solver.Ny - 1, ne), np.nan)
    for k, e in enumerate(solver.active):
        j, i = solver.elem_ji[e]
        D[j, i] = solver.D[k]
        sxx, syy, sxy = solver.stresses[k]
        szz = solver.nu * (sxx + syy)
        S[j, i] = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                 + (szz - sxx) ** 2) + 3.0 * sxy ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(D, origin="lower", cmap="inferno", vmin=0, vmax=1,
                        extent=[0, solver.L, 0, solver.L])
    axes[0].set_title("Damage D (final)")
    axes[0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0])
    im2 = axes[1].imshow(S, origin="lower", cmap="viridis",
                         extent=[0, solver.L, 0, solver.L])
    axes[1].set_title("Von Mises (MPa, final)")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out_dir / "final_fields.png", dpi=120)
    plt.close(fig)

    return {
        "seed": int(seed),
        "git_head": git_head(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "Nx": Nx, "steps": steps, "disp_step_mm": solver.disp_step,
            "E": solver.E, "nu": solver.nu, "sigma_t": solver.sigma_t,
            "K_Ic": solver.K_Ic, "beta_crack": solver.beta_crack,
            "a_crack": solver.a_crack,
        },
        "metrics": {
            "peak_load_kN": max(f for _, f in loads) / 1e3,
            "final_load_kN": loads[-1][1] / 1e3,
            "final_maxD": float(np.max(solver.D)),
            "splitting_strength_MPa": splitting_strength(max(f for _, f in loads)),
            "calibrated_strength_MPa": calibrated_strength(max(f for _, f in loads)),
        },
        "files": {
            "load_disp_csv": "load_disp.csv",
            "final_fields_png": "final_fields.png",
            "meta": "run_meta.json",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--Nx", type=int, default=40)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "brazilian_repro")
    args = ap.parse_args()

    out_dir = args.out / str(args.seed)
    meta = run_case(args.seed, args.Nx, args.steps, out_dir)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print("outputs ->", out_dir)


if __name__ == "__main__":
    main()
