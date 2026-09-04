"""C1 — ISRM Brazilian-splitting calibration of the *production* staggered engine.

Reference: ISRM (1978) sigma_t = 2 P / (pi D t); intact D=50 mm disc, sigma_t
= 6 MPa, unit thickness -> P_ref = sigma_t*pi*D/2 ~= 0.471 kN.

Measures the engine's predicted peak splitting load and its ratio to P_ref
across meshes (48/72/96) for the intact disc, plus the 45-degree pre-crack
reference at Nx=48, and stores peak-field cloud images.

Outputs: benchmarks/figures/isrm_calibration_*.png + metrics json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.solvers.crack import CrackSolver

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SIGMA_T = 6.0
D = 50.0
P_REF_KN = SIGMA_T * np.pi * D / 2.0 / 1000.0
STEPS = 50
DISP_STEP = 3e-3


def run(Nx: int, beta: float, a: float) -> dict:
    torch.manual_seed(2026)
    np.random.seed(2026)
    s = CrackSolver(
        Nx=Nx, Ny=Nx, L_domain=60.0, R=25.0, flat_height=0.4,
        E=30000.0, nu=0.25, sigma_t=SIGMA_T, K_Ic=31.62,
        loading_half_width=4.4, disp_step=DISP_STEP,
        beta_crack=beta, a_crack=a,
        n_warmup=STEPS, n_coupled=0, hidden_dim=8, lr=2e-3,
    )
    loads = []
    Fmax = -1.0
    peak = None
    for t in range(STEPS):
        disp = (t + 1) * s.disp_step
        (F, _e, _l), _w = s.step(disp, t)
        loads.append((float(disp), float(F)))
        if F > Fmax:
            Fmax = float(F)
            peak = {"disp": disp, "D": s.D.copy(), "stresses": s.stresses.copy()}
    return {
        "Nx": Nx, "beta": beta, "a_mm": a, "steps": STEPS,
        "peak_load_kN": Fmax / 1e3,
        "peak_disp_um": peak["disp"] * 1e3,
        "ratio": Fmax / 1e3 / P_REF_KN,
        "back_strength_MPa": (Fmax / 1e3) * 1000.0 * 2.0 / (np.pi * D),
        "load_disp": [(d, f / 1e3) for d, f in loads],
        "final_maxD": float(np.max(s.D)),
        "solver": s, "peak": peak,
    }


def _mask(s, values):
    ny, nx = s.Nx - 1, s.Nx - 1
    g = np.full((ny, nx), np.nan)
    for k, e in enumerate(s.active):
        j, i = s.elem_ji[e]
        g[j, i] = values[k]
    return g


def _vm(s, stresses):
    sxx, syy, sxy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    szz = s.nu * (sxx + syy)
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy ** 2)


def plot_ld(cases):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for c in cases:
        label = f"{'precrack45' if c['a_mm'] > 0 else 'intact'} Nx{c['Nx']}"
        ax.plot([d * 1e3 for d, f in c["load_disp"]], [f for d, f in c["load_disp"]],
                lw=1.5, label=f"{label}  (peak {c['peak_load_kN']:.3f} kN)")
    ax.axhline(P_REF_KN, ls=":", color="k", label=f"ISRM P_ref={P_REF_KN:.3f} kN")
    ax.set_xlabel("displacement (um)")
    ax.set_ylabel("load (kN)")
    ax.set_title("C1 — ISRM calibration of the production staggered engine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "isrm_calibration_load_disp.png", dpi=130)
    plt.close(fig)


def plot_ratio(cases):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [f"{'precrack45' if c['a_mm'] > 0 else 'intact'} Nx{c['Nx']}" for c in cases]
    vals = [c["ratio"] for c in cases]
    ax.bar(range(len(cases)), vals, color="tab:blue")
    ax.axhline(1.0, ls="--", color="k", label="target ratio 1.0")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("P_peak / P_ref")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_title("C1 — splitting-strength ratio (accuracy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "isrm_calibration_ratio.png", dpi=130)
    plt.close(fig)


def plot_peak_cloud(c):
    s = c["solver"]
    d_im = _mask(s, c["peak"]["D"])
    vm = _mask(s, _vm(s, c["peak"]["stresses"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(d_im, origin="lower", cmap="inferno", vmin=0, vmax=1,
                        extent=[0, 60, 0, 60])
    axes[0].set_title(f"Damage @ peak ({c['peak_disp_um']:.1f} um)")
    axes[0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0])
    vmax = float(np.nanmax(vm))
    im2 = axes[1].imshow(vm, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                         extent=[0, 60, 0, 60])
    axes[1].set_title("Von Mises @ peak (MPa)")
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])
    key = f"intact_Nx{c['Nx']}" if c["a_mm"] == 0 else f"precrack45_Nx{c['Nx']}"
    fig.suptitle(f"C1 peak cloud {key}  P/P_ref={c['ratio']:.3f}")
    fig.tight_layout()
    fig.savefig(FIG / f"isrm_calibration_cloud_{key}.png", dpi=130)
    plt.close(fig)


def main():
    cases = []
    for nx in (48, 72, 96):
        c = run(nx, beta=0.0, a=0.0)
        cases.append(c)
        print(f"intact Nx{nx}: peak={c['peak_load_kN']:.4f} kN @ {c['peak_disp_um']:.1f} um "
              f"ratio={c['ratio']:.3f}  sigma_b={c['back_strength_MPa']:.2f} MPa", flush=True)
        plot_peak_cloud(c)
    c = run(48, beta=45.0, a=5.0)
    cases.append(c)
    print(f"precrack45 Nx48: peak={c['peak_load_kN']:.4f} kN ratio={c['ratio']:.3f} "
          f"sigma_b={c['back_strength_MPa']:.2f} MPa", flush=True)

    plot_ld(cases)
    plot_ratio(cases)

    summary = {
        "reference_kN": float(P_REF_KN),
        "cases": [{k: v for k, v in cc.items()
                   if k not in ("solver", "peak", "load_disp")} for cc in cases],
    }
    (FIG / "isrm_calibration_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
