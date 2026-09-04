from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from src.fem_utils import mazars_equivalent_strain


def mazars_damage_target(
    eps_eq: np.ndarray,
    eps0: float,
    beta_soft: float,
    eps_eq_cap: float,
    exp_clip: float = 50.0,
) -> np.ndarray:
    """Mazars damage target value D_target ∈ [0, 1)."""
    eps_clip = np.clip(eps_eq, 0.0, eps_eq_cap)
    arg = np.clip(beta_soft * (eps_clip - eps0), 0.0, exp_clip)
    return np.where(
        eps_clip > eps0,
        1.0 - (eps0 / (eps_clip + 1e-30)) * np.exp(-arg),
        0.0,
    )


def compute_damage_base(
    strains: np.ndarray,
    D: np.ndarray,
    eps0: float,
    beta_soft: float,
    eps_eq_cap: float,
    exp_clip: float,
    damping_warmup: float,
    damping_base: float,
    damping_fast: float,
    phase: str,
    eps_eq_grid: np.ndarray | None = None,
    nonlocal_radius: int = 0,
    active_elem_indices: list | None = None,
    elem_ji: dict | None = None,
    Ny: int = 0,
    Nx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mazars damage increment with adaptive damping.

    Returns (delta_D, eps_eq).
    """
    exx, eyy, exy = strains[:, 0], strains[:, 1], strains[:, 2] * 0.5
    eps_eq = mazars_equivalent_strain(exx, eyy, exy)

    if nonlocal_radius > 0 and eps_eq_grid is not None:
        eps_grid = np.zeros((Ny, Nx))
        for idx, e in enumerate(active_elem_indices):
            j, i = elem_ji[e]
            eps_grid[j, i] = eps_eq[idx]
        eps_nl = uniform_filter(eps_grid, size=2 * nonlocal_radius + 1, mode="constant", cval=0.0)
        for idx, e in enumerate(active_elem_indices):
            j, i = elem_ji[e]
            eps_eq[idx] = eps_nl[j, i]

    D_target = mazars_damage_target(eps_eq, eps0, beta_soft, eps_eq_cap, exp_clip)
    driving = np.maximum(D_target - D, 0.0)

    if phase == "warmup":
        damping = np.full_like(driving, damping_warmup)
    else:
        damping = np.where(driving > 0.1, damping_fast, damping_base)

    return driving * damping, eps_eq


def compute_damage_parameters(E: float, nu: float, sigma_t: float, K_Ic: float, char_len: float) -> tuple[float, float]:
    """Compute Mazars initiation strain eps0 and softening parameter beta."""
    eps0 = sigma_t / E
    Gf = K_Ic ** 2 * (1.0 - nu ** 2) / E
    beta_soft = sigma_t / max(Gf / char_len - sigma_t ** 2 / (2.0 * E), 1e-12)
    return eps0, beta_soft
