from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PhysicsScaleNetSolid(nn.Module):
    """Predicts local scale exponent d(x) from 5 mechanical invariants.

    Input features: [D, tanh(η), tanh(θ̄), tanh(ε_eq/ε₀ - 1), tanh(l_c·|∇D|)]
    Output: d(x) ∈ (-∞, -0.5]  — elastic anchor at d=-0.5
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 - nn.functional.softplus(self.net(x))


def compute_features(
    D: np.ndarray,
    strains: np.ndarray,
    stresses: np.ndarray,
    nu: float,
    eps0: float,
    l_c: float,
    Ny: int,
    Nx: int,
    active_indices: list[int],
    elem_ji: dict[int, tuple[int, int]],
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Assemble the 5D feature tensor for all active elements."""
    exx, eyy, exy = strains[:, 0], strains[:, 1], strains[:, 2] * 0.5
    e_avg = 0.5 * (exx + eyy)
    e_diff = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
    eps_eq = np.sqrt(np.maximum(e_avg + e_diff, 0.0) ** 2 + np.maximum(e_avg - e_diff, 0.0) ** 2)

    sxx, syy, sxy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    szz = nu * (sxx + syy)
    sm = (sxx + syy + szz) / 3.0
    Sxx, Syy, Szz = sxx - sm, syy - sm, szz - sm
    J2 = 0.5 * (Sxx ** 2 + Syy ** 2 + Szz ** 2) + sxy ** 2
    seq = np.sqrt(3.0 * J2 + 1e-30)
    eta = sm / (seq + 1e-12)
    J3 = Sxx * Syy * Szz - Szz * sxy ** 2
    cos_arg = np.clip(27.0 * J3 / (2.0 * seq ** 3 + 1e-30), -1.0, 1.0)
    theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos_arg)

    D_grid = np.zeros((Ny, Nx))
    for idx, e in enumerate(active_indices):
        j, i = elem_ji[e]
        D_grid[j, i] = D[idx]
    gdy, gdx = np.gradient(D_grid, dy, dx)
    grad_mag = np.sqrt(gdx ** 2 + gdy ** 2)
    gD_arr = np.zeros(len(active_indices))
    for idx, e in enumerate(active_indices):
        j, i = elem_ji[e]
        gD_arr[idx] = l_c * grad_mag[j, i]

    F_np = np.stack([
        D,
        np.tanh(eta),
        np.tanh(theta_bar),
        np.tanh(eps_eq / eps0 - 1.0),
        np.tanh(gD_arr),
    ], axis=1)
    return torch.tensor(F_np, dtype=torch.float32)


def compute_germano_signal(
    strains: np.ndarray,
    stresses: np.ndarray,
    D: np.ndarray,
    delta_D_base: np.ndarray,
    Ny: int,
    Nx: int,
    active_indices: list[int],
    elem_ji: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Germano self-similarity signal (dual-filtered dissipation).

    Returns (H_arr, w_arr, phi, phi_test).
    """
    exx, eyy, exy = strains[:, 0], strains[:, 1], strains[:, 2] * 0.5
    W = 0.5 * (stresses[:, 0] * exx + stresses[:, 1] * eyy + 2.0 * stresses[:, 2] * exy)
    Y = W / ((1.0 - D) ** 2 + 1e-30)
    phi = Y * delta_D_base

    phi_grid = np.zeros((Ny, Nx))
    for idx, e in enumerate(active_indices):
        j, i = elem_ji[e]
        phi_grid[j, i] = phi[idx]

    from scipy.ndimage import uniform_filter
    phi_test = uniform_filter(phi_grid, size=3, mode="constant", cval=0.0)

    H_arr = np.zeros(len(active_indices))
    w_arr = np.zeros(len(active_indices))
    for idx, e in enumerate(active_indices):
        j, i = elem_ji[e]
        p_local = phi_grid[j, i]
        if p_local > 1e-15:
            H_arr[idx] = phi_test[j, i] / p_local
            w_arr[idx] = p_local
    return H_arr, w_arr, phi, phi_test


def compute_loss(
    d_pred: torch.Tensor,
    D: np.ndarray,
    phi: np.ndarray,
    phi_test: np.ndarray,
    lambda_L: float,
    lam_g: float,
    lam_e: float,
    lam_f: float,
    lam_d: float,
    lam_s: float,
    l_d: float,
    dx: float,
    dy: float,
    Ny: int,
    Nx: int,
    active_indices: list[int],
    smooth_edges: list[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Five-term hybrid loss function.

    ``smooth_edges`` optionally supplies an explicit (i, j) neighbour graph so
    the smoothing term works on unstructured meshes (FEA path).  When given,
    the grid-based finite-difference smoothing (``Ny/Nx/active_indices``) is
    ignored.  Returns (loss_total, loss_g, loss_e, loss_f, loss_s).
    """
    phi_g_t = torch.tensor(phi, dtype=torch.float32).unsqueeze(1)
    phi_t_t = torch.tensor(phi_test, dtype=torch.float32).unsqueeze(1)
    pred_ratio = lambda_L ** d_pred
    loss_g = torch.sum((pred_ratio * phi_g_t - phi_t_t) ** 2) / (torch.sum(phi_g_t ** 2) + 1e-15)

    D_t = torch.tensor(D, dtype=torch.float32).unsqueeze(1)
    mask_e = (D_t < 0.01).float()
    loss_e = torch.sum(mask_e * (d_pred - (-0.5)) ** 2) / (mask_e.sum() + 1e-15)

    mask_f = (D_t > 0.9).float()
    loss_f = torch.sum(mask_f * torch.exp(2.0 * d_pred)) / (mask_f.sum() + 1e-15)

    D_np = np.clip(D.copy(), 0.0, 0.999)
    f_t = torch.tensor(-0.5 + np.log(1.0 - D_np) * 0.3, dtype=torch.float32).unsqueeze(1)
    loss_d = torch.mean((d_pred - f_t) ** 2)

    if smooth_edges is not None:
        d_flat = d_pred.squeeze()
        terms = [(d_flat[i] - d_flat[j]) ** 2 for i, j in smooth_edges]
        loss_s = l_d ** 2 * torch.mean(torch.stack(terms)) if terms else d_flat.sum() * 0.0
    else:
        d_grid_flat = torch.full((Ny * Nx,), -0.5, dtype=torch.float32)
        d_grid_flat[torch.tensor(active_indices)] = d_pred.squeeze()
        d_grid = d_grid_flat.view(Ny, Nx)
        gdx = (d_grid[:, 1:] - d_grid[:, :-1]) / dx
        gdy = (d_grid[1:, :] - d_grid[:-1, :]) / dy
        loss_s = l_d ** 2 * (torch.mean(gdx ** 2) + torch.mean(gdy ** 2))

    loss_total = lam_g * loss_g + lam_e * loss_e + lam_f * loss_f + lam_d * loss_d + lam_s * loss_s
    return loss_total, loss_g, loss_e, loss_f, loss_s
