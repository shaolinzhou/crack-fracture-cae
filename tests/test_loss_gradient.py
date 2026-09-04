from __future__ import annotations

import numpy as np
import torch

from src.networks import compute_loss


def _setup(n: int = 16, grid: int = 4):
    torch.manual_seed(0)
    np.random.seed(0)
    D = np.random.uniform(0.0, 0.6, n).astype(np.float32)
    phi = np.abs(np.random.randn(n)).astype(np.float32)
    phi_test = np.abs(np.random.randn(n)).astype(np.float32)
    active = list(range(n))
    d0 = torch.tensor(np.random.uniform(-2.0, -0.5, n), dtype=torch.float32)
    return D, phi, phi_test, active, d0, grid


def test_compute_loss_total_grad_finite_difference():
    D, phi, phi_test, active, d0, grid = _setup()
    Ny = Nx = grid

    def scalar(x: torch.Tensor) -> float:
        loss_t, *_ = compute_loss(
            x, D, phi, phi_test, 3.0,
            0.3, 0.5, 0.3, 0.2, 0.1,
            1.0, 1.0, 1.0, Ny, Nx, active,
        )
        return float(loss_t.detach())

    x = d0.clone().requires_grad_(True)
    scalar(x)
    loss_t, *_ = compute_loss(
        x, D, phi, phi_test, 3.0,
        0.3, 0.5, 0.3, 0.2, 0.1,
        1.0, 1.0, 1.0, Ny, Nx, active,
    )
    loss_t.backward()
    grad = x.grad.numpy()

    h = 1e-3
    fd = np.empty_like(grad)
    for i in range(len(d0)):
        e = torch.zeros_like(d0)
        e[i] = h
        fp = scalar((d0 + e).detach())
        fm = scalar((d0 - e).detach())
        fd[i] = (fp - fm) / (2.0 * h)

    err = np.max(np.abs(grad - fd)) / (np.max(np.abs(grad)) + 1e-12)
    assert err < 5e-3, f"autograd/finite-difference mismatch: rel err={err:.3e}"
