from __future__ import annotations

import numpy as np


def plane_strain_C(E: float, nu: float) -> np.ndarray:
    """Plane-strain elastic stiffness matrix (Voigt: [σ_xx, σ_yy, σ_xy])."""
    f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return np.array([
        [f * (1.0 - nu), f * nu, 0.0],
        [f * nu, f * (1.0 - nu), 0.0],
        [0.0, 0.0, f * (1.0 - 2.0 * nu) / 2.0],
    ], dtype=float)


def q4_B_matrix(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """B matrix and Jacobian determinant for a general Q4 element."""
    dN_dxi = np.array(
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        dtype=float,
    ) * 0.25
    dN_deta = np.array(
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        dtype=float,
    ) * 0.25

    jac = np.array([
        [np.dot(dN_dxi, coords[:, 0]), np.dot(dN_dxi, coords[:, 1])],
        [np.dot(dN_deta, coords[:, 0]), np.dot(dN_deta, coords[:, 1])],
    ], dtype=float)
    det_j = float(np.linalg.det(jac))
    if det_j <= 0:
        raise ValueError(f"Invalid Q4 element with non-positive detJ={det_j}")

    inv_j = np.linalg.inv(jac)
    grads = inv_j @ np.vstack([dN_dxi, dN_deta])
    dN_dx, dN_dy = grads[0], grads[1]

    B = np.zeros((3, 8), dtype=float)
    for i in range(4):
        B[0, 2 * i] = dN_dx[i]
        B[1, 2 * i + 1] = dN_dy[i]
        B[2, 2 * i] = dN_dy[i]
        B[2, 2 * i + 1] = dN_dx[i]
    return B, det_j


def q4_unit_stiffness(coords: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit-modulus stiffness matrix (E=1), center B matrix, and area for a Q4 element."""
    C_unit = plane_strain_C(1.0, nu)
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    k = np.zeros((8, 8), dtype=float)
    area = 0.0
    for xi in gp:
        for eta in gp:
            B, det_j = q4_B_matrix(coords, xi, eta)
            k += B.T @ C_unit @ B * det_j
            area += det_j
    B_center, _ = q4_B_matrix(coords, 0.0, 0.0)
    return k, B_center, area


def rect_q4_stiffness_template(dx: float, dy: float, C: np.ndarray) -> np.ndarray:
    """Stiffness matrix for a *rectangular* Q4 element (2×2 Gauss)."""
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    a, b = dx / 2.0, dy / 2.0
    k0 = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) * 0.25
            dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) * 0.25
            dN_dx = dN_dxi / a
            dN_dy = dN_deta / b
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i] = dN_dx[i]
                B[1, 2 * i + 1] = dN_dy[i]
                B[2, 2 * i] = dN_dy[i]
                B[2, 2 * i + 1] = dN_dx[i]
            k0 += B.T @ C @ B * a * b
    return k0


def rect_b_matrix_center(dx: float, dy: float) -> np.ndarray:
    """B matrix at the center of a rectangular Q4 element."""
    dNdx = np.array([-1.0, 1.0, 1.0, -1.0]) / (2.0 * dx)
    dNdy = np.array([-1.0, -1.0, 1.0, 1.0]) / (2.0 * dy)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dNdx[i]
        B[1, 2 * i + 1] = dNdy[i]
        B[2, 2 * i] = dNdy[i]
        B[2, 2 * i + 1] = dNdx[i]
    return B


def compute_principal_strains(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal strains ε₁, ε₂ from Voigt strain components."""
    e_avg = 0.5 * (exx + eyy)
    e_diff = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
    return e_avg + e_diff, e_avg - e_diff


def mazars_equivalent_strain(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray) -> np.ndarray:
    """Mazars equivalent tensile strain (Macaulay bracket on principal strains)."""
    e1, e2 = compute_principal_strains(exx, eyy, exy)
    return np.sqrt(np.maximum(e1, 0.0) ** 2 + np.maximum(e2, 0.0) ** 2)


def von_mises_stress(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray, nu: float) -> np.ndarray:
    """Von Mises equivalent stress (plane-strain out-of-plane component)."""
    szz = nu * (sxx + syy)
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * sxy ** 2
    )


def stress_invariants(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stress triaxiality η, Lode angle parameter θ̄, equivalent stress σ_eq."""
    szz = nu * (sxx + syy)
    sm = (sxx + syy + szz) / 3.0
    Sxx, Syy, Szz = sxx - sm, syy - sm, szz - sm
    J2 = 0.5 * (Sxx ** 2 + Syy ** 2 + Szz ** 2) + sxy ** 2
    seq = np.sqrt(3.0 * J2 + 1e-30)
    eta = sm / (seq + 1e-12)
    J3 = Sxx * Syy * Szz - Szz * sxy ** 2
    cos_arg = np.clip(27.0 * J3 / (2.0 * seq ** 3 + 1e-30), -1.0, 1.0)
    theta_bar = 1.0 - (2.0 / np.pi) * np.arccos(cos_arg)
    return eta, theta_bar, seq
