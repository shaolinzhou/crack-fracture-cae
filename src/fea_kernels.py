"""B3 — generic element kernels shared by the DAT-driven FEA solver.

Centralises the unstructured-mesh operations that the FEA path implements on
top of the structured-grid kernels in :mod:`src.fem_utils` / :mod:`src.networks`:
ball-based nonlocal averaging, kNN damage gradients, the 5-D damage-feature
matrix, and the explicit neighbour graph used by the loss smoothing term.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.fem_utils import mazars_equivalent_strain, stress_invariants


def ball_neighbor_mean(centers: np.ndarray, values: np.ndarray, radius: float) -> np.ndarray:
    """Per-element mean of ``values`` over all neighbours within ``radius``."""
    tree = cKDTree(centers)
    out = np.empty_like(values, dtype=float)
    for i, ids in enumerate(tree.query_ball_point(centers, radius)):
        out[i] = float(np.mean(values[ids])) if ids else float(values[i])
    return out


def knn_max_gradient(centers: np.ndarray, values: np.ndarray, k: int = 7) -> np.ndarray:
    """Max-normalised forward difference proxy ``max_j |D_j-D_i|/dist`` per element."""
    n = len(centers)
    tree = cKDTree(centers)
    nbrs = np.atleast_2d(tree.query(centers, k=min(k, n))[1])
    out = np.zeros(n, dtype=float)
    for i, ids in enumerate(nbrs):
        dist = np.linalg.norm(centers[ids] - centers[i], axis=1)
        diff = np.abs(values[ids] - values[i])
        mask = dist > 1e-12
        out[i] = float(np.max(diff[mask] / dist[mask])) if np.any(mask) else 0.0
    return out


def element_damage_features(
    D: np.ndarray,
    strains: np.ndarray,
    stresses: np.ndarray,
    nu: float,
    eps0: float,
    l_c: float,
    gD: np.ndarray,
) -> np.ndarray:
    """5-D damage feature matrix for arbitrary elements (tanh-normalised)."""
    eta, theta_bar, _ = stress_invariants(stresses[:, 0], stresses[:, 1], stresses[:, 2], nu)
    exy = strains[:, 2] * 0.5
    eps_eq = mazars_equivalent_strain(strains[:, 0], strains[:, 1], exy)
    return np.stack(
        [
            D,
            np.tanh(eta),
            np.tanh(theta_bar),
            np.tanh(eps_eq / eps0 - 1.0),
            np.tanh(l_c * gD),
        ],
        axis=1,
    )


def smooth_edges_from_tree(centers: np.ndarray, k: int = 5) -> list[tuple[int, int]]:
    """(i0, j) neighbour pairs used by the loss smoothing term on a mesh."""
    n = len(centers)
    tree = cKDTree(centers)
    nbrs = np.atleast_2d(tree.query(centers, k=min(k, n))[1])
    edges: list[tuple[int, int]] = []
    for i, ids in enumerate(nbrs):
        i0 = int(ids[0])
        for j in ids[1:]:
            edges.append((i0, int(j)))
    return edges
