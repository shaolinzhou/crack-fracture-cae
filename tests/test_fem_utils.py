from __future__ import annotations

import numpy as np

from src.fem_utils import plane_strain_C, mazars_equivalent_strain, von_mises_stress


def test_plane_strain_C_symmetry():
    C = plane_strain_C(30000.0, 0.25)
    assert C.shape == (3, 3)
    assert np.allclose(C, C.T)


def test_plane_strain_C_isotropic():
    C = plane_strain_C(30000.0, 0.25)
    assert abs(C[0, 0] - C[1, 1]) < 1e-12
    assert abs(C[0, 2]) < 1e-12
    assert abs(C[1, 2]) < 1e-12


def test_plane_strain_C_uniaxial():
    """For uniaxial strain ε_xx, σ_xx = E*(1-ν)/((1+ν)(1-2ν)) * ε_xx."""
    E, nu = 30000.0, 0.25
    C = plane_strain_C(E, nu)
    # uniaxial strain: ε = [1, 0, 0]
    sigma = C @ np.array([1.0, 0.0, 0.0])
    expected = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    assert abs(sigma[0] - expected) < 1e-10 * expected
    assert abs(sigma[1] - expected * nu / (1.0 - nu)) < 1e-10


def test_mazars_equivalent_strain_uniaxial():
    """Uniaxial tension: ε_xx > 0, others 0 → eps_eq = ε_xx."""
    exx = np.array([0.001, 0.002, 0.0])
    eyy = np.zeros(3)
    exy = np.zeros(3)
    eps_eq = mazars_equivalent_strain(exx, eyy, exy)
    assert np.allclose(eps_eq, exx)


def test_mazars_equivalent_strain_compression():
    """All compressive → eps_eq = 0."""
    exx = np.array([-0.001, -0.002])
    eyy = np.array([-0.0005, -0.001])
    exy = np.zeros(2)
    eps_eq = mazars_equivalent_strain(exx, eyy, exy)
    assert np.allclose(eps_eq, 0.0)


def test_von_mises_uniaxial():
    """Uniaxial stress σ_xx → σ_vm = |σ_xx| (plane strain, ν=0)."""
    sxx = np.array([10.0, 20.0, -5.0])
    syy = np.zeros(3)
    sxy = np.zeros(3)
    vm = von_mises_stress(sxx, syy, sxy, nu=0.0)
    assert np.allclose(vm, np.abs(sxx))
