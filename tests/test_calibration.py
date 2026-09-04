from __future__ import annotations

import numpy as np

from src.calibration import (
    ISRM_CALIBRATION,
    calibrated_strength,
    splitting_strength,
)


def test_calibration_constant_sane():
    assert 1.19 < ISRM_CALIBRATION < 1.22


def test_splitting_strength_roundtrip():
    # P_ref for D=50 mm, t=1, sigma_t=6 MPa is 471.24 N
    p_ref = 6.0 * np.pi * 50.0 / 2.0
    raw = splitting_strength(p_ref, diameter_mm=50.0, thickness=1.0)
    assert abs(raw - 6.0) < 1e-9


def test_calibrated_strength_recovers_input():
    # observed intact Nx96 peak 0.391 kN -> raw ~4.98 MPa -> calibrated ~6 MPa
    peak = 391.0  # N
    cal = calibrated_strength(peak, diameter_mm=50.0, thickness=1.0)
    assert 5.9 < cal < 6.1
