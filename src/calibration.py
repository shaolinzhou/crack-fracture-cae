"""C1 calibration constant and reporting helpers.

From the ISRM Brazilian-splitting calibration of the *production* staggered
engine (benchmarks/run_isrm_calibration.py), the mesh-converged intact-disc
peak ratio is ``P_peak/P_ref ~= 0.83`` (Nx96), i.e. the engine is about 17%
conservative against the ISRM splitting-strength relation

    sigma_t = 2 P_peak / (pi D t).

To *report* an effective splitting strength consistent with the input tensile
strength the raw back-calculation is multiplied by ``ISRM_CALIBRATION``
(~1.20).  This is a reporting/engineering calibration factor, not a physical
material constant; the physical premature-softening origin is tracked in the
A-series roadmap item.
"""

from __future__ import annotations

import numpy as np

# intact-disc peak ratio at Nx96 from the C1 calibration run
ISRM_INTACT_RATIO_NX96 = 0.8298
# reporting calibration factor  k = 1 / 0.8298
ISRM_CALIBRATION = 1.0 / ISRM_INTACT_RATIO_NX96


def splitting_strength(peak_load: float, diameter_mm: float = 50.0, thickness: float = 1.0) -> float:
    """ISRM splitting tensile strength from the peak load (force units)."""
    return float(2.0 * peak_load / (np.pi * diameter_mm * thickness))


def calibrated_strength(
    peak_load: float, diameter_mm: float = 50.0, thickness: float = 1.0
) -> float:
    """Engine-reported strength = ISRM back-calculation x calibration factor."""
    return float(splitting_strength(peak_load, diameter_mm, thickness) * ISRM_CALIBRATION)
