from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    """Unified configuration for the crack solver."""

    # ── mesh ──
    n_warmup: int = 500
    n_coupled: int = 200

    # ── NN ──
    hidden_dim: int = 32
    lr: float = 2e-3

    # ── numerical stability ──
    residual_stiffness: float = 1e-6
    damping_warmup: float = 0.3
    damping_base: float = 0.5
    damping_fast: float = 0.7
    exp_clip: float = 50.0
    eps_eq_cap_factor: float = 200.0

    # ── scale correction ──
    scale_ratio: float = 0.3

    # ── loss weights ──
    lam_g: float = 0.3
    lam_e: float = 0.5
    lam_f: float = 0.3
    lam_d: float = 0.2
    lam_s: float = 0.1

    # ── length scales ──
    l_c: float = 0.5
    l_d: float = 1.0

    # ── I/O ──
    output_stride: int = 10
    auto_anchor_x: bool = True
    gid_name: str | None = None
    output_dir: str | Path = "results"

    @property
    def total_steps(self) -> int:
        return self.n_warmup + self.n_coupled
