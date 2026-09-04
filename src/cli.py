"""Console entry points (installed via [project.scripts]).

The flagships keep their natural module form (`python -m src.solvers.crack`),
so the console wrappers simply re-invoke the module in a subprocess to reuse
the existing `if __name__ == "__main__"` drivers without duplicating logic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_module(module: str) -> int:
    return subprocess.call([sys.executable, "-m", module])


def crack_v2() -> None:
    """Run the v2.0 Brazilian-disc scale-invariant solver."""
    raise SystemExit(_run_module("src.solvers.crack"))


def crack_v1() -> None:
    """Run the v1.0 Brazilian-disc solver."""
    raise SystemExit(_run_module("src.solvers.brazilian_disc_v1"))


def pcg_demo() -> None:
    """Run the PCG vs spsolve consistency demo."""
    raise SystemExit(_run_module("src.pcg_demo"))


def fea_run() -> None:
    """Run the DAT-driven FEA solver from a repository checkout.

    FEA/ is intentionally not part of the installed wheel; this entry point
    shells out to the checked-out ``FEA/run_fea.py`` and forwards CLI arguments.
    """
    script = Path(__file__).resolve().parents[1] / "FEA" / "run_fea.py"
    if not script.exists():
        print("crack-fea requires a repository checkout (FEA/ is not packaged).")
        raise SystemExit(2)
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))
