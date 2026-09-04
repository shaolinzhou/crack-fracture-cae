from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dat_parser import read_dat
from solver import DatCrackSolver, SolverConfig


def default_dat_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "c1.dat",
        here.parent / "c1.dat",
        here / "c1.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    return here.parent / "data" / "c1.dat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the crack solver from a DAT pre-processing file."
    )
    parser.add_argument(
        "dat_file",
        nargs="?",
        type=Path,
        default=default_dat_path(),
        help="Path to the .dat file. Defaults to ./c1.dat or ../c1.dat.",
    )
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps without NN scaling.")
    parser.add_argument("--coupled", type=int, default=500, help="Coupled NN/Germano steps.")
    parser.add_argument("--stride", type=int, default=10, help="Snapshot/console output stride.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Output directory.",
    )
    parser.add_argument(
        "--gid-name",
        default=None,
        help="Base name for GiD .msh/.res files. Defaults to the DAT file stem.",
    )
    parser.add_argument(
        "--no-auto-anchor-x",
        action="store_true",
        help="Do not add an automatic UX anchor if the DAT file has no UX fixed dof.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dat_path = args.dat_file.resolve()
    if not dat_path.exists():
        print(f"DAT file not found: {dat_path}", file=sys.stderr)
        return 2

    model = read_dat(dat_path)
    config = SolverConfig(
        n_warmup=args.warmup,
        n_coupled=args.coupled,
        output_stride=max(args.stride, 1),
        auto_anchor_x=not args.no_auto_anchor_x,
        gid_name=args.gid_name,
    )
    solver = DatCrackSolver(model, output_dir=args.output, config=config)
    solver.run()
    print(f"\nDone. Results: {args.output.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
