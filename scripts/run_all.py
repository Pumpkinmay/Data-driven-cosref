#!/usr/bin/env python3
"""Run all four public experiments with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from cosref_core import DEFAULT_DATA_DIR, DEFAULT_OUTPUTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--demo", action="store_true", help="Use the built-in small graph")
    parser.add_argument("--quick", action="store_true", help="Use minimal test settings")
    return parser.parse_args()


def run(script: str, common: list[str], extra: list[str]) -> None:
    command = [sys.executable, str(Path(__file__).resolve().parent / script), *common, *extra]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    common = ["--data-dir", str(args.data_dir), "--outputs-dir", str(args.outputs_dir)]
    if args.demo:
        common.append("--demo")
    if args.quick:
        parameter = ["--n-cascades", "8", "--max-steps", "8"]
        grid = [
            "--a-values", "0.8", "--b-values", "0.4", "--theta-values", "0.1",
            "--n-cascades", "8", "--max-steps", "8",
        ]
        diagnostics = ["--n-cascades", "12", "--max-steps", "8"]
        intervention = [
            "--b-ratios", "0.5", "1.0", "1.5", "--n-repeats", "2",
            "--n-cascades", "8", "--max-steps", "8",
        ]
    else:
        parameter = grid = diagnostics = intervention = []
    run("parameter_recovery.py", common, parameter)
    run("parameter_grid_recovery.py", common, grid)
    run("identifiability_diagnostics.py", common, diagnostics)
    run("intercommunity_intervention.py", common, intervention)
    print(f"All experiments completed. Outputs: {args.outputs_dir.resolve()}")


if __name__ == "__main__":
    main()
