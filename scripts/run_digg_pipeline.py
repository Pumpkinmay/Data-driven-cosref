#!/usr/bin/env python3
"""Run the reproducible Digg observational-prediction pipeline in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "audit_digg.py",
    "prepare_digg.py",
    "build_digg_exposure_pilot.py",
    "fit_digg_pilot_models.py",
    "diagnose_digg_pilot.py",
    "fit_digg_controlled_models.py",
    "analyze_digg_two_community.py",
    "fit_digg_xgboost.py",
    "validate_digg_xgboost.py",
    "validate_digg_xgboost_group_cv.py",
    "finalize_digg_xgboost.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    args = parser.parse_args()
    for script in STEPS:
        command = [sys.executable, str(ROOT / "scripts" / script)]
        print("+", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
