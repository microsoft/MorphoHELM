#!/usr/bin/env python3
"""Run configured enrichment and replicate-analysis benchmark sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from config import DEFAULT_CONFIG, default_profile, load_yaml, profile_features_dir, profile_results_dir  # noqa: E402


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MorphoHELM benchmark sweeps")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--features-base", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    profile = args.profile or default_profile(config)
    features_base = args.features_base or str(profile_features_dir(config, profile))
    results_dir = Path(args.results_dir) if args.results_dir else profile_results_dir(config, profile)

    if config.get("enrichment", {}).get("enabled", True):
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "benchmarks" / "enrichment" / "run_enrichment_sweep.py"),
                "--config",
                str(Path(args.config)),
                "--profile",
                profile,
                "--features-base",
                features_base,
                "--results-dir",
                str(results_dir / "enrichment"),
            ],
            dry_run=args.dry_run,
        )

    if config.get("replicate_analysis", {}).get("enabled", True):
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "benchmarks" / "replicate_analysis" / "run_replicate_analysis_sweep.py"),
                "--config",
                str(Path(args.config)),
                "--profile",
                profile,
                "--features-base",
                features_base,
                "--results-dir",
                str(results_dir / "replicate_analysis"),
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
