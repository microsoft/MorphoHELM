#!/usr/bin/env python3
"""Config-driven enrichment benchmark sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from config import (  # noqa: E402
    DEFAULT_CONFIG,
    default_profile,
    enabled_names,
    load_yaml,
    parse_csv,
    profile_features_dir,
    profile_results_dir,
    resolve_path,
)
from qc.cell_count_qc import add_cell_count_qc_args, cell_count_qc_cli_args  # noqa: E402


VALID_BENCHMARKS = {"crispr", "cpg_moa", "bbbc036_moa"}


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enrichment benchmarks from config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Benchmark sweep YAML config")
    parser.add_argument("--profile", default=None, help="Normalization profile to benchmark")
    parser.add_argument("--benchmarks", default=None, help="Comma-separated subset: crispr,cpg_moa,bbbc036_moa")
    parser.add_argument("--features-base", default=None, help="Override normalized profile directory")
    parser.add_argument("--results-dir", default=None, help="Override output result directory")
    parser.add_argument("--n-resamples", type=int, default=None, help="Override Monte Carlo resamples")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    add_cell_count_qc_args(parser)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    profile = args.profile or default_profile(config)
    features_dir = Path(args.features_base) if args.features_base else profile_features_dir(config, profile)
    results_dir = Path(args.results_dir) if args.results_dir else profile_results_dir(config, profile) / "enrichment"
    metadata = config["paths"]["metadata"]

    configured = config.get("enrichment", {}).get("run", {})
    selected = set(parse_csv(args.benchmarks) or enabled_names(configured))
    if not selected:
        raise ValueError("No enrichment benchmarks selected")
    unknown = selected - VALID_BENCHMARKS
    if unknown:
        raise ValueError(f"Unknown enrichment benchmarks: {sorted(unknown)}")

    qc_defaults = config.get("qc", {})
    if qc_defaults.get("enabled", False):
        args.cell_count_qc = True
        args.qc_cell_counts_dir = str(resolve_path(config["paths"]["qc_cell_counts_dir"]))
        args.qc_cell_count_cutoff = qc_defaults.get("cell_count_cutoff", args.qc_cell_count_cutoff)
        args.qc_sample_wells = qc_defaults.get("sample_wells", args.qc_sample_wells)
        args.qc_seed = qc_defaults.get("seed", args.qc_seed)

    n_resamples = str(args.n_resamples or config.get("enrichment", {}).get("n_resamples", 100))
    qc = cell_count_qc_cli_args(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Enrichment benchmarks: {', '.join(sorted(selected))}")
    print(f"Profile: {profile}")
    print(f"Features: {features_dir}")
    print(f"Results: {results_dir}")

    if "bbbc036_moa" in selected:
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "benchmarks" / "enrichment" / "bbbc036_moa" / "run_bbbc036_moa.py"),
                "--features-dir",
                str(features_dir / "bbbc036"),
                "--metadata",
                str(resolve_path(metadata["bbbc036_metadata"])),
                "--output",
                str(results_dir / "bbbc036_moa_enrichment.pkl"),
                "--n-resamples",
                n_resamples,
                *qc,
            ],
            dry_run=args.dry_run,
        )

    if "cpg_moa" in selected:
        run_cmd(
            [
                sys.executable,
                str(PROJECT_ROOT / "benchmarks" / "enrichment" / "cpg_moa" / "build_moa_profiles.py"),
                "--normalized-dir",
                str(features_dir / "cpg-moa"),
                "--moa-labels",
                str(resolve_path(metadata["moa_labels"])),
                "--plate-metadata",
                str(resolve_path(metadata["cpg_plate_metadata"])),
                "--output",
                str(results_dir / "moa_cross_source_profiles.pkl"),
                *qc,
            ],
            dry_run=args.dry_run,
        )
        cpg_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "enrichment" / "cpg_moa" / "run_cpg_moa.py"),
            "--profiles",
            str(results_dir / "moa_cross_source_profiles.pkl"),
            "--output",
            str(results_dir / "moa_enrichment_results.pkl"),
            "--n-resamples",
            n_resamples,
        ]
        if "bbbc036_moa" in selected:
            cpg_cmd.extend(["--bbbc-results", str(results_dir / "bbbc036_moa_enrichment.pkl")])
        run_cmd(cpg_cmd, dry_run=args.dry_run)

    if "crispr" in selected:
        for mode in ("no_restriction", "not_same_batch"):
            run_cmd(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "benchmarks" / "enrichment" / "crispr" / "run_crispr_enrichment.py"),
                    "--features-dir",
                    str(features_dir / "cpg-crispr"),
                    "--mode",
                    mode,
                    "--n_resamples",
                    n_resamples,
                    "--stringdb",
                    str(resolve_path(metadata["stringdb"])),
                    "--corum",
                    str(resolve_path(metadata["corum"])),
                    "--humap",
                    str(resolve_path(metadata["humap"])),
                    "--reactome",
                    str(resolve_path(metadata["reactome"])),
                    "--signor",
                    str(resolve_path(metadata["signor"])),
                    "--output",
                    str(results_dir / f"crispr_enrichment_{mode}.pkl"),
                    *qc,
                ],
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
