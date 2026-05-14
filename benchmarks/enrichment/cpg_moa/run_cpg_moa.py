#!/usr/bin/env python3
"""Run cpg-MoA enrichment from prebuilt profile pickles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cpg-MoA enrichment")
    parser.add_argument("--profiles", required=True, help="Input cpg-MoA profile pickle")
    parser.add_argument("--output", required=True, help="Output enrichment result pickle")
    parser.add_argument("--bbbc-results", help="Optional BBBC036 MoA result pickle for combined heatmap rendering")
    parser.add_argument("--n-resamples", type=int, help="Override permutation resamples for smoke validation")
    parser.add_argument("--max-batches", type=int, help="Limit not-same-batch groups for smoke validation")
    parser.add_argument("--max-sources", type=int, help="Limit source groups for smoke validation")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_cpg_moa_enrichment as moa_script  # noqa: WPS433

    moa_script.DATA_PATH = args.profiles
    moa_script.OUTPUT_PKL = args.output
    moa_script.OUTPUT_PNG = args.output.replace(".pkl", ".png")
    if args.bbbc_results:
        moa_script.BBBC_RESULTS_PATH = args.bbbc_results
    if args.n_resamples is not None:
        moa_script.N_RESAMPLES = args.n_resamples
    if args.max_batches is not None:
        moa_script.MAX_BATCHES = args.max_batches
    if args.max_sources is not None:
        moa_script.MAX_SOURCES = args.max_sources
    sys.argv = [str(Path(moa_script.__file__).resolve())]
    moa_script.main()


if __name__ == "__main__":
    main()
