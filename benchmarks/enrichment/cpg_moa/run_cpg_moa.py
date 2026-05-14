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
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_cpg_moa_enrichment as moa_script  # noqa: WPS433

    moa_script.DATA_PATH = args.profiles
    moa_script.OUTPUT_PKL = args.output
    moa_script.OUTPUT_PNG = args.output.replace(".pkl", ".png")
    if args.bbbc_results:
        moa_script.BBBC_RESULTS_PATH = args.bbbc_results
    moa_script.main()


if __name__ == "__main__":
    main()
