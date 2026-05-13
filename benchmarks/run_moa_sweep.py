#!/usr/bin/env python3
"""Wrapper to run MOA enrichment with custom profile path."""
import sys
import os

# Parse our args before importing the enrichment script
profiles_path = sys.argv[sys.argv.index("--profiles") + 1]
output_path = sys.argv[sys.argv.index("--output") + 1]

# Optional: per-pipeline BBBC036 enrichment results
bbbc_results_path = None
if "--bbbc-results" in sys.argv:
    bbbc_results_path = sys.argv[sys.argv.index("--bbbc-results") + 1]

# Clear sys.argv so the enrichment script's argparse doesn't see our args
sys.argv = [sys.argv[0]]

# Monkey-patch before import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "enrichment", "moa"))

import run_cpjump_enrichment_heatmap as moa_script

moa_script.DATA_PATH = profiles_path
moa_script.OUTPUT_PKL = output_path
moa_script.OUTPUT_PNG = output_path.replace(".pkl", ".png")

if bbbc_results_path:
    moa_script.BBBC_RESULTS_PATH = bbbc_results_path

moa_script.main()
