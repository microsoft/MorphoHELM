#!/usr/bin/env python3
"""Compatibility wrapper for replicate-analysis kNN/mAP sweep."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--benchmarks" not in sys.argv:
    sys.argv.extend(["--benchmarks", "knn_replicate,map"])

runpy.run_path(
    str(Path(__file__).resolve().parent / "replicate_analysis" / "run_replicate_analysis_sweep.py"),
    run_name="__main__",
)
