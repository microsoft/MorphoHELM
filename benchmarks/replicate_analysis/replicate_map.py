#!/usr/bin/env python3
"""Compatibility entrypoint for the standard replicate mAP benchmark."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--benchmarks" not in sys.argv:
    sys.argv.extend(["--benchmarks", "map"])

runpy.run_path(str(Path(__file__).with_name("run_replicate_analysis_sweep.py")), run_name="__main__")
