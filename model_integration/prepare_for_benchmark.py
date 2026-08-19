#!/usr/bin/env python3
"""Make a freshly-normalized model parquet benchmark-compatible and reproducible.

Two adjustments, matching how the paper baselines are shaped:
  1. Align the well set to a reference baseline (drops wells the baselines/QC
     don't have, e.g. a handful of edge wells) so the cell-count QC mask matches.
  2. For bbbc036, drop annotation columns (Metadata_broad_sample/moa/target) that
     the MoA enrichment script merges itself (avoids a column-name collision).

Usage:
  python model_integration/prepare_for_benchmark.py \
    --split bbbc036 \
    --our    <norm_root>/<profile>/bbbc036/<model>_normalized.parquet \
    --baseline <baseline_root>/<profile>/bbbc036/resnet_normalized.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

CONFLICT_COLS = ["Metadata_broad_sample", "Metadata_moa", "Metadata_target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--our", required=True, help="Our <model>_normalized.parquet (edited in place).")
    ap.add_argument("--baseline", required=True, help="A baseline *_normalized.parquet for the same split.")
    args = ap.parse_args()

    d = pd.read_parquet(args.our)
    d["Metadata_Plate"] = d["Metadata_Plate"].astype(str)
    n0 = len(d)

    b = pd.read_parquet(args.baseline, columns=["Metadata_Plate", "Metadata_Well"])
    b["Metadata_Plate"] = b["Metadata_Plate"].astype(str)
    keys = set(map(tuple, b[["Metadata_Plate", "Metadata_Well"]].values))
    d = d[[(p, w) in keys for p, w in zip(d["Metadata_Plate"], d["Metadata_Well"])]]

    if args.split == "bbbc036":
        d = d.drop(columns=[c for c in CONFLICT_COLS if c in d.columns])

    d = d.sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)
    d.to_parquet(args.our, index=False)
    print(f"{args.split}: wells {n0} -> {len(d)}; cols={[c for c in d.columns if not c.startswith('PC_')]}")


if __name__ == "__main__":
    main()
