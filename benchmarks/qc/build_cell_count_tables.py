#!/usr/bin/env python3
"""Build well-level cell-count parquet tables from bounding-box pickles."""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_count_qc import build_cell_count_table, write_cell_count_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark cell-count QC tables")
    parser.add_argument(
        "--bbox-root",
        default="data/bounding_boxes",
        help="Directory containing bounding_boxes_cpg and bounding_boxes_bbbc036.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/qc/cell_counts",
        help="Output directory for cell-count parquet tables.",
    )
    parser.add_argument(
        "--source",
        choices=["all", "cpg", "bbbc036"],
        default="all",
        help="Which count table to build.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional smoke-test limit on number of pickle files per source.",
    )
    args = parser.parse_args()

    bbox_root = Path(args.bbox_root)
    sources = ["cpg", "bbbc036"] if args.source == "all" else [args.source]
    source_dirs = {
        "cpg": bbox_root / "bounding_boxes_cpg",
        "bbbc036": bbox_root / "bounding_boxes_bbbc036",
    }

    for source in sources:
        print(f"Building {source} cell-count table from {source_dirs[source]}...")
        table = build_cell_count_table(source_dirs[source], source, max_files=args.max_files)
        out = write_cell_count_table(table, args.output_dir, source)
        print(
            f"Saved {out}: {len(table)} wells, "
            f"median cell_count={table['cell_count'].median():.1f}"
        )


if __name__ == "__main__":
    main()

