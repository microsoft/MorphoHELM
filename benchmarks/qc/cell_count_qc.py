"""Cell-count quality-control utilities for benchmark inputs.

The benchmark data are well-level, so cell-count QC is also applied at the
well level. Counts are derived once from bounding-box pickles and then reused
as small parquet tables during benchmark runs.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_CELL_COUNTS_DIR = Path("data/qc/cell_counts")
QC_PLATE_COL = "__qc_plate"
QC_WELL_COL = "__qc_well"
KEEP_COL = "keep_cell_count_qc"

COUNT_FILENAMES = {
    "cpg": "cpg_cell_counts.parquet",
    "bbbc036": "bbbc036_cell_counts.parquet",
}

CPG_IMAGE_RE = re.compile(r"(?P<well>[A-Za-z]{1,2}\d{2})_i\d+_ch\d+")
BBBC036_IMAGE_RE = re.compile(r"(?:^|_)(?P<well>[A-Za-z]{1,2}\d{1,2})_s\d+_w\d+")


def add_cell_count_qc_args(parser: argparse.ArgumentParser) -> None:
    """Add shared cell-count QC CLI options to a parser."""
    parser.add_argument(
        "--cell-count-qc",
        action="store_true",
        help="Enable bottom-percentile well-level cell-count QC.",
    )
    parser.add_argument(
        "--qc-cell-counts-dir",
        default=str(DEFAULT_CELL_COUNTS_DIR),
        help="Directory containing precomputed CPG/BBBC036 cell-count parquet files.",
    )
    parser.add_argument(
        "--qc-cell-count-cutoff",
        type=float,
        default=0.05,
        help="Lower-tail quantile cutoff for cell-count QC when enabled.",
    )
    parser.add_argument(
        "--qc-sample-wells",
        type=int,
        default=5000,
        help="Number of benchmark wells sampled to estimate the cutoff; <=0 uses all wells.",
    )
    parser.add_argument(
        "--qc-seed",
        type=int,
        default=42,
        help="Random seed for deterministic QC well sampling.",
    )


def cell_count_qc_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "cell_count_qc", False))


def qc_result_subdir(cutoff: float, sample_wells: int, seed: int) -> str:
    cutoff_tag = f"p{int(round(cutoff * 100)):02d}" if cutoff < 1 else f"q{cutoff:g}"
    sample_tag = "all" if sample_wells <= 0 else str(sample_wells)
    return f"qc_cellcount_{cutoff_tag}_n{sample_tag}_seed{seed}"


def cell_count_qc_cli_args(args: argparse.Namespace) -> list[str]:
    if not cell_count_qc_enabled(args):
        return []
    return [
        "--cell-count-qc",
        "--qc-cell-counts-dir", str(args.qc_cell_counts_dir),
        "--qc-cell-count-cutoff", str(args.qc_cell_count_cutoff),
        "--qc-sample-wells", str(args.qc_sample_wells),
        "--qc-seed", str(args.qc_seed),
    ]


def normalize_well_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.fullmatch(r"([A-Za-z]{1,2})0?(\d{1,2})", text)
    if match:
        row, col = match.groups()
        return f"{row.upper()}{int(col):02d}"
    return text.upper()


def add_normalized_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Metadata_Plate", "Metadata_Well"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cell-count QC requires columns {sorted(missing)}")
    out = df.copy()
    out[QC_PLATE_COL] = out["Metadata_Plate"].astype(str)
    out[QC_WELL_COL] = out["Metadata_Well"].map(normalize_well_value)
    return out


def dataset_to_count_source(dataset_name: str) -> str:
    return "bbbc036" if dataset_name.lower().startswith("bbbc036") else "cpg"


def count_table_path(cell_counts_dir: str | Path, source: str) -> Path:
    if source not in COUNT_FILENAMES:
        raise ValueError(f"Unknown cell-count source: {source}")
    return Path(cell_counts_dir) / COUNT_FILENAMES[source]


def load_cell_counts(cell_counts_dir: str | Path, dataset_name: str) -> pd.DataFrame:
    source = dataset_to_count_source(dataset_name)
    path = count_table_path(cell_counts_dir, source)
    if not path.exists() and not Path(cell_counts_dir).is_absolute():
        project_relative = Path(__file__).resolve().parents[2] / path
        if project_relative.exists():
            path = project_relative
    if not path.exists():
        raise FileNotFoundError(
            f"Cell-count QC table not found: {path}. "
            "Build it with benchmarks/qc/build_cell_count_tables.py first."
        )
    counts = pd.read_parquet(path)
    counts = add_normalized_key_columns(counts)
    columns = [QC_PLATE_COL, QC_WELL_COL, "cell_count", "n_fovs", "bbox_source"]
    return counts[columns].drop_duplicates([QC_PLATE_COL, QC_WELL_COL])


def _box_count(boxes: object) -> int:
    if boxes is None:
        return 0
    return int(len(boxes))


def _parse_cpg_pickle(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected CPG bbox pickle schema in {path}")

    records: dict[tuple[str, str], dict[str, object]] = {}
    for plate, images in payload.items():
        if not isinstance(images, dict):
            raise ValueError(f"Unexpected CPG plate payload in {path} for plate {plate}")
        plate_text = str(plate)
        for image_name, boxes in images.items():
            match = CPG_IMAGE_RE.search(Path(str(image_name)).name)
            if not match:
                raise ValueError(f"Could not parse CPG well from {image_name!r} in {path}")
            well = normalize_well_value(match.group("well"))
            key = (plate_text, well)
            if key not in records:
                records[key] = {
                    "Metadata_Plate": plate_text,
                    "Metadata_Well": well,
                    "cell_count": 0,
                    "n_fovs": 0,
                    "bbox_source": "cpg",
                }
            records[key]["cell_count"] += _box_count(boxes)
            records[key]["n_fovs"] += 1
    return list(records.values())


def _parse_bbbc036_pickle(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected BBBC036 bbox pickle schema in {path}")

    plate = path.stem
    records: dict[tuple[str, str], dict[str, object]] = {}
    for image_id, boxes in payload.items():
        match = BBBC036_IMAGE_RE.search(str(image_id))
        if not match:
            raise ValueError(f"Could not parse BBBC036 well from {image_id!r} in {path}")
        well = normalize_well_value(match.group("well"))
        key = (plate, well)
        if key not in records:
            records[key] = {
                "Metadata_Plate": plate,
                "Metadata_Well": well,
                "cell_count": 0,
                "n_fovs": 0,
                "bbox_source": "bbbc036",
            }
        records[key]["cell_count"] += _box_count(boxes)
        records[key]["n_fovs"] += 1
    return list(records.values())


def build_cell_count_table(
    bbox_dir: str | Path,
    source: str,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Build a well-level cell-count table from bounding-box pickles."""
    source = source.lower()
    if source not in COUNT_FILENAMES:
        raise ValueError(f"source must be one of {sorted(COUNT_FILENAMES)}")

    paths = sorted(Path(bbox_dir).glob("*.pkl"))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {bbox_dir}")

    parser = _parse_cpg_pickle if source == "cpg" else _parse_bbbc036_pickle
    rows: list[dict[str, object]] = []
    for idx, path in enumerate(paths, start=1):
        rows.extend(parser(path))
        if idx % 100 == 0:
            print(f"  {source}: parsed {idx}/{len(paths)} files", flush=True)

    table = pd.DataFrame(rows)
    table = (
        table.groupby(["Metadata_Plate", "Metadata_Well", "bbox_source"], as_index=False)
        .agg(cell_count=("cell_count", "sum"), n_fovs=("n_fovs", "sum"))
    )
    table = table[["Metadata_Plate", "Metadata_Well", "cell_count", "n_fovs", "bbox_source"]]
    table = table.sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)
    return table


def write_cell_count_table(table: pd.DataFrame, output_dir: str | Path, source: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = count_table_path(output_dir, source)
    table.to_parquet(path, index=False)
    return path


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    strata_cols: Iterable[str],
) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.copy()

    strata_cols = [c for c in strata_cols if c in df.columns]
    if not strata_cols:
        return df.sample(n=n, random_state=seed).copy()

    grouped = list(df.groupby(strata_cols, dropna=False, sort=True))
    sizes = np.array([len(group) for _, group in grouped], dtype=np.int64)
    raw = sizes / sizes.sum() * n
    take = np.floor(raw).astype(np.int64)
    remainder = raw - take

    remaining = n - int(take.sum())
    if remaining > 0:
        order = np.argsort(-remainder)
        for idx in order:
            if remaining == 0:
                break
            if take[idx] < sizes[idx]:
                take[idx] += 1
                remaining -= 1

    samples = []
    for idx, (_, group) in enumerate(grouped):
        if take[idx] == 0:
            continue
        samples.append(group.sample(n=int(take[idx]), random_state=seed + idx))
    return pd.concat(samples, ignore_index=True)


def _qc_dkw_error(sample_n: int, alpha: float = 0.05) -> float:
    if sample_n <= 0:
        return 0.0
    return float(math.sqrt(math.log(2.0 / alpha) / (2.0 * sample_n)))


def _qc_stem(dataset_name: str, cutoff: float, sample_n: int, seed: int, label: str | None) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label or dataset_name)
    cutoff_tag = f"{cutoff:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"cell_count_qc_{safe_label}_cutoff{cutoff_tag}_sample{sample_n}_seed{seed}"


def make_cell_count_qc_mask(
    feature_df: pd.DataFrame,
    dataset_name: str,
    cell_counts_dir: str | Path,
    cutoff: float = 0.05,
    sample_wells: int = 5000,
    seed: int = 42,
    output_dir: str | Path | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    """Create a deterministic keep/drop mask for a benchmark dataset."""
    if not 0 <= cutoff <= 1:
        raise ValueError(f"QC cutoff must be in [0, 1], got {cutoff}")

    metadata_cols = [
        c for c in ["Metadata_Plate", "Metadata_Well", "Metadata_Source", "Metadata_Batch"]
        if c in feature_df.columns
    ]
    wells = add_normalized_key_columns(feature_df[metadata_cols].drop_duplicates())
    wells = wells.drop_duplicates([QC_PLATE_COL, QC_WELL_COL])
    counts = load_cell_counts(cell_counts_dir, dataset_name)
    merged = wells.merge(counts, on=[QC_PLATE_COL, QC_WELL_COL], how="left")

    missing = merged["cell_count"].isna()
    if missing.any():
        preview = merged.loc[missing, ["Metadata_Plate", "Metadata_Well"]].head(10)
        raise ValueError(
            f"Cell-count QC missing {int(missing.sum())}/{len(merged)} wells for "
            f"{dataset_name}. Preview: {preview.to_dict(orient='records')}"
        )

    strata_cols = [c for c in ["Metadata_Source", "Metadata_Batch", "Metadata_Plate"] if c in merged.columns]
    sampled = _stratified_sample(merged, sample_wells, seed, strata_cols)
    threshold = float(sampled["cell_count"].quantile(cutoff))
    merged[KEEP_COL] = merged["cell_count"] >= threshold

    summary = {
        "dataset": dataset_name,
        "bbox_source": dataset_to_count_source(dataset_name),
        "cutoff_quantile": float(cutoff),
        "cell_count_threshold": threshold,
        "requested_sample_wells": int(sample_wells),
        "sampled_wells": int(len(sampled)),
        "seed": int(seed),
        "dkw_95_rank_error": _qc_dkw_error(len(sampled)),
        "wells_before": int(len(merged)),
        "wells_after": int(merged[KEEP_COL].sum()),
        "wells_dropped": int((~merged[KEEP_COL]).sum()),
        "fraction_dropped": float((~merged[KEEP_COL]).mean()),
    }

    mask_cols = [
        "Metadata_Plate", "Metadata_Well", QC_PLATE_COL, QC_WELL_COL,
        "cell_count", "n_fovs", "bbox_source", KEEP_COL,
    ]
    mask = merged[mask_cols].copy()
    mask.attrs["cell_count_qc_summary"] = summary

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _qc_stem(dataset_name, cutoff, len(sampled), seed, label)
        mask.to_parquet(output_dir / f"{stem}.parquet", index=False)
        with (output_dir / f"{stem}_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    print(
        f"[cell-count QC] {dataset_name}: threshold={threshold:.3f}, "
        f"kept={summary['wells_after']}/{summary['wells_before']} "
        f"({summary['fraction_dropped']:.2%} dropped), "
        f"sampled={summary['sampled_wells']}",
        flush=True,
    )
    return mask


def apply_cell_count_qc_mask(df: pd.DataFrame, mask: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Filter a feature dataframe with a precomputed QC mask."""
    original_columns = list(df.columns)
    keyed = add_normalized_key_columns(df)
    joined = keyed.merge(
        mask[[QC_PLATE_COL, QC_WELL_COL, KEEP_COL, "cell_count"]],
        on=[QC_PLATE_COL, QC_WELL_COL],
        how="left",
    )
    missing = joined[KEEP_COL].isna()
    if missing.any():
        preview = joined.loc[missing, ["Metadata_Plate", "Metadata_Well"]].head(10)
        raise ValueError(
            f"Cell-count QC mask missing {int(missing.sum())}/{len(joined)} feature rows "
            f"for {dataset_name}. Preview: {preview.to_dict(orient='records')}"
        )
    filtered = joined[joined[KEEP_COL]].copy()
    return filtered[original_columns].reset_index(drop=True)
