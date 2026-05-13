"""Aggregate per-GPU inference results into well-level feature parquet files.

Strategy:
  1. Parallel pkl loading via ProcessPoolExecutor (I/O-bound, benefits from concurrency)
  2. For single-channel models: average FOVs per (plate, well, channel), then concatenate channels
  3. For multi-channel models: average FOVs per (plate, well)
  4. DINOv2 produces two outputs: cls_token and patch_token (+ canonical dino_v2 alias)
  5. Optionally merge plate metadata (Metadata_Source, Metadata_Batch)
  6. Parallel model aggregation when running --model all

Usage:
  python aggregate_features.py --config feature-extraction/configs/inference_splits.yaml --model dino_v2
  python aggregate_features.py --config feature-extraction/configs/inference_splits.yaml --model all
  python aggregate_features.py --config feature-extraction/configs/inference_splits.yaml --model all --output-path results/aggregated/cpg-tgt2
"""

import os
import re
import glob
import pickle
import argparse
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model category classification
# ---------------------------------------------------------------------------

DINO_MODELS = {"dino_v2", "dino_v2_448"}
SINGLE_CHANNEL_MODELS = DINO_MODELS | {"resnet", "resnet_untrained", "vgg19"}
OPENPHENOM_MODELS = {"open_phenom"}
MULTI_CHANNEL_MODELS = {"cloome", "subcell"}

NUM_CHANNELS = 5  # Canonical Cell Painting order: AGP, DNA, ER, Mito, RNA
BBBC036_CHANNEL_MAP = {
    "Ph_golgi": 0,
    "Hoechst": 1,
    "ERSyto": 2,
    "Mito": 3,
    "ERSytoBleed": 4,
}


def get_model_category(model_name: str) -> str:
    if model_name in SINGLE_CHANNEL_MODELS:
        return "single_channel"
    elif model_name in OPENPHENOM_MODELS:
        return "openphenom"
    elif model_name in MULTI_CHANNEL_MODELS:
        return "multi_channel"
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

# Single-channel / OpenPhenom filename patterns:
# CPG0016: "A04_i1_ch0.png" or "AA01_i1_ch0.png"
# BBBC036: "cdp2bioactives_a01_s1_w127df6541-..._ch_Hoechst.png"
_SINGLE_CH_CPG_RE = re.compile(r"^([A-Z]+\d+)_(i\d+)_ch(\d+)\.png$")
_SINGLE_CH_BBBC_RE = re.compile(r"^.+?_([A-Za-z]+\d+)_([A-Za-z]*\d+)_w(\d)")

# Multi-channel (after channel_concat_collate_fn): "plate_well_fov"
# CPG0016: "25983_A04_1" or "25983_AA01_1"
# BBBC036: "25983_p06_s3"
_MULTI_CH_RE = re.compile(r"^(.+?)_([A-Za-z]+\d+)_([A-Za-z]*\d+)$")


def parse_single_channel_filename(filename: str):
    """Parse single-channel filename → (well, fov, channel_idx).

    CPG0016: 'A04_i1_ch0.png'  → ('A04', 'i1', 0)
    BBBC036: 'cdp2bioactives_a01_s1_w1{UUID}_ch_Hoechst.png' → ('a01', 's1', 1)
    """
    m = _SINGLE_CH_CPG_RE.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    if "_ch_" in filename:
        channel_name = filename.rsplit("_ch_", 1)[1].rsplit(".", 1)[0]
        if channel_name not in BBBC036_CHANNEL_MAP:
            raise ValueError(f"Unknown BBBC036 channel '{channel_name}' in filename: '{filename}'")
        m = _SINGLE_CH_BBBC_RE.match(filename)
        if m:
            return m.group(1), m.group(2), BBBC036_CHANNEL_MAP[channel_name]
    m = _SINGLE_CH_BBBC_RE.match(filename)
    if m:
        # Legacy BBBC036 fallback: channels are 1-indexed (w1..w5).
        return m.group(1), m.group(2), int(m.group(3)) - 1
    raise ValueError(f"Cannot parse single-channel filename: '{filename}'")


def parse_multi_channel_filename(filename: str):
    """Parse 'plate_well_fov' → (plate, well, fov)."""
    m = _MULTI_CH_RE.match(filename)
    if not m:
        raise ValueError(f"Cannot parse multi-channel filename: '{filename}'")
    return m.group(1), m.group(2), m.group(3)


def parse_openphenom_filename(filename: str, plate_name: str):
    """Parse OpenPhenom output filename from HF or direct inference."""
    try:
        well, fov, _ = parse_single_channel_filename(filename)
        return str(plate_name), well, fov
    except ValueError:
        plate_from_fn, well, fov = parse_multi_channel_filename(filename)
        return str(plate_from_fn or plate_name), well, fov


# ---------------------------------------------------------------------------
# Streaming accumulator
# ---------------------------------------------------------------------------

class StreamingAggregator:
    """Accumulate running sum and count per aggregation key."""

    def __init__(self):
        self._sum = {}   # key -> np.ndarray (float64 for numeric stability)
        self._count = {} # key -> int

    def add(self, key, features: np.ndarray):
        if key in self._sum:
            self._sum[key] += features.astype(np.float64)
            self._count[key] += 1
        else:
            self._sum[key] = features.astype(np.float64).copy()
            self._count[key] = 1

    def merge(self, other: "StreamingAggregator"):
        """Merge another aggregator into this one (for combining parallel results)."""
        for key in other._sum:
            if key in self._sum:
                self._sum[key] += other._sum[key]
                self._count[key] += other._count[key]
            else:
                self._sum[key] = other._sum[key]
                self._count[key] = other._count[key]

    def mean(self, key):
        return (self._sum[key] / self._count[key]).astype(np.float32)

    def keys(self):
        return self._sum.keys()

    def __len__(self):
        return len(self._sum)


# ---------------------------------------------------------------------------
# Parallel pkl loading
# ---------------------------------------------------------------------------

def iter_pkl_files(model_dir: str):
    """Return sorted list of results_part_*.pkl file paths."""
    pattern = os.path.join(model_dir, "gpu_*_results", "results_part_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No pickle files found matching {pattern}")
    return files


def _load_pkl(pkl_path: str):
    """Load a single pickle file. Used as worker function for parallel loading."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]


def load_all_pkls_parallel(pkl_files: list, num_workers: int = 4):
    """Load all pkl files in parallel and yield batches in order."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_pkl, p): i for i, p in enumerate(pkl_files)}
        results_by_idx = {}
        for future in as_completed(futures):
            idx = futures[future]
            results_by_idx[idx] = future.result()
    # Yield in original order
    for i in range(len(pkl_files)):
        yield from results_by_idx[i]


# ---------------------------------------------------------------------------
# Aggregation: single-channel models (DINOv2, ResNet, ResNet-UT, VGG19)
# ---------------------------------------------------------------------------

def aggregate_single_channel(model_dir: str, model_name: str, num_workers: int = 4, logger=None):
    """Aggregate single-channel model results.

    Returns dict mapping feature_type → DataFrame.
    DINOv2 variants: {'cls_token': df, 'patch_token': df}
    Others: {'features': df}
    """
    is_dino = model_name in DINO_MODELS
    feature_keys = ["cls_token_features", "patch_token_features"] if is_dino else ["features"]

    # Partition aggregators by plate: {plate: {fk: StreamingAggregator}}
    plate_aggs = defaultdict(lambda: {fk: StreamingAggregator() for fk in feature_keys})

    pkl_files = iter_pkl_files(model_dir)
    total_samples = 0
    seen_keys = set()

    if logger:
        logger.info(f"Loading {len(pkl_files)} pkl files with {num_workers} workers...")

    for batch in load_all_pkls_parallel(pkl_files, num_workers):
        filenames = batch["filename"]
        plates = batch["plate_name"]
        n = len(filenames)
        total_samples += n

        for i in range(n):
            well, fov, ch_idx = parse_single_channel_filename(filenames[i])
            plate = plates[i]

            sample_key = (plate, well, fov, ch_idx)
            if sample_key in seen_keys:
                continue
            seen_keys.add(sample_key)

            for fk in feature_keys:
                feat_array = batch[fk]
                plate_aggs[plate][fk].add((well, ch_idx), feat_array[i])

    del seen_keys

    if logger:
        logger.info(f"Loaded {total_samples} samples from {len(pkl_files)} files, {len(plate_aggs)} plates")

    # Build per-plate DataFrames, then concatenate
    result = {fk_label: [] for fk_label in (["cls_token", "patch_token"] if is_dino else ["features"])}
    base_dim = None

    for plate in sorted(plate_aggs.keys()):
        for fk, fk_label in zip(feature_keys, result.keys()):
            agg = plate_aggs[plate][fk]
            if not agg.keys():
                continue

            if base_dim is None:
                first_key = next(iter(agg.keys()))
                base_dim = agg.mean(first_key).shape[0]

            wells = sorted({k[0] for k in agg.keys()})
            for well in wells:
                channel_features = []
                complete = True
                for ch in range(NUM_CHANNELS):
                    key = (well, ch)
                    if key in agg._sum:
                        channel_features.append(agg.mean(key))
                    else:
                        complete = False
                        break

                if not complete:
                    if logger:
                        logger.warning(f"Incomplete channels for plate={plate} well={well}, skipping")
                    continue

                concatenated = np.concatenate(channel_features)
                result[fk_label].append((plate, well, concatenated))

        del plate_aggs[plate]

    # Convert to DataFrames
    final_result = {}
    for fk_label, rows in result.items():
        if not rows:
            continue

        col_names = []
        for ch in range(NUM_CHANNELS):
            for i in range(base_dim):
                col_names.append(f"ch{ch}_feature_{i}")

        plates_col = [r[0] for r in rows]
        wells_col = [r[1] for r in rows]
        features_matrix = np.vstack([r[2] for r in rows])

        df = pd.DataFrame(features_matrix, columns=col_names)
        df.insert(0, "Metadata_Plate", plates_col)
        df.insert(1, "Metadata_Well", wells_col)
        final_result[fk_label] = df

        if logger:
            logger.info(f"  {fk_label}: {len(df)} wells, {features_matrix.shape[1]} features "
                        f"(base_dim={base_dim} × {NUM_CHANNELS} channels)")

    return final_result


# ---------------------------------------------------------------------------
# Aggregation: OpenPhenom (collate_cpjump groups 5 channels)
# ---------------------------------------------------------------------------

def aggregate_openphenom(model_dir: str, num_workers: int = 4, logger=None):
    """Aggregate OpenPhenom results. Average FOVs per (plate, well).

    HF inference emits a representative single-channel filename such as
    ``A01_i1_ch0.png``. Direct inference emits ``plate_well_fov``. Both are
    accepted so direct cpg-MoA can converge to the same aggregate schema.
    """
    agg = StreamingAggregator()
    pkl_files = iter_pkl_files(model_dir)
    total_samples = 0
    seen = set()

    if logger:
        logger.info(f"Loading {len(pkl_files)} pkl files with {num_workers} workers...")

    for batch in load_all_pkls_parallel(pkl_files, num_workers):
        filenames = batch["filename"]
        plates = batch["plate_name"]
        features = batch["features"]
        n = len(filenames)

        for i in range(n):
            plate, well, fov = parse_openphenom_filename(filenames[i], plates[i])

            sample_key = (plate, well, fov)
            if sample_key in seen:
                continue
            seen.add(sample_key)

            key = (plate, well)
            agg.add(key, features[i])
            total_samples += 1

    del seen
    if logger:
        logger.info(f"Processed {total_samples} samples from {len(pkl_files)} files")

    return _aggregator_to_df(agg, logger)


# ---------------------------------------------------------------------------
# Aggregation: CLOOME / SubCell (channel_concat_collate groups channels)
# ---------------------------------------------------------------------------

def aggregate_multi_channel(model_dir: str, num_workers: int = 4, logger=None):
    """Aggregate CLOOME/SubCell results. Filename is 'plate_well_fov'. Average over FOVs."""
    agg = StreamingAggregator()
    pkl_files = iter_pkl_files(model_dir)
    total_samples = 0
    seen = set()

    if logger:
        logger.info(f"Loading {len(pkl_files)} pkl files with {num_workers} workers...")

    for batch in load_all_pkls_parallel(pkl_files, num_workers):
        filenames = batch["filename"]
        plates = batch["plate_name"]
        features = batch["features"]
        n = len(filenames)

        for i in range(n):
            plate_from_fn, well, fov = parse_multi_channel_filename(filenames[i])
            plate = plates[i]

            sample_key = (plate, well, fov)
            if sample_key in seen:
                continue
            seen.add(sample_key)

            key = (plate, well)
            agg.add(key, features[i])
            total_samples += 1

    del seen
    if logger:
        logger.info(f"Processed {total_samples} samples from {len(pkl_files)} files")

    return _aggregator_to_df(agg, logger)


def _aggregator_to_df(agg: StreamingAggregator, logger=None):
    """Convert a flat (plate, well) aggregator to {'features': DataFrame}."""
    keys_sorted = sorted(agg.keys())
    if not keys_sorted:
        raise RuntimeError("No data found in aggregator")

    first_key = keys_sorted[0]
    dim = agg.mean(first_key).shape[0]

    plates_col = [k[0] for k in keys_sorted]
    wells_col = [k[1] for k in keys_sorted]
    features_matrix = np.vstack([agg.mean(k) for k in keys_sorted])

    col_names = [f"feature_{i}" for i in range(dim)]
    df = pd.DataFrame(features_matrix, columns=col_names)
    df.insert(0, "Metadata_Plate", plates_col)
    df.insert(1, "Metadata_Well", wells_col)

    if logger:
        logger.info(f"  {len(df)} wells, {dim} features")

    return {"features": df}


# ---------------------------------------------------------------------------
# Metadata merging
# ---------------------------------------------------------------------------

def merge_plate_metadata(df: pd.DataFrame, metadata_path: str, logger=None):
    """Merge Metadata_Source / Metadata_Batch from plate.csv.gz if available."""
    candidates = [
        os.path.join(metadata_path, "metadata", "plate.csv.gz"),
        os.path.join(metadata_path, "plate.csv.gz"),
    ]
    plate_csv = None
    for c in candidates:
        if os.path.exists(c):
            plate_csv = c
            break

    if plate_csv is None:
        if logger:
            logger.info("No plate.csv.gz found, skipping metadata merge")
        return df

    plates_meta = pd.read_csv(plate_csv)
    merge_cols = [c for c in ["Metadata_Source", "Metadata_Batch", "Metadata_PlateType"]
                  if c in plates_meta.columns]
    if not merge_cols:
        return df

    plates_meta = plates_meta[["Metadata_Plate"] + merge_cols].drop_duplicates()
    merged = df.merge(plates_meta, on="Metadata_Plate", how="left")

    # Reorder: metadata columns first, then features
    meta_cols = ["Metadata_Plate", "Metadata_Well"] + merge_cols
    feat_cols = [c for c in merged.columns if c not in meta_cols]
    merged = merged[meta_cols + feat_cols]

    if logger:
        logger.info(f"Merged plate metadata columns: {merge_cols}")

    return merged


def validate_aggregated_df(df: pd.DataFrame, label: str):
    """Validate the well-level aggregate contract."""
    required = {"Metadata_Plate", "Metadata_Well"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing required metadata columns: {missing}")
    duplicates = df.duplicated(subset=["Metadata_Plate", "Metadata_Well"]).sum()
    if duplicates:
        raise ValueError(f"{label}: found {duplicates} duplicate plate/well rows")
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    if not feature_cols:
        raise ValueError(f"{label}: no feature columns found")
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"{label}: non-numeric feature columns: {non_numeric[:10]}")
    values = df[feature_cols].to_numpy()
    if np.isnan(values).all(axis=0).any():
        raise ValueError(f"{label}: at least one feature column is all-NaN")
    return {
        "rows": int(len(df)),
        "metadata_columns": int(len(df.columns) - len(feature_cols)),
        "feature_columns": int(len(feature_cols)),
        "nan_values": int(np.isnan(values).sum()),
        "inf_values": int(np.isinf(values).sum()),
    }


def summarize_parquet(path: str) -> Dict[str, object]:
    df = pd.read_parquet(path)
    summary = validate_aggregated_df(df, os.path.basename(path))
    summary["path"] = path
    return summary


def select_models(cfg: dict, model_override: Optional[str] = None) -> List[str]:
    if model_override and model_override != "all":
        if model_override not in cfg["models"]:
            raise ValueError(f"Unknown model '{model_override}'. Available: {sorted(cfg['models'])}")
        return [model_override]
    return [name for name, mcfg in cfg["models"].items() if mcfg.get("enabled", False)]


# ---------------------------------------------------------------------------
# Single-model aggregation entry point
# ---------------------------------------------------------------------------

def aggregate_model(model_name: str, inference_output_path: str,
                    aggregated_output_path: str, metadata_path: str = None,
                    num_workers: int = 4, logger=None):
    """Aggregate a single model's results and save as parquet.

    Args:
        model_name: Name of the model to aggregate.
        inference_output_path: Path to inference results (contains {model}/gpu_* dirs).
        aggregated_output_path: Path to write aggregated parquet files.
        metadata_path: Optional path to plate metadata for merging.
        num_workers: Number of parallel workers for pkl loading.
        logger: Logger instance.
    """
    model_dir = os.path.join(inference_output_path, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    category = get_model_category(model_name)
    logger.info(f"Aggregating {model_name} (category={category}) from {model_dir}")
    t0 = time.time()

    if category == "single_channel":
        results = aggregate_single_channel(model_dir, model_name, num_workers=num_workers, logger=logger)
    elif category == "openphenom":
        results = aggregate_openphenom(model_dir, num_workers=num_workers, logger=logger)
    elif category == "multi_channel":
        results = aggregate_multi_channel(model_dir, num_workers=num_workers, logger=logger)

    os.makedirs(aggregated_output_path, exist_ok=True)
    saved_files = []

    for feat_type, df in results.items():
        if metadata_path:
            df = merge_plate_metadata(df, metadata_path, logger=logger)

        df = df.sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)
        validation = validate_aggregated_df(df, f"{model_name}/{feat_type}")

        # Output filenames:
        #   dino_v2 cls_token → dino_v2_cls_token_aggregated.parquet
        #   dino_v2_448 cls_token → dino_v2_448_cls_token_aggregated.parquet
        #   others → {model_name}_aggregated.parquet
        if model_name in DINO_MODELS and feat_type != "features":
            out_name = f"{model_name}_{feat_type}_aggregated.parquet"
        else:
            out_name = f"{model_name}_aggregated.parquet"

        out_path = os.path.join(aggregated_output_path, out_name)
        df.to_parquet(out_path, index=False)
        saved_files.append(out_path)
        logger.info(
            f"Saved: {out_path} ({validation['rows']} wells × "
            f"{validation['feature_columns']} features)"
        )

    # DINOv2 variants: also save canonical alias (= cls_token)
    if model_name in DINO_MODELS and "cls_token" in results:
        canonical_path = os.path.join(aggregated_output_path, f"{model_name}_aggregated.parquet")
        df_canonical = results["cls_token"]
        if metadata_path:
            df_canonical = merge_plate_metadata(df_canonical, metadata_path, logger=logger)
        df_canonical = df_canonical.sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)
        validate_aggregated_df(df_canonical, f"{model_name}/canonical")
        df_canonical.to_parquet(canonical_path, index=False)
        saved_files.append(canonical_path)
        logger.info(f"Saved canonical DINOv2 alias: {canonical_path}")

    elapsed = time.time() - t0
    logger.info(f"Finished {model_name} in {elapsed:.1f}s")
    return saved_files


def write_aggregation_manifest(
    manifest_path: str,
    cfg: dict,
    models: List[str],
    inference_output_path: str,
    aggregated_output_path: str,
    summaries: List[Dict[str, object]],
):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    manifest = {
        "dataset": cfg.get("dataset", {}).get("name"),
        "split": cfg.get("dataset", {}).get("split") or cfg.get("dataset", {}).get("name"),
        "models": models,
        "inference_output_path": inference_output_path,
        "aggregated_output_path": aggregated_output_path,
        "outputs": summaries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_aggregation_for_config(
    cfg: dict,
    model_override: Optional[str] = None,
    output_path: Optional[str] = None,
    inference_output_path: Optional[str] = None,
    num_workers: int = 4,
    parallel_models: bool = False,
    continue_on_error: bool = False,
    logger=None,
) -> List[Dict[str, object]]:
    logger = logger or logging.getLogger("aggregate")
    mode = cfg["execution"]["mode"]
    section = cfg[mode]
    inference_output_path = inference_output_path or section["output_path"]
    agg_cfg = cfg.get("aggregation", {})
    aggregated_output_path = output_path or agg_cfg.get("output_path") or inference_output_path
    metadata_path = section.get("git_clone_dir") or section.get("metadata_path")
    models = select_models(cfg, model_override)

    logger.info(f"Inference output path: {inference_output_path}")
    logger.info(f"Aggregated output path: {aggregated_output_path}")
    logger.info(f"Metadata path: {metadata_path}")
    logger.info(f"Models to aggregate: {models}")

    saved_files = []
    failures = []
    if parallel_models and len(models) > 1:
        worker_args = [
            (m, inference_output_path, aggregated_output_path, metadata_path, max(1, num_workers // len(models)))
            for m in models
        ]
        with ProcessPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(_aggregate_model_worker, a) for a in worker_args]
            for future in as_completed(futures):
                model_name, files, error = future.result()
                if error:
                    logger.error(f"Failed to aggregate {model_name}:\n{error}")
                    failures.append(model_name)
                    if not continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        raise RuntimeError(f"Failed to aggregate {model_name}")
                else:
                    logger.info(f"Aggregated {model_name}: {len(files)} files saved")
                    saved_files.extend(files)
    else:
        for model_name in models:
            try:
                saved_files.extend(
                    aggregate_model(
                        model_name,
                        inference_output_path,
                        aggregated_output_path,
                        metadata_path,
                        num_workers=num_workers,
                        logger=logger,
                    )
                )
            except Exception as exc:
                logger.exception(f"Failed to aggregate {model_name}")
                failures.append(model_name)
                if not continue_on_error:
                    raise

    if failures:
        raise RuntimeError(f"Aggregation failed for models: {failures}")

    summaries = [summarize_parquet(path) for path in saved_files]
    write_aggregation_manifest(
        os.path.join(aggregated_output_path, "aggregation_manifest.json"),
        cfg,
        models,
        inference_output_path,
        aggregated_output_path,
        summaries,
    )
    return summaries


def _aggregate_model_worker(args):
    """Top-level worker for parallel model aggregation (must be picklable)."""
    model_name, inference_output_path, aggregated_output_path, metadata_path, num_workers = args
    # Each worker gets its own logger
    logger = logging.getLogger(f"aggregate.{model_name}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"%(asctime)s [{model_name}] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
    try:
        files = aggregate_model(model_name, inference_output_path, aggregated_output_path,
                                metadata_path, num_workers=num_workers, logger=logger)
        return model_name, files, None
    except Exception as e:
        import traceback
        return model_name, [], traceback.format_exc()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate inference results to well-level features")
    parser.add_argument("--config", type=str, required=True, help="Path to inference config YAML")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name to aggregate, or 'all' for all enabled models")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Override: aggregated output directory (default: from config aggregation.output_path)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers for pkl loading per model")
    parser.add_argument("--parallel-models", action="store_true", default=False,
                        help="Aggregate models in parallel (uses more memory)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue aggregating remaining models if one model fails")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [aggregate] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("aggregate")

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    run_aggregation_for_config(
        cfg,
        model_override=args.model,
        output_path=args.output_path,
        num_workers=args.num_workers,
        parallel_models=args.parallel_models,
        continue_on_error=args.continue_on_error,
        logger=logger,
    )
    total_time = time.time() - t0
    logger.info(f"Aggregation complete in {total_time:.1f}s")


if __name__ == "__main__":
    main()
