"""Build MoA cross-source profiles from normalized well-level parquets.

Creates `moa_cross_source_profiles.pkl`: a dict mapping model_name → DataFrame
with columns: Metadata_JCP2022, Metadata_Source, Metadata_Batch, PC1-PC128, Metadata_MoA.

Each row is one compound aggregated per (source, batch) combination.
Controls are excluded; only MoA-labeled compounds are kept.

Usage:
    python build_moa_profiles.py \
        --normalized-dir data/main_paper_inputs/normalized/CSAll_Plate__PCA64__MADCtrl_Plate__NoSph/cpg-moa \
        --moa-labels feature-extraction/configs/labeled_moa_samples.csv \
        --plate-metadata feature-extraction/metadata/metadata.parquet \
        --output data/moa_cross_source_profiles.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qc.cell_count_qc import (
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    make_cell_count_qc_mask,
)


CONTROL_JCP = "JCP2022_033924"  # DMSO

# Model names in normalized parquets → canonical names in profiles
MODEL_NAME_MAP = {
    "dino_v2_cls_token": "dino_v2",
    "dino_v2_patch_token": "dino_v2_patch",
    "resnet": "resnet",
    "resnet_untrained": "resnet_untrained",
    "open_phenom": "openphenom",
    "cloome": "cloome",
    "subcell": "subcell",
    "cellprofiler": "cellprofiler",
}


def build_profiles_for_model(
    parquet_path: str,
    moa_labels: pd.DataFrame,
    plate_metadata: pd.DataFrame,
    qc_mask: pd.DataFrame | None = None,
    qc_args: argparse.Namespace | None = None,
    qc_output_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load normalized parquet and build compound-level MoA profiles."""
    print(f"  Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"    {len(df)} wells loaded")

    if qc_args is not None and qc_args.cell_count_qc:
        if qc_mask is None:
            qc_mask = make_cell_count_qc_mask(
                df,
                dataset_name="cpg-moa",
                cell_counts_dir=qc_args.qc_cell_counts_dir,
                cutoff=qc_args.qc_cell_count_cutoff,
                sample_wells=qc_args.qc_sample_wells,
                seed=qc_args.qc_seed,
                output_dir=qc_output_dir,
            )
        n_before_qc = len(df)
        df = apply_cell_count_qc_mask(df, qc_mask, "cpg-moa")
        print(f"    Cell-count QC: {n_before_qc} -> {len(df)} wells")

    pc_cols = [c for c in df.columns if c.startswith("PC") or c.startswith("F_")]
    print(f"    {len(pc_cols)} PC features")

    # Ensure we have JCP2022 column (should come from normalization)
    if "Metadata_JCP2022" not in df.columns:
        print("    Joining JCP2022 from plate metadata...")
        jcp_map = plate_metadata[
            ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
        ].drop_duplicates()
        df = df.merge(jcp_map, on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"], how="inner")
        df = df.drop_duplicates(subset=["Metadata_Source", "Metadata_Plate", "Metadata_Well"], keep="first")
        print(f"    After JCP2022 join: {len(df)} wells")

    # Remove controls
    n_before = len(df)
    df = df[df["Metadata_JCP2022"] != CONTROL_JCP].copy()
    print(f"    Removed {n_before - len(df)} control wells, {len(df)} remain")

    # Join MoA labels
    moa_map = moa_labels[["Metadata_JCP2022", "Metadata_MoA"]].drop_duplicates("Metadata_JCP2022")
    df = df.merge(moa_map, on="Metadata_JCP2022", how="inner")
    print(f"    After MoA join: {len(df)} wells, {df['Metadata_JCP2022'].nunique()} compounds")

    # Check for NaN/Inf in features
    feat_arr = df[pc_cols].values
    n_nan = np.isnan(feat_arr).sum()
    n_inf = np.isinf(feat_arr).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"    WARNING: {n_nan} NaN, {n_inf} Inf in features — replacing with 0")
        df[pc_cols] = np.nan_to_num(feat_arr, nan=0, posinf=0, neginf=0)

    # Aggregate by (compound, source, batch) → mean of PC features
    agg = df.groupby(["Metadata_JCP2022", "Metadata_Source", "Metadata_Batch"]).agg(
        {**{col: "mean" for col in pc_cols}, "Metadata_MoA": "first"}
    ).reset_index()

    print(f"    Result: {len(agg)} profiles, {agg['Metadata_JCP2022'].nunique()} compounds, "
          f"{agg['Metadata_Source'].nunique()} sources, {agg['Metadata_Batch'].nunique()} batches")

    return agg, qc_mask


def main():
    parser = argparse.ArgumentParser(description="Build MoA cross-source profiles")
    parser.add_argument("--normalized-dir", required=True,
                        help="Directory containing *_normalized.parquet files")
    parser.add_argument("--moa-labels", required=True,
                        help="Path to labeled_moa_samples.csv")
    parser.add_argument("--plate-metadata", required=True,
                        help="Path to metadata_with_controls.parquet")
    parser.add_argument("--output", required=True,
                        help="Output path for moa_cross_source_profiles.pkl")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model names to process (default: all found)")
    add_cell_count_qc_args(parser)
    args = parser.parse_args()

    # Load MoA labels
    print("Loading MoA labels...")
    moa_labels = pd.read_csv(args.moa_labels)
    if "Unnamed: 0" in moa_labels.columns:
        moa_labels.drop(columns=["Unnamed: 0"], inplace=True)
    print(f"  {len(moa_labels)} samples, {moa_labels['Metadata_MoA'].nunique()} MoAs, "
          f"{moa_labels['Metadata_JCP2022'].nunique()} compounds")

    # Load plate metadata
    print("Loading plate metadata...")
    plate_metadata = pd.read_parquet(args.plate_metadata)
    print(f"  {len(plate_metadata)} rows")

    # Find normalized parquets
    parquet_files = sorted([
        f for f in os.listdir(args.normalized_dir)
        if f.endswith("_normalized.parquet")
    ])
    print(f"Found {len(parquet_files)} normalized parquets: {parquet_files}")

    profiles = {}
    qc_mask = None
    qc_output_dir = os.path.join(os.path.dirname(args.output) or ".", "cell_count_qc")
    for pf in parquet_files:
        model_name = pf.replace("_normalized.parquet", "")
        if args.models and model_name not in args.models:
            continue

        canonical = MODEL_NAME_MAP.get(model_name, model_name)
        print(f"\nProcessing {model_name} → {canonical}...")

        parquet_path = os.path.join(args.normalized_dir, pf)
        profile_df, qc_mask = build_profiles_for_model(
            parquet_path,
            moa_labels,
            plate_metadata,
            qc_mask=qc_mask,
            qc_args=args,
            qc_output_dir=qc_output_dir,
        )
        profiles[canonical] = profile_df

    # Save
    print(f"\nSaving profiles for {len(profiles)} models to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(profiles, f)

    # Summary
    print("\n=== SUMMARY ===")
    for name, df in profiles.items():
        print(f"  {name}: {df.shape} — {df['Metadata_JCP2022'].nunique()} compounds, "
              f"{df['Metadata_Source'].nunique()} sources")


if __name__ == "__main__":
    main()
