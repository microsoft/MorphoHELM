#!/usr/bin/env python3
"""
MoA Enrichment Analysis for BBBC036 dataset.

Per-compound enrichment analysis:
  1. Load normalized features (per model) from parquet directory.
  2. (Optional) Filter to specific plates — default uses ALL plates.
  3. Aggregate to compound level (mean across wells per Metadata_broad_sample).
  4. Map compounds to MoA (from Metadata_moa column or ground truth CSV); drop singletons.
  5. Compute pairwise cosine similarity matrix.
  6. For each compound, rank neighbours by similarity.
     - Target = top 1% most similar.
     - Background = the rest.
     - Fisher exact test via scipy.stats.permutation_test (100 resamples).
  7. Report per-model: fraction significant, arithmetic mean OR, geometric mean OR.

Uses Haldane-Anscombe correction for odds ratio calculation (Weber et al. 2020).

Usage:
  python run_moa_enrichment.py \
    --features-dir /path/to/bbbc036_moa_normalized/ \
    --output results/moa_enrichment/bbbc036_moa_enrichment.pkl
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import fisher_exact, permutation_test
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qc.cell_count_qc import (
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    make_cell_count_qc_mask,
)


def load_features(features_dir):
    """Load per-model normalized parquets."""
    parquet_files = sorted(glob.glob(os.path.join(features_dir, "*_normalized.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No *_normalized.parquet files in {features_dir}")

    models = {}
    for fpath in parquet_files:
        name = os.path.basename(fpath).replace("_normalized.parquet", "")
        df = pd.read_parquet(fpath)
        df["Metadata_Plate"] = df["Metadata_Plate"].astype(str)
        models[name] = df
        print(f"  {name}: {df.shape}")
    return models


def filter_to_test_plates(df, plates):
    """Keep only rows from the 55 test plates."""
    return df[df["Metadata_Plate"].isin(plates)].reset_index(drop=True)


def aggregate_by_compound(df):
    """Mean-aggregate features per compound (Metadata_broad_sample).
    
    Excludes mock/DMSO control wells — only treated wells are aggregated.
    """
    feature_cols = [c for c in df.columns if c.startswith("PC") or c.startswith("feature_")]
    if "Metadata_broad_sample" not in df.columns:
        raise ValueError("Missing Metadata_broad_sample column")

    # Keep only treated wells (exclude mock/DMSO controls)
    if "Metadata_ASSAY_WELL_ROLE" in df.columns:
        df = df[df["Metadata_ASSAY_WELL_ROLE"] == "treated"]

    mask = df["Metadata_broad_sample"].notna() & (df["Metadata_broad_sample"] != "")
    df = df[mask]

    agg = df.groupby("Metadata_broad_sample")[feature_cols].mean().reset_index()
    return agg


def map_moa(agg_df, ground_truth_path):
    """Map Metadata_broad_sample to Metadata_moa via the ground truth CSV."""
    gt = pd.read_csv(ground_truth_path)
    moa_dict = dict(zip(gt["Metadata_broad_sample"], gt["Metadata_moa"]))
    agg_df["Metadata_moa"] = agg_df["Metadata_broad_sample"].map(moa_dict)

    n_before = len(agg_df)
    agg_df = agg_df.dropna(subset=["Metadata_moa"]).reset_index(drop=True)
    print(f"  MoA mapped: {len(agg_df)}/{n_before} compounds")

    moa_counts = agg_df["Metadata_moa"].value_counts()
    valid_moas = moa_counts[moa_counts >= 2].index
    agg_df = agg_df[agg_df["Metadata_moa"].isin(valid_moas)].reset_index(drop=True)
    print(f"  After dropping singleton MoAs: {len(agg_df)} compounds, "
          f"{agg_df['Metadata_moa'].nunique()} MoAs")
    return agg_df


def get_odds_ratio_haldane(target_hits, background_hits):
    """Calculate odds ratio using Haldane-Anscombe correction.

    Adds 0.5 to all cells when any cell is zero (Weber et al. 2020, PMID 32991790).
    Otherwise uses standard Fisher exact test.
    """
    table = np.array([
        [np.sum(np.array(target_hits) == 1),
         np.sum(np.array(target_hits) == 0)],
        [np.sum(np.array(background_hits) == 1),
         np.sum(np.array(background_hits) == 0)]
    ])
    if np.any(table == 0):
        table = table.astype(float) + 0.5
        odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
    else:
        odds_ratio, _ = fisher_exact(table, alternative='greater')
    return odds_ratio


def get_odds_ratio_imputation(target_hits, background_hits):
    """Calculate odds ratio using old imputation method.

    Uses Fisher exact test; replaces inf with background size.
    """
    table = np.array([
        [np.sum(np.array(target_hits) == 1),
         np.sum(np.array(target_hits) == 0)],
        [np.sum(np.array(background_hits) == 1),
         np.sum(np.array(background_hits) == 0)]
    ])
    odds_ratio, _ = fisher_exact(table, alternative='greater')
    if np.isinf(odds_ratio):
        odds_ratio = np.sum(np.array(background_hits) == 0)
    return odds_ratio


OR_METHODS = {
    "haldane": get_odds_ratio_haldane,
    "imputation": get_odds_ratio_imputation,
}


def run_moa_enrichment_for_model(agg_df, full_moa_map, testable_compounds,
                                 n_resamples=100, or_func=get_odds_ratio_haldane):
    """Per-compound MoA enrichment using 1% cutoff.

    Uses full_moa_map (ALL compounds with MoA) for neighbour matching,
    but only iterates over testable_compounds (non-singleton MoA) for testing.
    Self is excluded from the ranked match list.

    Parameters
    ----------
    or_func : callable
        Odds ratio function to use (get_odds_ratio_haldane or get_odds_ratio_imputation).
    """
    feature_cols = [c for c in agg_df.columns
                    if c.startswith("PC") or c.startswith("feature_")]
    features = agg_df[feature_cols].values
    compounds = agg_df["Metadata_broad_sample"].values
    n = len(compounds)

    sim_matrix = cosine_similarity(features)

    ranked_matches = {}
    for i in range(n):
        if compounds[i] not in testable_compounds:
            continue
        my_moa = full_moa_map.get(compounds[i])
        if my_moa is None:
            continue
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        order = np.argsort(sims)[::-1]  # descending similarity
        match_list = []
        for j in order:
            if j == i:
                continue
            match_list.append(int(full_moa_map.get(compounds[j]) == my_moa))
        ranked_matches[compounds[i]] = match_list

    print(f"  {n} compounds in similarity matrix, "
          f"{len(ranked_matches)} testable")

    # Per-compound permutation test with 1% cutoff
    odds = []
    pvals = []
    significant_compounds = []

    for compound in tqdm(sorted(ranked_matches.keys())):
        compound_rank = ranked_matches[compound]

        # Skip compounds with no MoA matches at all
        if np.sum(compound_rank) == 0:
            continue

        # Top 1% cutoff (matching CRISPR methodology)
        cutoff_items = max(2, len(compound_rank) // 100)

        if len(compound_rank) < cutoff_items + 2:
            continue

        target_hits = compound_rank[:cutoff_items]
        background_hits = compound_rank[cutoff_items:]

        if len(target_hits) < 2 or len(background_hits) < 2:
            continue

        result = permutation_test(
            (target_hits, background_hits),
            statistic=or_func,
            n_resamples=n_resamples,
            alternative='greater',
            random_state=61,
        )
        odds.append(result.statistic)
        pvals.append(result.pvalue)
        if result.pvalue < 0.05:
            significant_compounds.append(compound)

    n_sig = len(significant_compounds)
    n_proc = len(odds)
    frac_sig = n_sig / n_proc if n_proc > 0 else 0.0
    mean_odds = float(np.mean(odds)) if odds else 0.0
    mean_log_odds = float(np.mean(np.log(odds))) if odds else 0.0
    geometric_mean_odds = float(np.exp(np.mean(np.log(odds)))) if odds else 0.0

    print(f"  Processed: {n_proc}, Significant: {n_sig}, "
          f"Fraction: {frac_sig:.4f}, Mean Odds: {mean_odds:.2f}, "
          f"Geometric Mean Odds: {geometric_mean_odds:.2f}")

    return {
        "n_processed": n_proc,
        "n_significant": n_sig,
        "fraction_significant": frac_sig,
        "mean_odds": mean_odds,
        "mean_log_odds": mean_log_odds,
        "geometric_mean_odds": geometric_mean_odds,
        "significant_compounds": significant_compounds,
        "all_odds": odds,
        "all_pvals": pvals,
    }


def main():
    parser = argparse.ArgumentParser(description="BBBC036 MoA Enrichment Analysis")
    parser.add_argument("--features-dir", required=True,
                        help="Directory with *_normalized.parquet files")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to BBBC036_v1_DatasetGroundTruth.csv (optional; "
                             "if omitted, uses Metadata_moa column from parquet)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Which models to analyse (default: all)")
    parser.add_argument("--output", required=True,
                        help="Output pickle path for results")
    parser.add_argument("--n-resamples", type=int, default=100,
                        help="Number of permutation resamples (default: 100)")
    parser.add_argument("--or-method", choices=["haldane", "imputation"], default="haldane",
                        help="Odds ratio method: 'haldane' (Haldane-Anscombe correction) "
                             "or 'imputation' (old: replace inf with background size)")
    parser.add_argument("--plates", nargs="*", default=None,
                        help="Optional plate filter (default: use ALL plates)")
    parser.add_argument("--randomize-labels", action="store_true",
                        help="Shuffle MoA labels across compounds (preserving group sizes) "
                             "to produce a random baseline")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for label shuffling (default: 42)")
    add_cell_count_qc_args(parser)
    args = parser.parse_args()

    print("Loading features...")
    all_models = load_features(args.features_dir)

    if args.models:
        selected = [m for m in args.models if m in all_models]
        missing = set(args.models) - set(selected)
        if missing:
            print(f"WARNING: models not found: {missing}")
    else:
        selected = list(all_models.keys())
    print(f"Models: {selected}")

    or_func = OR_METHODS[args.or_method]
    print(f"OR method: {args.or_method}")
    if args.randomize_labels:
        print(f"BASELINE MODE: MoA labels will be shuffled (seed={args.random_seed})")

    all_results = {}
    qc_mask = None
    for model_name in selected:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        df = all_models[model_name]
        if args.cell_count_qc:
            if qc_mask is None:
                qc_mask = make_cell_count_qc_mask(
                    df,
                    dataset_name="bbbc036",
                    cell_counts_dir=args.qc_cell_counts_dir,
                    cutoff=args.qc_cell_count_cutoff,
                    sample_wells=args.qc_sample_wells,
                    seed=args.qc_seed,
                    output_dir=os.path.join(os.path.dirname(args.output) or ".", "cell_count_qc"),
                )
            before = len(df)
            df = apply_cell_count_qc_mask(df, qc_mask, "bbbc036")
            print(f"  Cell-count QC: {before} -> {len(df)} wells")

        # Optional plate filtering
        if args.plates:
            df = filter_to_test_plates(df, args.plates)
            print(f"  Wells after plate filter: {len(df)}")
        else:
            print(f"  Wells (all plates): {len(df)}")

        if len(df) == 0:
            print(f"  SKIP: no wells")
            continue

        # Aggregate by compound
        agg = aggregate_by_compound(df)
        print(f"  Compounds after aggregation: {len(agg)}")

        # Get MoA labels — from ground truth CSV or from parquet column
        if args.ground_truth:
            gt = pd.read_csv(args.ground_truth)
            gt_compounds = set(gt["Metadata_broad_sample"].unique())
            agg = agg[agg["Metadata_broad_sample"].isin(gt_compounds)].reset_index(drop=True)
            moa_dict = dict(zip(gt["Metadata_broad_sample"], gt["Metadata_moa"]))
            agg["Metadata_moa"] = agg["Metadata_broad_sample"].map(moa_dict)
        else:
            # Use Metadata_moa already in the parquet
            if "Metadata_moa" not in df.columns:
                print(f"  SKIP: no Metadata_moa column and no --ground-truth provided")
                continue
            # Re-derive MoA per compound from the well-level data
            moa_per_compound = df.groupby("Metadata_broad_sample")["Metadata_moa"].first()
            agg["Metadata_moa"] = agg["Metadata_broad_sample"].map(moa_per_compound)

        n_before = len(agg)
        agg = agg.dropna(subset=["Metadata_moa"]).reset_index(drop=True)
        print(f"  Compounds with MoA labels: {len(agg)}/{n_before}")

        # Build MoA mapping — ALL compounds get MoA for neighbour matching
        full_moa_map = {}
        for _, row in agg.iterrows():
            m = row["Metadata_moa"]
            if pd.notna(m):
                full_moa_map[row["Metadata_broad_sample"]] = m

        # Only non-singleton MoA compounds are iterated for testing
        moa_counts = agg["Metadata_moa"].value_counts()
        valid_moas = set(moa_counts[moa_counts >= 2].index)
        testable_compounds = set()
        for comp, moa in full_moa_map.items():
            if moa in valid_moas:
                testable_compounds.add(comp)

        n_testable = len(testable_compounds)
        n_moas = len(valid_moas)
        print(f"  Testable (non-singleton MoA): {n_testable}/{len(agg)}, "
              f"{n_moas} MoAs")

        if n_testable < 5:
            print(f"  SKIP: too few compounds with MoA")
            continue

        # Optionally shuffle MoA labels (baseline mode)
        if args.randomize_labels:
            rng = np.random.default_rng(args.random_seed)
            compounds_list = list(full_moa_map.keys())
            moa_values = list(full_moa_map.values())
            rng.shuffle(moa_values)
            full_moa_map = dict(zip(compounds_list, moa_values))
            print(f"  [BASELINE] Shuffled MoA labels (seed={args.random_seed})")

        result = run_moa_enrichment_for_model(
            agg, full_moa_map, testable_compounds,
            n_resamples=args.n_resamples, or_func=or_func)
        all_results[model_name] = result

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
