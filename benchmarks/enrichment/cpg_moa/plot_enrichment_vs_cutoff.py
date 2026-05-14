#!/usr/bin/env python3
"""
Enrichment vs Cutoff line graph for BBBC036 MoA enrichment.

Sweeps cutoff percentages (1%, 5%, 10%, 15%) and plots how the odds ratio
varies per model. Each model is a separate line.

The similarity matrix and ranked matches are computed once per model;
only the cutoff split changes. No permutation test — just the raw OR.

Usage:
  python plot_enrichment_vs_cutoff.py \
    --features-dir /path/to/bbbc036_moa_normalized/ \
    --output graphics/enrichment/moa_enrichment/enrichment_vs_cutoff.png
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import fisher_exact

MODEL_LABELS = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls": "DINOv2-CLS",
    "dino_v2_patch": "DINOv2-Patch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet-UT",
    "subcell": "SubCell",
}

MODEL_COLORS = {
    "cellprofiler": "#1f77b4",
    "cloome": "#ff7f0e",
    "dino_v2_cls": "#2ca02c",
    "dino_v2_patch": "#d62728",
    "open_phenom": "#9467bd",
    "resnet": "#8c564b",
    "resnet_untrained": "#7f7f7f",
    "subcell": "#e377c2",
}

CUTOFF_PCTS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def load_features(features_dir):
    """Load per-model normalized parquets."""
    import glob as g
    parquet_files = sorted(g.glob(os.path.join(features_dir, "*_normalized.parquet")))
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


def aggregate_by_compound(df):
    """Mean-aggregate features per compound."""
    feature_cols = [c for c in df.columns if c.startswith("PC") or c.startswith("feature_")]
    if "Metadata_ASSAY_WELL_ROLE" in df.columns:
        df = df[df["Metadata_ASSAY_WELL_ROLE"] == "treated"]
    mask = df["Metadata_broad_sample"].notna() & (df["Metadata_broad_sample"] != "")
    df = df[mask]
    agg = df.groupby("Metadata_broad_sample")[feature_cols].mean().reset_index()
    return agg


def get_odds_ratio(target_hits, background_hits):
    """Odds ratio with Haldane-Anscombe correction."""
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


def compute_ranked_matches(agg_df, full_moa_map, testable_compounds):
    """Compute similarity matrix and ranked match lists (done once per model)."""
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
        order = np.argsort(sims)[::-1]
        match_list = []
        for j in order:
            if j == i:
                continue
            match_list.append(int(full_moa_map.get(compounds[j]) == my_moa))
        ranked_matches[compounds[i]] = match_list

    return ranked_matches


def compute_or_at_cutoff(ranked_matches, cutoff_pct):
    """Compute per-compound OR and hit counts at a given cutoff percentage."""
    odds = []
    hit_counts = []
    hit_fractions = []

    for compound in sorted(ranked_matches.keys()):
        compound_rank = ranked_matches[compound]
        total_hits = int(np.sum(compound_rank))

        if total_hits == 0:
            continue

        cutoff_items = max(2, len(compound_rank) * cutoff_pct // 100)

        if len(compound_rank) < cutoff_items + 2:
            continue

        target_hits = compound_rank[:cutoff_items]
        background_hits = compound_rank[cutoff_items:]

        if len(target_hits) < 2 or len(background_hits) < 2:
            continue

        or_val = get_odds_ratio(target_hits, background_hits)
        odds.append(or_val)

        hits_in_target = int(np.sum(target_hits))
        hit_counts.append(hits_in_target)
        hit_fractions.append(hits_in_target / total_hits if total_hits > 0 else 0.0)

    n_proc = len(odds)
    geom_or = float(np.exp(np.mean(np.log(odds)))) if odds else 0.0
    mean_or = float(np.mean(odds)) if odds else 0.0
    mean_hits = float(np.mean(hit_counts)) if hit_counts else 0.0
    mean_hit_frac = float(np.mean(hit_fractions)) if hit_fractions else 0.0

    return {
        "n_processed": n_proc,
        "geometric_mean_odds": geom_or,
        "mean_odds": mean_or,
        "mean_hits_in_target": mean_hits,
        "mean_hit_fraction": mean_hit_frac,
        "all_odds": odds,
        "all_hit_counts": hit_counts,
        "all_hit_fractions": hit_fractions,
    }


def prepare_model_data(df):
    """Aggregate features and build MoA maps for a single model."""
    agg = aggregate_by_compound(df)

    if "Metadata_moa" not in df.columns:
        return None, None, None

    moa_per_compound = df.groupby("Metadata_broad_sample")["Metadata_moa"].first()
    agg["Metadata_moa"] = agg["Metadata_broad_sample"].map(moa_per_compound)
    agg = agg.dropna(subset=["Metadata_moa"]).reset_index(drop=True)

    full_moa_map = {}
    for _, row in agg.iterrows():
        m = row["Metadata_moa"]
        if pd.notna(m):
            full_moa_map[row["Metadata_broad_sample"]] = m

    moa_counts = agg["Metadata_moa"].value_counts()
    valid_moas = set(moa_counts[moa_counts >= 2].index)
    testable_compounds = set()
    for comp, moa in full_moa_map.items():
        if moa in valid_moas:
            testable_compounds.add(comp)

    return agg, full_moa_map, testable_compounds


def plot_enrichment_vs_cutoff(results, output_path):
    """Line graph: OR and mean MoA hits vs cutoff, one line per model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models = sorted(results.keys())

    # Panel 1: Geometric mean OR
    ax = axes[0]
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        y_vals = [results[model][c]["geometric_mean_odds"] for c in CUTOFF_PCTS]
        ax.plot(CUTOFF_PCTS, y_vals, marker='o', linewidth=2.2, markersize=8,
                label=label, color=color)
    ax.set_xlabel("Cutoff (%)", fontsize=13)
    ax.set_ylabel("Geometric Mean Odds Ratio", fontsize=13)
    ax.set_title("Geometric Mean OR vs Cutoff", fontweight="bold", fontsize=13)
    ax.set_xticks(CUTOFF_PCTS)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: MoA recall (fraction of matches captured in cutoff)
    ax = axes[1]
    for model in models:
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, None)
        y_vals = [results[model][c]["mean_hit_fraction"] for c in CUTOFF_PCTS]
        ax.plot(CUTOFF_PCTS, y_vals, marker='s', linewidth=2.2, markersize=8,
                label=label, color=color)
    # Random baseline: recall = cutoff_pct / 100
    random_y = [c / 100 for c in CUTOFF_PCTS]
    ax.plot(CUTOFF_PCTS, random_y, '--', color='gray', linewidth=1.5,
            alpha=0.7, label='Random')
    ax.set_xlabel("Cutoff (%)", fontsize=13)
    ax.set_ylabel("Mean MoA Recall", fontsize=13)
    ax.set_title("MoA Recall vs Cutoff", fontweight="bold", fontsize=13)
    ax.set_xticks(CUTOFF_PCTS)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("BBBC036 MoA Enrichment vs Similarity Cutoff\n"
                 "(Haldane-Anscombe correction)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BBBC036 MoA enrichment vs cutoff sweep")
    parser.add_argument("--features-dir", required=True,
                        help="Directory with *_normalized.parquet files")
    parser.add_argument("--output", default="graphics/enrichment/moa_enrichment/enrichment_vs_cutoff.png",
                        help="Output plot path")
    parser.add_argument("--results-pkl", default="results/moa_enrichment/bbbc036_moa_cutoff_sweep.pkl",
                        help="Output pickle for raw results")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Which models to analyse (default: all)")
    args = parser.parse_args()

    print("Loading features...")
    all_models = load_features(args.features_dir)

    if args.models:
        selected = [m for m in args.models if m in all_models]
    else:
        selected = sorted(all_models.keys())
    print(f"Models: {selected}")
    print(f"Cutoffs: {CUTOFF_PCTS}%")

    all_results = {}

    for model_name in selected:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        df = all_models[model_name]
        print(f"  Wells: {len(df)}")

        agg, full_moa_map, testable_compounds = prepare_model_data(df)
        if agg is None:
            print(f"  SKIP: no Metadata_moa column")
            continue
        if len(testable_compounds) < 5:
            print(f"  SKIP: too few testable compounds")
            continue

        print(f"  Compounds: {len(agg)}, Testable: {len(testable_compounds)}")

        # Compute ranked matches ONCE
        print(f"  Computing similarity matrix and rankings...")
        ranked_matches = compute_ranked_matches(agg, full_moa_map, testable_compounds)
        print(f"  Ranked matches: {len(ranked_matches)}")

        model_results = {}
        for cutoff_pct in CUTOFF_PCTS:
            print(f"\n  --- Cutoff: {cutoff_pct}% ---")
            result = compute_or_at_cutoff(ranked_matches, cutoff_pct)
            print(f"    Processed: {result['n_processed']}, "
                  f"Geom OR: {result['geometric_mean_odds']:.2f}, "
                  f"Mean OR: {result['mean_odds']:.2f}")
            model_results[cutoff_pct] = result

        all_results[model_name] = model_results

    # Save raw results
    os.makedirs(os.path.dirname(args.results_pkl) or ".", exist_ok=True)
    with open(args.results_pkl, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved: {args.results_pkl}")

    # Plot
    plot_enrichment_vs_cutoff(all_results, args.output)

    # Summary table
    print(f"\n{'Model':<20}", end="")
    for c in CUTOFF_PCTS:
        print(f"  {c}%_OR {c}%_hits", end="")
    print()
    print("=" * (20 + len(CUTOFF_PCTS) * 14))
    for m in sorted(all_results.keys()):
        print(f"{m:<20}", end="")
        for c in CUTOFF_PCTS:
            r = all_results[m][c]
            print(f"  {r['geometric_mean_odds']:>5.2f} {r['mean_hits_in_target']:>5.2f}", end="")
        print()


if __name__ == "__main__":
    main()
