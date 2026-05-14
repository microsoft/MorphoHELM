#!/usr/bin/env python3
"""Run BBBC036 MoA enrichment with metadata join for a given pipeline."""
import argparse, os, sys, pickle, glob
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import permutation_test

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from qc.cell_count_qc import (
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    make_cell_count_qc_mask,
)

DEFAULT_BBBC_META = str(
    PROJECT_ROOT
    / "data"
    / "main_paper_inputs"
    / "metadata"
    / "bbbc036_metadata.parquet"
)
RANDOM_STATE = 61


def get_odds_ratio(target_hits, background_hits):
    """Haldane-Anscombe corrected odds ratio for binary hit arrays."""
    target_hits = np.asarray(target_hits, dtype=np.int8)
    background_hits = np.asarray(background_hits, dtype=np.int8)
    a = float(target_hits.sum())
    b = float(len(target_hits) - a)
    c = float(background_hits.sum())
    d = float(len(background_hits) - c)
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    return (a * d) / (b * c)


def permutation_pvalue(target_hits, background_hits, n_resamples):
    """Permutation p-value matching the paper-era BBBC036 MoA benchmark."""
    target_hits = np.asarray(target_hits, dtype=np.int8)
    background_hits = np.asarray(background_hits, dtype=np.int8)
    result = permutation_test(
        (target_hits, background_hits),
        statistic=get_odds_ratio,
        n_resamples=n_resamples,
        alternative="greater",
        random_state=RANDOM_STATE,
    )
    return result.statistic, result.pvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default=DEFAULT_BBBC_META,
                        help="BBBC036 metadata parquet with plate, well, sample, and MoA columns")
    parser.add_argument("--n-resamples", "--n_resamples", dest="n_resamples",
                        type=int, default=100,
                        help="Monte Carlo resamples for permutation-style p-values")
    add_cell_count_qc_args(parser)
    args = parser.parse_args()
    
    # Load bbbc036 metadata
    meta = pd.read_parquet(args.metadata, columns=["Metadata_Plate", "Metadata_Well", "Metadata_broad_sample", "Metadata_moa"])
    meta = meta.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"])
    meta["Metadata_Plate"] = meta["Metadata_Plate"].astype(str)
    
    # Import enrichment functions
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "enrichment", "moa"))
    pass  # enrichment logic is inline below
    
    parquets = sorted(glob.glob(os.path.join(args.features_dir, "*_normalized.parquet")))
    results = {}
    qc_mask = None
    
    for fpath in parquets:
        model = os.path.basename(fpath).replace("_normalized.parquet", "")
        print(f"  {model}...", flush=True)
        
        df = pd.read_parquet(fpath)
        df["Metadata_Plate"] = df["Metadata_Plate"].astype(str)
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
            print(f"    Cell-count QC: {before} -> {len(df)} wells")
        
        # Join metadata
        df = df.merge(meta, on=["Metadata_Plate", "Metadata_Well"], how="left")
        
        feat_cols = [c for c in df.columns if c.startswith("PC") or c.startswith("F_")]
        if not feat_cols:
            feat_cols = [c for c in df.columns if not c.startswith("Metadata")]
        
        # Filter to treated compounds with MoA
        mask = (df["Metadata_broad_sample"].notna() & 
                (df["Metadata_broad_sample"] != "") &
                df["Metadata_moa"].notna() & 
                (df["Metadata_moa"] != ""))
        df_filt = df[mask].copy()
        
        if len(df_filt) < 100:
            print(f"    Too few samples ({len(df_filt)}), skipping")
            continue
        
        # Aggregate to compound level
        agg = df_filt.groupby("Metadata_broad_sample")[feat_cols].mean().reset_index()
        moa_map = df_filt.drop_duplicates("Metadata_broad_sample").set_index("Metadata_broad_sample")["Metadata_moa"]
        agg["Metadata_moa"] = agg["Metadata_broad_sample"].map(moa_map)
        
        # Keep every MoA-labeled compound in the ranked candidate pool, but only
        # query compounds whose MoA has at least one possible match. This matches
        # the paper-era BBBC036 enrichment implementation.
        moa_counts = agg["Metadata_moa"].value_counts()
        valid_moas = set(moa_counts[moa_counts >= 2].index)
        testable_mask = agg["Metadata_moa"].isin(valid_moas).to_numpy()
        n_testable = int(testable_mask.sum())
        
        if n_testable < 5:
            print(f"    Too few compounds after filtering ({len(agg)})")
            continue
        
        # Enrichment
        from sklearn.metrics.pairwise import cosine_similarity

        features = agg[feat_cols].values
        sim = cosine_similarity(features)
        np.fill_diagonal(sim, -1)
        
        n = len(agg)
        moas = agg["Metadata_moa"].values
        
        odds_ratios = []
        p_values = []
        processed_compounds = []
        significant_compounds = []
        cutoff = None
        
        for i in range(n):
            if not testable_mask[i]:
                continue
            ranked_idx = [j for j in np.argsort(-sim[i]) if j != i]
            cutoff = max(2, len(ranked_idx) // 100)
            if len(ranked_idx) < cutoff + 2:
                continue
            top_idx = ranked_idx[:cutoff]
            background_idx = ranked_idx[cutoff:]
            if i in top_idx:
                raise AssertionError("Query compound found in BBBC036 enrichment target")
            if len(top_idx) != len(set(top_idx)):
                raise AssertionError("Duplicate compound found in BBBC036 enrichment target")
            target = np.asarray([moas[j] == moas[i] for j in top_idx], dtype=np.int8)
            background = np.asarray([moas[j] == moas[i] for j in background_idx], dtype=np.int8)
            if target.sum() + background.sum() == 0:
                continue
            odds, pval = permutation_pvalue(target, background, args.n_resamples)
            odds_ratios.append(odds)
            p_values.append(pval)
            compound = agg.iloc[i]["Metadata_broad_sample"]
            processed_compounds.append(compound)
            if pval < 0.05:
                significant_compounds.append(compound)
        
        if not p_values:
            print("    No eligible compounds after enrichment filtering, skipping")
            continue
        frac_sig = np.mean(np.array(p_values) < 0.05)
        geom_or = np.exp(np.mean(np.log(np.array(odds_ratios))))
        
        results[model] = {
            "fraction_significant": frac_sig,
            "geometric_mean_or": geom_or,
            "all_odds": odds_ratios,
            "all_pvals": p_values,
            "processed_compounds": processed_compounds,
            "significant_compounds": significant_compounds,
            "n_compounds": len(agg),
            "n_processed": len(p_values),
            "n_moas": len(valid_moas),
            "n_unique_candidates": len(agg) - 1,
            "cutoff_items": cutoff,
            "cutoff_fraction": 0.01,
            "ranking_unit": "unique_compound",
            "p_value_method": "permutation",
            "n_resamples": args.n_resamples,
            "random_state": RANDOM_STATE,
        }
        print(f"    frac_sig={frac_sig:.3f}, geom_or={geom_or:.2f}, n={len(agg)}")
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
