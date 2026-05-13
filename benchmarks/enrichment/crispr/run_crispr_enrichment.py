#!/usr/bin/env python3
"""
CRISPR Enrichment Analysis Script

This script performs enrichment analysis comparing gene feature similarity
to protein interaction databases (StringDB, CORUM, HuMAP, REACTOME, SIGNOR).

Supports two input modes:
  --features-dir  : directory of per-model normalized parquets (new pipeline)
  --features      : legacy pickle file (old pipeline)
"""

import argparse
import glob
import os
import pickle
import time
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import fisher_exact
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qc.cell_count_qc import (
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    make_cell_count_qc_mask,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTEIN_DB_ROOT = PROJECT_ROOT / "data" / "enrichment" / "protein_databases"


def load_parquet_features(
    features_dir,
    mode='no_restriction',
    cell_count_qc=False,
    qc_cell_counts_dir='data/qc/cell_counts',
    qc_cell_count_cutoff=0.05,
    qc_sample_wells=5000,
    qc_seed=42,
    qc_output_dir=None,
    exclude_batches=None,
):
    """Load per-model normalized parquets from a directory.

    Each file should be named ``<model>_normalized.parquet`` and contain
    ``Metadata_Symbol`` plus feature columns (``PC*`` or ``feature_*``).

    Args:
        features_dir: directory containing *_normalized.parquet files
        mode: 'no_restriction' — aggregate all wells per gene (1 embedding)
              'not_same_batch' — aggregate wells per gene per batch (multiple embeddings)

    Returns:
        If mode='no_restriction':
            dict {model_name: gene-aggregated DataFrame}
        If mode='not_same_batch':
            dict {model_name: dict {gene: {batch: np.array embedding}}}
    """
    parquet_files = sorted(glob.glob(os.path.join(features_dir, "*_normalized.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No *_normalized.parquet files found in {features_dir}")

    aggregated = {}
    qc_mask = None
    exclude_batches = set(exclude_batches or [])
    for fpath in parquet_files:
        model_name = os.path.basename(fpath).replace("_normalized.parquet", "")
        df = pd.read_parquet(fpath)
        if cell_count_qc:
            if qc_mask is None:
                qc_mask = make_cell_count_qc_mask(
                    df,
                    dataset_name="cpg-crispr",
                    cell_counts_dir=qc_cell_counts_dir,
                    cutoff=qc_cell_count_cutoff,
                    sample_wells=qc_sample_wells,
                    seed=qc_seed,
                    output_dir=qc_output_dir,
                    label=f"cpg-crispr_{mode}",
                )
            before = len(df)
            df = apply_cell_count_qc_mask(df, qc_mask, "cpg-crispr")
            print(f"  [cell-count QC] {model_name}: wells {before} -> {len(df)}")

        # Detect feature columns
        feature_cols = [c for c in df.columns if c.startswith("PC") or c.startswith("F_") or c.startswith("feature_")]
        if not feature_cols:
            print(f"  WARNING: {model_name} has no PC*/feature_* columns — skipping")
            continue

        # Drop non-targeting and InChI-key entries (not real genes)
        gene_mask = (
            (df["Metadata_Symbol"] != "non-targeting")
            & ~df["Metadata_Symbol"].str.contains("-", na=False)
            & df["Metadata_Symbol"].notna()
        )
        df = df[gene_mask]

        if exclude_batches:
            if 'Metadata_Batch' not in df.columns:
                raise ValueError("--exclude-batches requires a Metadata_Batch column")
            before = len(df)
            df = df[~df["Metadata_Batch"].isin(exclude_batches)]
            removed = before - len(df)
            print(
                f"  [batch exclusion] {model_name}: removed {removed} rows "
                f"from {sorted(exclude_batches)}"
            )

        if mode == 'not_same_batch':
            if 'Metadata_Batch' not in df.columns:
                print(f"  WARNING: {model_name} has no Metadata_Batch column — skipping")
                continue
            # Aggregate per gene per batch
            gene_batch_embeddings = {}
            grouped = df.groupby(['Metadata_Symbol', 'Metadata_Batch'])[feature_cols].mean()
            for (gene, batch), row in grouped.iterrows():
                if gene not in gene_batch_embeddings:
                    gene_batch_embeddings[gene] = {}
                gene_batch_embeddings[gene][batch] = row.values
            n_genes = len(gene_batch_embeddings)
            n_combos = sum(len(v) for v in gene_batch_embeddings.values())
            print(f"  {model_name}: {n_genes} genes, {n_combos} gene-batch combos, {len(feature_cols)} features")
            aggregated[model_name] = gene_batch_embeddings
        else:
            # Aggregate to gene level (mean across wells)
            agg = df.groupby("Metadata_Symbol")[feature_cols].mean().reset_index()
            print(f"  {model_name}: {len(agg)} genes, {len(feature_cols)} features")
            aggregated[model_name] = agg

    return aggregated


def load_pickle_features(features_path):
    """Legacy loader: pickle dict → filter → gene-aggregate."""
    from crispr_enrichment_utils import load_features, filter_features, aggregate_features
    normalized_features = load_features(features_path)
    normalized_features = filter_features(normalized_features)
    return aggregate_features(normalized_features)


def create_lookup_set(df, col1='protein1_symbol', col2='protein2_symbol'):
    """Create a bidirectional lookup set for protein pairs."""
    lookup = set(zip(df[col1], df[col2]))
    lookup.update(set(zip(df[col2], df[col1])))
    return lookup


def get_odds_ratio(target_hits, background_hits):
    """
    Calculate odds ratio for enrichment.
    target_hits and background_hits are binary arrays (1 = hit, 0 = not)
    
    Uses Haldane-Anscombe correction (add 0.5 to all cells) only when at
    least one cell is zero, per Weber et al. 2020 (PMID 32991790).
    """
    table = np.array([
        [np.sum(np.array(target_hits) == 1), np.sum(np.array(target_hits) == 0)],
        [np.sum(np.array(background_hits) == 1), np.sum(np.array(background_hits) == 0)]
    ])
    if np.any(table == 0):
        table = table.astype(float) + 0.5
        odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
    else:
        odds_ratio, _ = fisher_exact(table, alternative='greater')
    return odds_ratio


# ── Vectorized core (replaces per-gene loop + scipy.stats.permutation_test) ──

RANDOM_STATE = 61

def _odds_ratio_batch(target, background):
    """Vectorized odds ratio for a batch of genes.

    Equivalent to calling ``get_odds_ratio`` per row, but ~1000× faster.
    fisher_exact returns (a*d)/(b*c); Haldane-Anscombe adds 0.5 when any cell is 0.

    Args:
        target:     (n_genes, cutoff) int8/bool array
        background: (n_genes, n_bg)   int8/bool array

    Returns:
        (n_genes,) float64 odds ratios
    """
    n_target = target.shape[1]
    n_bg = background.shape[1]
    a = target.sum(axis=1).astype(np.float64)
    b = n_target - a
    c = background.sum(axis=1).astype(np.float64)
    d = n_bg - c
    needs_correction = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a = np.where(needs_correction, a + 0.5, a)
    b = np.where(needs_correction, b + 0.5, b)
    c = np.where(needs_correction, c + 0.5, c)
    d = np.where(needs_correction, d + 0.5, d)
    return (a * d) / (b * c)


def _enrichment_from_hits(hits, unique_gene_list, n_resamples=100, verbose=True,
                          use_exact=False):
    """Vectorized enrichment from a pre-computed hits matrix.

    Two p-value methods:
      permutation (default): Monte-Carlo random target-set resampling, matching
          the cpg-MoA enrichment implementation.
      exact: Hypergeometric survival function — gives exact p-values
          because the odds ratio is a monotone function of the number of target
          hits, which follows a hypergeometric distribution under the null.

    Args:
        hits:             (n_genes, n_neighbors) bool — hits[i,j] is True when
                          the j-th most-similar gene to gene i is in the DB
        unique_gene_list: array-like of gene names (same order as rows of hits)
        n_resamples:      number of permutation resamples (only used when
                          ``use_exact=False``)
        verbose:          show progress bar
        use_exact:        if True use hypergeometric test, else permutation

    Returns:
        dict with same keys as the legacy per-gene implementation
    """
    from scipy.stats import hypergeom

    n_genes, n_neighbors = hits.shape
    cutoff_items = max(1, int(n_neighbors * 0.01))  # top 1% of self-excluded unique genes

    if n_neighbors < cutoff_items + 2:
        return _empty_result()

    # Filter to genes that have at least one hit (same as original)
    gene_hit_counts = hits.sum(axis=1)
    valid_mask = gene_hit_counts > 0
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return _empty_result()

    valid_hits = hits[valid_mask].astype(np.int8)
    valid_indices = np.where(valid_mask)[0]

    if verbose:
        print(f"Total hits: {int(gene_hit_counts[valid_mask].sum())}")

    # Observed odds ratios (vectorised)
    target = valid_hits[:, :cutoff_items]
    background = valid_hits[:, cutoff_items:]
    observed_odds = _odds_ratio_batch(target, background)

    # ── p-values ──
    if use_exact:
        # Exact hypergeometric test.  P(X >= a_obs) where
        # X ~ Hypergeom(N=n_neighbors, K=total_hits_per_gene, n=cutoff_items).
        total_hits = valid_hits.sum(axis=1).astype(int)
        a_observed = target.sum(axis=1).astype(int)
        pvals = hypergeom.sf(a_observed - 1, n_neighbors, total_hits, cutoff_items)
    else:
        # Monte-Carlo random target-set test. Under the null, the number of
        # database hits in the top-k set is hypergeometric with fixed total
        # database hits per query.
        rng = np.random.default_rng(RANDOM_STATE)
        exceed_count = np.zeros(n_valid, dtype=np.int64)
        iterator = (tqdm(range(n_resamples), desc="Permutations")
                    if verbose else range(n_resamples))
        total_hits = valid_hits.sum(axis=1).astype(int)
        for _ in iterator:
            perm_target_hits = rng.hypergeometric(
                ngood=total_hits,
                nbad=n_neighbors - total_hits,
                nsample=cutoff_items,
            )
            perm_odds = _odds_ratio_from_counts(
                perm_target_hits, total_hits, cutoff_items, n_neighbors
            )
            exceed_count += (perm_odds >= observed_odds)
        pvals = (exceed_count + 1) / (n_resamples + 1)

    sig_mask = pvals < 0.05
    significant_genes = [
        unique_gene_list[valid_indices[i]]
        for i in range(n_valid)
        if sig_mask[i]
    ]

    log_odds = np.log(np.maximum(observed_odds, 1e-300))
    return {
        'n_processed': n_valid,
        'n_significant': len(significant_genes),
        'fraction_significant': len(significant_genes) / n_valid if n_valid > 0 else 0,
        'mean_odds': float(np.mean(observed_odds)),
        'mean_log_odds': float(np.mean(log_odds)),
        'geometric_mean_odds': float(np.exp(np.mean(log_odds))),
        'significant_genes': significant_genes,
        'all_odds': observed_odds.tolist(),
        'all_pvals': pvals.tolist(),
        'n_unique_candidates': int(n_neighbors),
        'cutoff_items': int(cutoff_items),
        'cutoff_fraction': 0.01,
        'ranking_unit': 'unique_gene',
        'p_value_method': 'exact_hypergeometric' if use_exact else 'permutation',
        'n_resamples': 0 if use_exact else int(n_resamples),
        'random_state': None if use_exact else RANDOM_STATE,
    }


def _empty_result():
    return {
        'n_processed': 0, 'n_significant': 0, 'fraction_significant': 0,
        'mean_odds': 0, 'mean_log_odds': 0, 'geometric_mean_odds': 0,
        'significant_genes': [], 'all_odds': [], 'all_pvals': [],
        'n_unique_candidates': 0, 'cutoff_items': 0,
        'cutoff_fraction': 0.01, 'ranking_unit': 'unique_gene',
    }


# ── Similarity computation (factored out for reuse across databases) ────────

def _compute_no_restriction_sim(feature_df):
    """Compute sorted similarity indices for ``no_restriction`` mode.

    Returns:
        sorted_sim:      (n_genes, n_genes-1) int array — column-indices sorted
                         by descending cosine similarity (self excluded)
        unique_gene_list: 1-D array of gene symbols
        gene_to_idx:     dict {gene_symbol: row_index}
        n_genes:         int
    """
    feature_cols = [c for c in feature_df.columns
                    if c.startswith('PC') or c.startswith('F_') or c.startswith('feature_')]
    unique_gene_list = feature_df['Metadata_Symbol'].values
    gene_to_idx = {g: i for i, g in enumerate(unique_gene_list)}
    n_genes = len(unique_gene_list)

    cosine_sim = cosine_similarity(feature_df[feature_cols].values)
    sorted_sim = cosine_sim.argsort(axis=1)[:, -2::-1]
    del cosine_sim  # free (n_genes, n_genes) float64
    return sorted_sim, unique_gene_list, gene_to_idx, n_genes


def _normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _prepare_gene_batch_profiles(gene_batch_embeddings):
    unique_gene_list = sorted(gene_batch_embeddings.keys())
    gene_to_idx = {g: i for i, g in enumerate(unique_gene_list)}
    n_genes = len(unique_gene_list)

    all_batches = sorted(set(
        b for embs in gene_batch_embeddings.values() for b in embs
    ))
    n_batches = len(all_batches)

    presence = np.zeros((n_batches, n_genes), dtype=bool)
    profiles_by_batch = []
    for batch_idx, batch in enumerate(all_batches):
        batch_gene_indices = []
        batch_embeddings = []
        for gene in unique_gene_list:
            embedding = gene_batch_embeddings[gene].get(batch)
            if embedding is None:
                continue
            batch_gene_indices.append(gene_to_idx[gene])
            batch_embeddings.append(embedding)

        gene_indices = np.asarray(batch_gene_indices, dtype=np.int32)
        matrix = np.asarray(batch_embeddings, dtype=np.float32)
        matrix = _normalize_rows(matrix)
        presence[batch_idx, gene_indices] = True
        profiles_by_batch.append((batch, gene_indices, matrix))

    return unique_gene_list, gene_to_idx, profiles_by_batch, presence


def _compute_not_same_batch_rankings(gene_batch_embeddings, cutoff_fraction=0.01,
                                     cutoff_items_override=None, verbose=True):
    """Rank unique genes for each gene-batch query against other batches only.

    This mirrors the cpg-MoA batchwise paradigm: each query is one batchwise
    profile, comparison rows come from other batches, repeated comparison genes
    are collapsed by their best cross-batch similarity, and significance is
    aggregated back to genes later by majority vote.
    """
    unique_gene_list, gene_to_idx, profiles_by_batch, presence = (
        _prepare_gene_batch_profiles(gene_batch_embeddings)
    )
    n_genes = len(unique_gene_list)
    n_batches = len(profiles_by_batch)
    if cutoff_items_override is None:
        cutoff_items = max(1, int((n_genes - 1) * cutoff_fraction))
    else:
        cutoff_items = max(1, int(cutoff_items_override))

    if verbose:
        genes_per_batch = presence.sum(axis=1).astype(int).tolist()
        print(f"  Batches: {n_batches}, genes per batch: {genes_per_batch}")
        print(f"  Batchwise query cutoff: {cutoff_items} unique genes")

    if n_genes < cutoff_items + 2:
        empty = np.array([], dtype=np.int32)
        return {
            'top_gene_idx': np.empty((0, cutoff_items), dtype=np.int32),
            'query_gene_idx': empty,
            'query_batch_idx': empty,
            'candidate_counts': empty,
            'candidate_masks_by_batch': np.zeros((n_batches, n_genes), dtype=bool),
            'gene_list': unique_gene_list,
            'gene_to_idx': gene_to_idx,
            'cutoff_items': cutoff_items,
            'n_unique_candidates_global': max(0, n_genes - 1),
        }

    top_parts = []
    query_gene_parts = []
    query_batch_parts = []
    candidate_count_parts = []
    candidate_masks_by_batch = np.zeros((n_batches, n_genes), dtype=bool)

    for query_batch_idx, (batch, query_gene_idx, query_matrix) in enumerate(profiles_by_batch):
        other_batch_mask = np.ones(n_batches, dtype=bool)
        other_batch_mask[query_batch_idx] = False
        candidate_mask = presence[other_batch_mask].any(axis=0)
        candidate_masks_by_batch[query_batch_idx] = candidate_mask

        best_sim = np.full((len(query_gene_idx), n_genes), -np.inf, dtype=np.float32)
        for comp_batch_idx, (_, comp_gene_idx, comp_matrix) in enumerate(profiles_by_batch):
            if comp_batch_idx == query_batch_idx:
                continue
            sim = query_matrix @ comp_matrix.T
            best_sim[:, comp_gene_idx] = np.maximum(best_sim[:, comp_gene_idx], sim)

        rows = np.arange(len(query_gene_idx))
        best_sim[rows, query_gene_idx] = -np.inf
        candidate_counts = np.isfinite(best_sim).sum(axis=1).astype(np.int32)
        valid_mask = candidate_counts >= cutoff_items + 1

        if not np.any(valid_mask):
            continue

        valid_best_sim = best_sim[valid_mask]
        top_unsorted = np.argpartition(
            -valid_best_sim, kth=cutoff_items - 1, axis=1
        )[:, :cutoff_items]
        top_scores = np.take_along_axis(valid_best_sim, top_unsorted, axis=1)
        top_order = np.argsort(-top_scores, axis=1)
        top_sorted = np.take_along_axis(top_unsorted, top_order, axis=1).astype(np.int32)

        valid_query_gene_idx = query_gene_idx[valid_mask]
        if np.any(top_sorted == valid_query_gene_idx[:, None]):
            raise AssertionError("Query gene found in not_same_batch enrichment target")

        top_parts.append(top_sorted)
        query_gene_parts.append(valid_query_gene_idx.astype(np.int32))
        query_batch_parts.append(
            np.full(valid_query_gene_idx.shape, query_batch_idx, dtype=np.int32)
        )
        candidate_count_parts.append(candidate_counts[valid_mask])
        del best_sim, valid_best_sim, top_unsorted, top_scores, top_order

    if top_parts:
        top_gene_idx = np.vstack(top_parts)
        query_gene_idx = np.concatenate(query_gene_parts)
        query_batch_idx = np.concatenate(query_batch_parts)
        candidate_counts = np.concatenate(candidate_count_parts)
    else:
        top_gene_idx = np.empty((0, cutoff_items), dtype=np.int32)
        query_gene_idx = np.array([], dtype=np.int32)
        query_batch_idx = np.array([], dtype=np.int32)
        candidate_counts = np.array([], dtype=np.int32)

    return {
        'top_gene_idx': top_gene_idx,
        'query_gene_idx': query_gene_idx,
        'query_batch_idx': query_batch_idx,
        'candidate_counts': candidate_counts,
        'candidate_masks_by_batch': candidate_masks_by_batch,
        'gene_list': unique_gene_list,
        'gene_to_idx': gene_to_idx,
        'cutoff_items': cutoff_items,
        'n_unique_candidates_global': max(0, n_genes - 1),
    }


def _odds_ratio_from_counts(target_hits, total_hits, cutoff_items, candidate_counts):
    a = target_hits.astype(np.float64)
    b = cutoff_items - a
    c = (total_hits - target_hits).astype(np.float64)
    d = (candidate_counts - cutoff_items - c).astype(np.float64)
    needs_correction = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a = np.where(needs_correction, a + 0.5, a)
    b = np.where(needs_correction, b + 0.5, b)
    c = np.where(needs_correction, c + 0.5, c)
    d = np.where(needs_correction, d + 0.5, d)
    return (a * d) / (b * c)


def _enrichment_from_batchwise_rankings(
    rankings,
    db_lookup,
    n_resamples=100,
    verbose=True,
    use_exact=False,
):
    """Compute enrichment for gene-batch queries and majority-vote by gene."""
    from scipy.stats import hypergeom

    gene_list = rankings['gene_list']
    gene_to_idx = rankings['gene_to_idx']
    n_genes = len(gene_list)
    top_gene_idx = rankings['top_gene_idx']
    query_gene_idx = rankings['query_gene_idx']
    query_batch_idx = rankings['query_batch_idx']
    candidate_counts = rankings['candidate_counts']
    cutoff_items = rankings['cutoff_items']

    if len(query_gene_idx) == 0:
        return _empty_result()

    db_adj = _build_adjacency(db_lookup, gene_to_idx, n_genes)
    target_hits = db_adj[query_gene_idx[:, None], top_gene_idx].sum(axis=1).astype(np.int32)

    total_hits = np.zeros(len(query_gene_idx), dtype=np.int32)
    candidate_masks_by_batch = rankings['candidate_masks_by_batch']
    for batch_idx in np.unique(query_batch_idx):
        rows = np.where(query_batch_idx == batch_idx)[0]
        batch_candidate_mask = candidate_masks_by_batch[batch_idx]
        row_gene_idx = query_gene_idx[rows]
        total_hits[rows] = db_adj[row_gene_idx][:, batch_candidate_mask].sum(axis=1)
        self_in_candidate_mask = batch_candidate_mask[row_gene_idx]
        if np.any(self_in_candidate_mask):
            total_hits[rows] -= (
                db_adj[row_gene_idx, row_gene_idx] & self_in_candidate_mask
            ).astype(np.int32)

    valid_mask = total_hits > 0
    n_valid_queries = int(valid_mask.sum())
    if n_valid_queries == 0:
        return _empty_result()

    target_hits = target_hits[valid_mask]
    total_hits = total_hits[valid_mask]
    candidate_counts = candidate_counts[valid_mask]
    valid_query_gene_idx = query_gene_idx[valid_mask]

    if verbose:
        print(f"Total hits: {int(total_hits.sum())}")
        print(f"Query profiles with hits: {n_valid_queries}")

    observed_odds = _odds_ratio_from_counts(
        target_hits, total_hits, cutoff_items, candidate_counts
    )
    if use_exact:
        pvals = hypergeom.sf(target_hits - 1, candidate_counts, total_hits, cutoff_items)
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        exceed_count = np.zeros(n_valid_queries, dtype=np.int64)
        iterator = (
            tqdm(range(n_resamples), desc="Permutations")
            if verbose else range(n_resamples)
        )
        for _ in iterator:
            perm_target_hits = rng.hypergeometric(
                ngood=total_hits,
                nbad=candidate_counts - total_hits,
                nsample=cutoff_items,
            )
            perm_odds = _odds_ratio_from_counts(
                perm_target_hits, total_hits, cutoff_items, candidate_counts
            )
            exceed_count += (perm_odds >= observed_odds)
        pvals = (exceed_count + 1) / (n_resamples + 1)
    sig_mask = pvals < 0.05

    gene_total = Counter()
    gene_sig = Counter()
    for gene_idx, is_sig in zip(valid_query_gene_idx, sig_mask):
        gene = gene_list[gene_idx]
        gene_total[gene] += 1
        if is_sig:
            gene_sig[gene] += 1

    significant_genes = [
        gene for gene, total in gene_total.items()
        if gene_sig[gene] / total > 0.5
    ]

    log_odds = np.log(np.maximum(observed_odds, 1e-300))
    n_unique_processed = len(gene_total)
    unique_candidate_counts = sorted(set(int(v) for v in candidate_counts.tolist()))
    return {
        'n_processed': n_unique_processed,
        'n_query_profiles_processed': n_valid_queries,
        'n_significant': len(significant_genes),
        'fraction_significant': len(significant_genes) / n_unique_processed if n_unique_processed > 0 else 0,
        'mean_odds': float(np.mean(observed_odds)),
        'mean_log_odds': float(np.mean(log_odds)),
        'geometric_mean_odds': float(np.exp(np.mean(log_odds))),
        'significant_genes': significant_genes,
        'all_odds': observed_odds.tolist(),
        'all_pvals': pvals.tolist(),
        'n_unique_candidates': int(rankings['n_unique_candidates_global']),
        'unique_candidate_counts': unique_candidate_counts,
        'cutoff_items': int(cutoff_items),
        'cutoff_fraction': 0.01,
        'cutoff_items_override': int(cutoff_items),
        'ranking_unit': 'gene_batch_query_unique_gene',
        'aggregation': 'majority_vote_by_gene',
        'p_value_method': 'exact_hypergeometric' if use_exact else 'permutation',
        'n_resamples': 0 if use_exact else int(n_resamples),
        'random_state': None if use_exact else RANDOM_STATE,
    }


def _build_adjacency(db_lookup, gene_to_idx, n_genes):
    """Build boolean adjacency matrix from a database lookup set."""
    db_adj = np.zeros((n_genes, n_genes), dtype=bool)
    for gene_a, gene_b in db_lookup:
        if gene_a in gene_to_idx and gene_b in gene_to_idx:
            db_adj[gene_to_idx[gene_a], gene_to_idx[gene_b]] = True
    return db_adj


# ── Public API (kept for backward compatibility with external callers) ──────

def run_enrichment_for_model(feature_df, db_lookup, n_resamples=100, verbose=True,
                             use_exact=False):
    """Run enrichment for one model × one database (no_restriction mode)."""
    sorted_sim, gene_list, gene_to_idx, n = _compute_no_restriction_sim(feature_df)
    db_adj = _build_adjacency(db_lookup, gene_to_idx, n)
    hits = db_adj[np.arange(n)[:, None], sorted_sim]
    return _enrichment_from_hits(hits, gene_list, n_resamples, verbose, use_exact=use_exact)


def run_enrichment_not_same_batch(gene_batch_embeddings, db_lookup,
                                n_resamples=100, verbose=True, use_exact=False):
    """Run enrichment for one model × one database (batchwise cross-batch mode)."""
    rankings = _compute_not_same_batch_rankings(gene_batch_embeddings, verbose=verbose)
    return _enrichment_from_batchwise_rankings(
        rankings,
        db_lookup,
        n_resamples=n_resamples,
        verbose=verbose,
        use_exact=use_exact,
    )


def main():
    parser = argparse.ArgumentParser(description='Run CRISPR enrichment analysis')

    # Input — new parquet dir mode (preferred) or legacy pickle
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--features-dir', type=str,
                             help='Directory containing <model>_normalized.parquet files')
    input_group.add_argument('--features', type=str,
                             help='(Legacy) Path to features pickle file')

    parser.add_argument('--models', type=str, nargs='*', default=None,
                        help='Which models to analyse (default: all found in features dir)')
    parser.add_argument('--stringdb', type=str,
                        default=str(PROTEIN_DB_ROOT / "StringDB.parquet"),
                        help='Path to StringDB parquet file')
    parser.add_argument('--corum', type=str,
                        default=str(PROTEIN_DB_ROOT / "CORUM.parquet"),
                        help='Path to CORUM parquet file')
    parser.add_argument('--humap', type=str,
                        default=str(PROTEIN_DB_ROOT / "HuMAP.parquet"),
                        help='Path to HuMAP parquet file')
    parser.add_argument('--reactome', type=str,
                        default=str(PROTEIN_DB_ROOT / "REACTOME.parquet"),
                        help='Path to REACTOME parquet file')
    parser.add_argument('--signor', type=str,
                        default=str(PROTEIN_DB_ROOT / "SIGNOR.parquet"),
                        help='Path to SIGNOR parquet file')
    parser.add_argument('--n_resamples', type=int, default=100,
                        help='Number of permutation resamples')
    parser.add_argument('--mode', type=str, default='no_restriction',
                        choices=['no_restriction', 'not_same_batch'],
                        help='Enrichment mode: no_restriction (global gene embeddings) '
                             'or not_same_batch (per-batch embeddings, cross-batch similarity)')
    parser.add_argument('--output', type=str, default='crispr_enrichment_results.pkl',
                        help='Output pickle file for results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bars')
    parser.add_argument('--exact-hypergeom', action='store_true',
                        help='Use exact hypergeometric p-values instead of the '
                             'default Monte-Carlo permutation test')
    parser.add_argument('--permutation', action='store_true',
                        help='Deprecated no-op: permutation testing is now the default')
    parser.add_argument('--exclude-batches', nargs='*', default=[],
                        help='Metadata_Batch values to exclude before aggregation')
    add_cell_count_qc_args(parser)
    
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features (mode={args.mode})...")
    if args.features_dir:
        aggregated_features = load_parquet_features(
            args.features_dir,
            mode=args.mode,
            cell_count_qc=args.cell_count_qc,
            qc_cell_counts_dir=args.qc_cell_counts_dir,
            qc_cell_count_cutoff=args.qc_cell_count_cutoff,
            qc_sample_wells=args.qc_sample_wells,
            qc_seed=args.qc_seed,
            qc_output_dir=os.path.join(os.path.dirname(args.output) or ".", "cell_count_qc"),
            exclude_batches=args.exclude_batches,
        )
    else:
        if args.mode == 'not_same_batch':
            raise ValueError("not_same_batch mode requires --features-dir (not legacy pickle)")
        aggregated_features = load_pickle_features(args.features)
    
    # Filter to requested models (or all)
    if args.models:
        available_models = [m for m in args.models if m in aggregated_features]
        missing = set(args.models) - set(available_models)
        if missing:
            print(f"WARNING: models not found: {missing}")
    else:
        available_models = list(aggregated_features.keys())
    print(f"Models to analyze: {available_models}")
    
    # Load all databases
    print("\nLoading databases...")
    
    # StringDB (filter for score > 950)
    print("  Loading StringDB...")
    stringdb_df = pd.read_parquet(args.stringdb)
    stringdb_high_conf = stringdb_df[stringdb_df['combined_score'] > 950]
    
    # CORUM
    print("  Loading CORUM...")
    corum_df = pd.read_parquet(args.corum)
    
    # HuMAP
    print("  Loading HuMAP...")
    humap_df = pd.read_parquet(args.humap)
    
    # REACTOME
    print("  Loading REACTOME...")
    reactome_df = pd.read_parquet(args.reactome)
    
    # SIGNOR
    print("  Loading SIGNOR...")
    signor_df = pd.read_parquet(args.signor)
    
    # Create lookup sets
    protein_databases = {
        'CORUM': create_lookup_set(corum_df),
        'HuMAP': create_lookup_set(humap_df),
        'REACTOME': create_lookup_set(reactome_df),
        'SIGNOR': create_lookup_set(signor_df, col1='protein1', col2='protein2'),
        'StringDB': create_lookup_set(stringdb_high_conf),
    }
    
    for db_name, lookup in protein_databases.items():
        print(f"  {db_name} lookup has {len(lookup)} unique pairs")
    
    # Run enrichment analysis for each model and database
    # Similarity is computed once per model, then reused across all databases.
    use_exact = args.exact_hypergeom
    pval_method = "exact (hypergeometric)" if use_exact else "permutation"
    print(f"\nRunning enrichment analysis (mode={args.mode}, p-value={pval_method}, n_resamples={args.n_resamples})...")
    
    all_results = {}
    
    for model_name in available_models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        t0 = time.time()
        if args.mode == 'not_same_batch':
            rankings = _compute_not_same_batch_rankings(
                aggregated_features[model_name], verbose=not args.quiet
            )
            gene_list = rankings['gene_list']
            gene_to_idx = rankings['gene_to_idx']
            n_genes = len(gene_list)
        else:
            feature_df = aggregated_features[model_name].copy()
            sorted_idx, gene_list, gene_to_idx, n_genes = _compute_no_restriction_sim(feature_df)
        print(f"  Similarity computed in {time.time() - t0:.1f}s ({n_genes} genes)")

        model_results = {}
        for db_name, db_lookup in protein_databases.items():
            print(f"\n  Database: {db_name}")

            t1 = time.time()
            if args.mode == 'not_same_batch':
                result = _enrichment_from_batchwise_rankings(
                    rankings, db_lookup, n_resamples=args.n_resamples,
                    verbose=not args.quiet, use_exact=use_exact
                )
            else:
                db_adj = _build_adjacency(db_lookup, gene_to_idx, n_genes)
                hits = db_adj[np.arange(n_genes)[:, None], sorted_idx]
                result = _enrichment_from_hits(
                    hits, gene_list, n_resamples=args.n_resamples,
                    verbose=not args.quiet, use_exact=use_exact
                )
            result['excluded_batches'] = list(args.exclude_batches)
            elapsed = time.time() - t1
            
            print(f"    Processed {result['n_processed']} genes")
            print(f"    Significant genes (p < 0.05): {result['n_significant']}")
            print(f"    Fraction significant: {result['fraction_significant']:.4f}")
            print(f"    Mean odds ratio: {result['mean_odds']:.4f}")
            print(f"    Geometric mean odds ratio: {result['geometric_mean_odds']:.4f}")
            print(f"    Elapsed: {elapsed:.1f}s")
            
            model_results[db_name] = result
        
        all_results[model_name] = model_results
    
    # Print summary
    print("\n" + "="*80)
    print("ENRICHMENT ANALYSIS SUMMARY")
    print("="*80)
    
    # Create summary dataframe
    summary_rows = []
    for model_name, model_results in all_results.items():
        for db_name, result in model_results.items():
            sorted_odds = sorted(result['all_odds'], reverse=True) if result['all_odds'] else []
            summary_rows.append({
                'Model': model_name,
                'Database': db_name,
                'N Processed': result['n_processed'],
                'N Significant': result['n_significant'],
                'Fraction Significant': result['fraction_significant'],
                'Mean Odds Ratio': result['mean_odds'],
                'Geometric Mean OR': result['geometric_mean_odds'],
                'Sorted Odds Ratios': sorted_odds
            })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    
    print("Done!")


if __name__ == '__main__':
    main()
