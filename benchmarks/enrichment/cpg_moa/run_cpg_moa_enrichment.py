"""
CPJump + BBBC036 Combined MOA Enrichment Heatmap

Runs CPJump MOA enrichment across 4 paradigms (Global, Within Source, Not Same Batch,
Not Same Source) using Haldane-Anscombe corrected OR and permutation tests, then combines
with pre-computed BBBC036 results into a single heatmap showing:
  - Fraction Significant (p < 0.05)
  - Geometric Mean OR

All 8 models shown as rows; CPJump-missing models get gray cells.
"""

import pickle
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from scipy.stats import fisher_exact, permutation_test
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[3]
DATA_PATH = str(BASE / "data" / "moa_not_same_source_profiles.pkl")
BBBC_RESULTS_PATH = str(BASE / "results" / "moa_enrichment" / "bbbc036_moa_with_pca.pkl")
OUTPUT_PKL = str(BASE / "results" / "enrichment" / "cpjump_enrichment_results_haldane.pkl")
OUTPUT_PNG = str(BASE / "results" / "enrichment" / "cpjump_enrichment_heatmap.png")

# ── Config ───────────────────────────────────────────────────────────────
CUTOFF_FRACTION = 0.01
N_RESAMPLES = 100
RANDOM_STATE = 61
MAX_BATCHES = None
MAX_SOURCES = None

# Canonical model order (all 8)
ALL_MODELS = [
    "cellprofiler", "cloome", "dino_v2_cls", "dino_v2_patch",
    "open_phenom", "resnet", "resnet_untrained", "subcell",
]

MODEL_LABELS = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls": "DINOv2 CLS",
    "dino_v2_patch": "DINOv2 Patch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet (Untrained)",
    "subcell": "SubCell",
}

# CPJump model names → canonical names
CPJUMP_TO_CANONICAL = {
    "openphenom": "open_phenom",
    "cellprofiler": "cellprofiler",
    "dino_v2": "dino_v2_cls",
    "dino_v2_patch": "dino_v2_patch",
    "resnet": "resnet",
    "resnet_untrained": "resnet_untrained",
    "cloome": "cloome",
    "subcell": "subcell",
}


# ── OR Function ──────────────────────────────────────────────────────────
def get_odds_ratio(target_hits, background_hits):
    """Haldane-Anscombe corrected odds ratio."""
    table = np.array([
        [np.sum(np.array(target_hits) == 1),
         np.sum(np.array(target_hits) == 0)],
        [np.sum(np.array(background_hits) == 1),
         np.sum(np.array(background_hits) == 0)]
    ])
    if np.any(table == 0):
        table = table.astype(float) + 0.5
        return (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
    else:
        odds_ratio, _ = fisher_exact(table, alternative='greater')
        return odds_ratio


def permutation_pvalue(target_hits, background_hits):
    """Permutation p-value for the observed odds ratio."""
    if len(target_hits) >= 2 and len(background_hits) >= 2:
        result = permutation_test(
            (target_hits, background_hits),
            statistic=get_odds_ratio,
            n_resamples=N_RESAMPLES,
            alternative='greater',
            random_state=RANDOM_STATE,
        )
        return result.statistic, result.pvalue

    observed = get_odds_ratio(target_hits, background_hits)
    combined = np.array(list(target_hits) + list(background_hits), dtype=np.int8)
    target_size = len(target_hits)
    rng = np.random.default_rng(RANDOM_STATE)
    exceed_count = 0
    for _ in range(N_RESAMPLES):
        permuted = rng.permutation(combined)
        perm_odds = get_odds_ratio(permuted[:target_size], permuted[target_size:])
        exceed_count += perm_odds >= observed
    return observed, (exceed_count + 1) / (N_RESAMPLES + 1)


# ── Helpers ──────────────────────────────────────────────────────────────
def aggregate_by_compound(df, pc_cols):
    """Aggregate PC features by compound, keeping MoA."""
    agg = df.groupby('Metadata_JCP2022')[pc_cols].mean().reset_index()
    moa_map = df.drop_duplicates('Metadata_JCP2022').set_index('Metadata_JCP2022')['Metadata_MoA']
    agg['Metadata_MoA'] = agg['Metadata_JCP2022'].map(moa_map)
    return agg


def run_enrichment(query_df, comparison_df=None, pc_cols=None, cutoff_fraction=CUTOFF_FRACTION,
                   cutoff_items_override=None):
    """Run enrichment with Haldane-Anscombe OR + permutation test."""
    if comparison_df is None:
        comparison_df = query_df

    query_features = query_df[pc_cols].values
    comp_features = comparison_df[pc_cols].values
    query_compounds = query_df['Metadata_JCP2022'].values
    query_moas = query_df['Metadata_MoA'].values
    comp_compounds = comparison_df['Metadata_JCP2022'].values
    comp_moas = comparison_df['Metadata_MoA'].values
    unique_compounds, comp_codes = np.unique(comp_compounds, return_inverse=True)
    sort_idx = np.argsort(comp_codes, kind="stable")
    sorted_codes = comp_codes[sort_idx]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    unique_order = sorted_codes[starts]
    unique_compounds = unique_compounds[unique_order]
    unique_moas = comp_moas[sort_idx[starts]]

    odds, pvals = [], []
    processed_comp, significant_comp = [], []
    cutoff_items, unique_candidate_counts = [], []

    chunk_size = 512
    for chunk_start in range(0, len(query_df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(query_df))
        sim_matrix = cosine_similarity(query_features[chunk_start:chunk_end], comp_features)
        max_by_compound = np.maximum.reduceat(sim_matrix[:, sort_idx], starts, axis=1)

        for local_i, i in enumerate(range(chunk_start, chunk_end)):
            query_moa = query_moas[i]
            query_compound = query_compounds[i]
            candidate_mask = unique_compounds != query_compound
            candidate_scores = max_by_compound[local_i, candidate_mask]
            candidate_moas = unique_moas[candidate_mask]
            candidate_compounds = unique_compounds[candidate_mask]

            n_unique_candidates = len(candidate_compounds)
            if cutoff_items_override is None:
                cutoff = max(1, int(n_unique_candidates * cutoff_fraction))
            else:
                cutoff = max(1, int(cutoff_items_override))
            if n_unique_candidates < cutoff + 1:
                continue

            ranked_idx = np.argsort(candidate_scores)[::-1]
            moa_matches = (candidate_moas[ranked_idx] == query_moa).astype(np.int8)
            if int(moa_matches.sum()) == 0:
                continue

            target = moa_matches[:cutoff]
            background = moa_matches[cutoff:]
            target_compounds = candidate_compounds[ranked_idx[:cutoff]]
            if query_compound in target_compounds:
                raise AssertionError("Query compound found in enrichment target")
            if len(target_compounds) != len(set(target_compounds)):
                raise AssertionError("Duplicate compound found in enrichment target")

            odds_ratio, pvalue = permutation_pvalue(target, background)
            odds.append(odds_ratio)
            pvals.append(pvalue)
            processed_comp.append(query_compound)
            cutoff_items.append(cutoff)
            unique_candidate_counts.append(n_unique_candidates)
            if pvalue < 0.05:
                significant_comp.append(query_compound)

    n_processed = len(odds)
    log_odds = [np.log(o) for o in odds if o > 0]
    return {
        'odds_ratios': odds,
        'p_values': pvals,
        'processed_compounds': processed_comp,
        'significant_compounds': significant_comp,
        'n_processed': n_processed,
        'n_significant': len(significant_comp),
        'fraction_significant': len(significant_comp) / n_processed if n_processed > 0 else 0,
        'mean_odds_ratio': np.mean(odds) if odds else 0,
        'geometric_mean_or': np.exp(np.mean(log_odds)) if log_odds else 0,
        'cutoff_fraction': cutoff_fraction,
        'cutoff_items': cutoff_items,
        'cutoff_items_override': cutoff_items_override,
        'unique_candidate_counts': unique_candidate_counts,
        'ranking_unit': 'unique_compound',
    }


def compute_aggregate(results_by_key):
    """Aggregate across sources/batches."""
    all_odds, all_significant = [], []
    for res in results_by_key.values():
        all_odds.extend(res['odds_ratios'])
        all_significant.extend(res['significant_compounds'])
    n = len(all_odds)
    log_odds = [np.log(o) for o in all_odds if o > 0]
    return {
        'n_processed': n,
        'n_significant': len(all_significant),
        'fraction_significant': len(all_significant) / n if n > 0 else 0,
        'mean_odds_ratio': np.mean(all_odds) if all_odds else 0,
        'geometric_mean_or': np.exp(np.mean(log_odds)) if log_odds else 0,
    }


def compute_aggregate_majority_vote(results_by_key):
    """Majority vote aggregation for cross-batch/cross-source."""
    compound_total = Counter()
    compound_sig = Counter()
    all_odds = []

    for res in results_by_key.values():
        for comp in res['processed_compounds']:
            compound_total[comp] += 1
        for comp in res['significant_compounds']:
            compound_sig[comp] += 1
        all_odds.extend(res['odds_ratios'])

    n_unique = len(compound_total)
    enriched = [comp for comp, total in compound_total.items()
                if compound_sig[comp] / total > 0.5]
    log_odds = [np.log(o) for o in all_odds if o > 0]

    return {
        'n_unique_compounds': n_unique,
        'n_significant': len(enriched),
        'fraction_significant': len(enriched) / n_unique if n_unique > 0 else 0,
        'mean_odds_ratio': np.mean(all_odds) if all_odds else 0,
        'geometric_mean_or': np.exp(np.mean(log_odds)) if log_odds else 0,
    }


# ── Subset Filtering (same as notebook) ─────────────────────────────────
def apply_subset_filter(moa_profiles):
    """Density optimization: drop sparse sources/MoAs until source_10 is dropped."""
    ref_df = list(moa_profiles.values())[0]
    pivoted = pd.pivot_table(ref_df, index='Metadata_MoA', columns='Metadata_Source',
                             values='Metadata_JCP2022', aggfunc='nunique', fill_value=0)
    pivoted_copy = pivoted.copy()
    dropped_items = []

    for _ in range(500):
        binary = pivoted_copy.clip(upper=1)
        source_means = binary.mean(axis=0)
        moa_means = binary.mean(axis=1)

        if source_means.min() < moa_means.min():
            drop_label = source_means.idxmin()
            pivoted_copy.drop(drop_label, axis=1, inplace=True)
            dropped_items.append(('source', drop_label))
        else:
            drop_label = moa_means.idxmin()
            pivoted_copy.drop(drop_label, axis=0, inplace=True)
            dropped_items.append(('moa', drop_label))

        if drop_label == 'source_10':
            break

    dropped_sources = {item for kind, item in dropped_items if kind == 'source'}
    dropped_moas = {item for kind, item in dropped_items if kind == 'moa'}

    print(f"Dropped {len(dropped_sources)} sources, {len(dropped_moas)} MoAs")
    for model in moa_profiles:
        df = moa_profiles[model]
        before = len(df)
        df = df[~df['Metadata_Source'].isin(dropped_sources)]
        df = df[~df['Metadata_MoA'].isin(dropped_moas)]
        moa_profiles[model] = df
        print(f"  {model}: {before} -> {len(df)} samples")
    return moa_profiles



def plot_heatmap(cpjump_summary):
    """Plot the combined BBBC036 + cpg-MoA enrichment heatmap matching the original style."""
    import matplotlib.gridspec as gridspec

    BBBC_NAME_MAP = {
        "dino_v2_cls_token": "dino_v2_cls",
        "dino_v2_patch_token": "dino_v2_patch",
    }
    bbbc_path = Path(BBBC_RESULTS_PATH)
    if bbbc_path.exists():
        print("Loading BBBC036 results...")
        with bbbc_path.open("rb") as f:
            bbbc_results = pickle.load(f)
        for bbbc_key, r in bbbc_results.items():
            canonical = BBBC_NAME_MAP.get(bbbc_key, bbbc_key)
            if canonical in ALL_MODELS:
                geom_or = r.get('geometric_mean_odds', r.get('geometric_mean_or'))
                cpjump_summary[(canonical, 'BBBC036')] = {
                    'frac_sig': r['fraction_significant'],
                    'geom_or': geom_or,
                }
    else:
        print(f"BBBC036 MoA results not found at {bbbc_path}; plotting cpg-MoA only.")

    cpg_keys = ['CPJump-Global', 'CPJump-Not Same Batch', 'CPJump-Not Same Source']
    bbbc_col_labels = ['No Restriction']
    cpg_col_labels = ['No Restriction', 'Not Same Batch', 'Not Same Source']

    n_models = len(ALL_MODELS)
    bbbc_frac = np.full((n_models, 1), np.nan)
    bbbc_geom = np.full((n_models, 1), np.nan)
    cpg_frac = np.full((n_models, 3), np.nan)
    cpg_geom = np.full((n_models, 3), np.nan)

    for i, model in enumerate(ALL_MODELS):
        key = (model, 'BBBC036')
        if key in cpjump_summary:
            bbbc_frac[i, 0] = cpjump_summary[key]['frac_sig']
            bbbc_geom[i, 0] = cpjump_summary[key]['geom_or']
        for j, ev in enumerate(cpg_keys):
            key = (model, ev)
            if key in cpjump_summary:
                cpg_frac[i, j] = cpjump_summary[key]['frac_sig']
                cpg_geom[i, j] = cpjump_summary[key]['geom_or']

    row_labels = [MODEL_LABELS[m] for m in ALL_MODELS]

    all_frac = np.concatenate([bbbc_frac.ravel(), cpg_frac.ravel()])
    all_geom = np.concatenate([bbbc_geom.ravel(), cpg_geom.ravel()])
    frac_vmin, frac_vmax = np.nanmin(all_frac), np.nanmax(all_frac)
    geom_vmin, geom_vmax = np.nanmin(all_geom), np.nanmax(all_geom)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='0.88')

    # [bbbc_frac, cpg_frac, cbar_frac, gap, bbbc_geom, cpg_geom, cbar_geom]
    fig = plt.figure(figsize=(22, 6))
    gs = gridspec.GridSpec(1, 7,
        width_ratios=[1.3, 3, 0.2, 1.2, 1.3, 3, 0.2],
        wspace=0.05)

    def draw_panel(ax, data, xlabels, title, fmt, vmin, vmax, show_ylabels):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        masked = np.ma.array(data, mask=np.isnan(data))
        im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha='right', fontsize=10)
        ax.set_yticks(range(n_models))
        if show_ylabels:
            ax.set_yticklabels(row_labels, fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                if np.isnan(val):
                    continue
                else:
                    rgba = cmap(norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    color = 'white' if lum < 0.5 else 'black'
                    ax.text(c, r, f'{val:{fmt}}', ha='center', va='center',
                            color=color, fontsize=10)
        return im

    # Fraction Significant
    ax_fb = fig.add_subplot(gs[0, 0])
    ax_fc = fig.add_subplot(gs[0, 1])
    ax_fcb = fig.add_subplot(gs[0, 2])
    draw_panel(ax_fb, bbbc_frac, bbbc_col_labels, 'BBBC036', '.3f', frac_vmin, frac_vmax, True)
    im_f = draw_panel(ax_fc, cpg_frac, cpg_col_labels, 'cpg-MoA', '.3f', frac_vmin, frac_vmax, False)
    fig.colorbar(im_f, cax=ax_fcb)

    # Gap
    ax_gap = fig.add_subplot(gs[0, 3])
    ax_gap.axis('off')

    # Geometric Mean OR
    ax_gb = fig.add_subplot(gs[0, 4])
    ax_gc = fig.add_subplot(gs[0, 5])
    ax_gcb = fig.add_subplot(gs[0, 6])
    draw_panel(ax_gb, bbbc_geom, bbbc_col_labels, 'BBBC036', '.2f', geom_vmin, geom_vmax, True)
    im_g = draw_panel(ax_gc, cpg_geom, cpg_col_labels, 'cpg-MoA', '.2f', geom_vmin, geom_vmax, False)
    fig.colorbar(im_g, cax=ax_gcb)

    # Metric supertitles
    fig.canvas.draw()
    frac_mid = (ax_fb.get_position().x0 + ax_fcb.get_position().x1) / 2
    fig.text(frac_mid, 0.98, 'Fraction Significant',
             ha='center', va='bottom', fontsize=13, fontweight='bold')
    geom_mid = (ax_gb.get_position().x0 + ax_gcb.get_position().x1) / 2
    fig.text(geom_mid, 0.98, 'Geometric Mean OR',
             ha='center', va='bottom', fontsize=13, fontweight='bold')

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap saved to {OUTPUT_PNG}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip enrichment computation, just replot from saved pkl")
    args = parser.parse_args()

    if args.plot_only:
        print("Loading saved results from", OUTPUT_PKL)
        with open(OUTPUT_PKL, "rb") as f:
            all_results = pickle.load(f)
        plot_heatmap(all_results['summary'])
        return

    # Load CPJump data
    print("Loading CPJump data...")
    with open(DATA_PATH, "rb") as f:
        moa_profiles = pickle.load(f)

    ref_model = next(iter(moa_profiles))
    pc_cols = [c for c in moa_profiles[ref_model].columns if c.startswith('PC') or c.startswith('F_')]
    print(f"Models: {list(moa_profiles.keys())}, PC cols: {len(pc_cols)}")

    # Apply subset filter
    print("\nApplying subset filtering...")
    moa_profiles = apply_subset_filter(moa_profiles)
    ref_agg = aggregate_by_compound(moa_profiles[ref_model], pc_cols)
    cpjump_cutoff_items = max(1, int((len(ref_agg) - 1) * CUTOFF_FRACTION))
    print(f"Using single CPJump cutoff: {cpjump_cutoff_items} "
          f"({len(ref_agg)} unique compounds, {len(ref_agg) - 1} self-excluded candidates)")

    # ── Run 4 paradigms ──────────────────────────────────────────────────
    # 1. Global
    print("\n=== Global Enrichment ===")
    global_results = {}
    for model, df in moa_profiles.items():
        print(f"  {model}...")
        agg = aggregate_by_compound(df, pc_cols)
        result = run_enrichment(agg, pc_cols=pc_cols, cutoff_items_override=cpjump_cutoff_items)
        global_results[model] = result
        print(f"    N={result['n_processed']}, Frac={result['fraction_significant']:.4f}, "
              f"GeomOR={result['geometric_mean_or']:.2f}")

    # 2. Within Source
    print("\n=== Within Source Enrichment ===")
    within_source_results = {}
    for model, df in moa_profiles.items():
        print(f"  {model}...")
        within_source_results[model] = {}
        sources = sorted(df['Metadata_Source'].unique())
        if MAX_SOURCES is not None:
            sources = sources[:MAX_SOURCES]
        for source in sources:
            source_df = df[df['Metadata_Source'] == source]
            agg = aggregate_by_compound(source_df, pc_cols)
            if len(agg) < 3:
                continue
            result = run_enrichment(agg, pc_cols=pc_cols, cutoff_items_override=cpjump_cutoff_items)
            within_source_results[model][source] = result
        agg_r = compute_aggregate(within_source_results[model])
        print(f"    Aggregate: N={agg_r['n_processed']}, Frac={agg_r['fraction_significant']:.4f}, "
              f"GeomOR={agg_r['geometric_mean_or']:.2f}")

    # 3. Not Same Batch
    print("\n=== Not Same Batch Enrichment ===")
    not_same_batch_results = {}
    for model, df in moa_profiles.items():
        print(f"  {model}...")
        not_same_batch_results[model] = {}
        batches = sorted(df['Metadata_Batch'].unique())
        if MAX_BATCHES is not None:
            batches = batches[:MAX_BATCHES]
        for batch in tqdm(batches, desc=f"    {model}", leave=False):
            query_df = df[df['Metadata_Batch'] == batch]
            comp_df = df[df['Metadata_Batch'] != batch]
            if len(query_df) == 0 or comp_df['Metadata_JCP2022'].nunique() < 3:
                continue
            result = run_enrichment(query_df, comp_df, pc_cols=pc_cols,
                                    cutoff_items_override=cpjump_cutoff_items)
            not_same_batch_results[model][batch] = result
        agg_r = compute_aggregate(not_same_batch_results[model])
        print(f"    Aggregate: N={agg_r['n_processed']}, Frac={agg_r['fraction_significant']:.4f}, "
              f"GeomOR={agg_r['geometric_mean_or']:.2f}")

    # 4. Not Same Source
    print("\n=== Not Same Source Enrichment ===")
    not_same_source_results = {}
    for model, df in moa_profiles.items():
        print(f"  {model}...")
        not_same_source_results[model] = {}
        sources = sorted(df['Metadata_Source'].unique())
        if MAX_SOURCES is not None:
            sources = sources[:MAX_SOURCES]
        for source in sources:
            query_df = df[df['Metadata_Source'] == source]
            comp_df = df[df['Metadata_Source'] != source]
            if len(query_df) == 0 or comp_df['Metadata_JCP2022'].nunique() < 3:
                continue
            result = run_enrichment(query_df, comp_df, pc_cols=pc_cols,
                                    cutoff_items_override=cpjump_cutoff_items)
            not_same_source_results[model][source] = result
        agg_r = compute_aggregate(not_same_source_results[model])
        print(f"    Aggregate: N={agg_r['n_processed']}, Frac={agg_r['fraction_significant']:.4f}, "
              f"GeomOR={agg_r['geometric_mean_or']:.2f}")

    # ── Build summary ────────────────────────────────────────────────────
    cpjump_summary = {}
    for cpjump_name, canonical in CPJUMP_TO_CANONICAL.items():
        if cpjump_name not in global_results:
            continue
        # Global
        r = global_results[cpjump_name]
        cpjump_summary[(canonical, 'CPJump-Global')] = {
            'frac_sig': r['fraction_significant'],
            'geom_or': r['geometric_mean_or'],
        }
        # Within Source
        r = compute_aggregate(within_source_results[cpjump_name])
        cpjump_summary[(canonical, 'CPJump-Within Source')] = {
            'frac_sig': r['fraction_significant'],
            'geom_or': r['geometric_mean_or'],
        }
        # Not Same Batch — majority vote
        r = compute_aggregate_majority_vote(not_same_batch_results[cpjump_name])
        cpjump_summary[(canonical, 'CPJump-Not Same Batch')] = {
            'frac_sig': r['fraction_significant'],
            'geom_or': r['geometric_mean_or'],
        }
        # Not Same Source — majority vote
        r = compute_aggregate_majority_vote(not_same_source_results[cpjump_name])
        cpjump_summary[(canonical, 'CPJump-Not Same Source')] = {
            'frac_sig': r['fraction_significant'],
            'geom_or': r['geometric_mean_or'],
        }


    # ── Save results ─────────────────────────────────────────────────────
    all_results = {
        'global': global_results,
        'within_source': within_source_results,
        'not_same_batch': not_same_batch_results,
        'not_same_source': not_same_source_results,
        'summary': cpjump_summary,
    }
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Results saved to {OUTPUT_PKL}')

    # ── Plot heatmap ─────────────────────────────────────────────────────
    plot_heatmap(cpjump_summary)


if __name__ == '__main__':
    main()
