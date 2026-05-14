"""
Replicate-analysis benchmark sweep.

Runs selectable replicate-analysis benchmarks while keeping all restrictions fixed:
KNN Recall@1, standard replicate mAP, and negative-control mAP.

Usage:
    python benchmarks/replicate_analysis/run_replicate_analysis_sweep.py \
      --features-base data/main_paper_inputs/normalized/CSAll_Plate__PCA64__MADCtrl_Plate__NoSph \
      --results-dir results/knn
"""

import argparse
import os, pickle, subprocess, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from config import (
    DEFAULT_CONFIG,
    default_profile,
    enabled_names,
    load_yaml,
    parse_csv,
    profile_features_dir,
    profile_results_dir,
    resolve_path,
)
from qc.cell_count_qc import (
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    cell_count_qc_cli_args,
    cell_count_qc_enabled,
    make_cell_count_qc_mask,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Benchmark sweep YAML config")
parser.add_argument("--profile", default=None, help="Normalization profile to benchmark")
parser.add_argument("--benchmarks", default=None, help="Comma-separated subset: knn_replicate,map,negcon_map")
parser.add_argument("--features-base", default=None, help="Base dir with {dataset}/ subdirs of normalized parquets")
parser.add_argument("--results-dir", default=None, help="Output dir for results pkl")
parser.add_argument("--dry-run", action="store_true", help="Print resolved actions without running benchmarks")
parser.add_argument(
    "--only-dataset",
    choices=["both", "cpg-tgt2", "cpg-compound"],
    default="both",
    help="Run both KNN panels or only one dataset panel.",
)
parser.add_argument(
    "--cpg-compound-selection",
    choices=["sample10k", "all_eligible"],
    default=None,
    help="How to select cpg-compound compounds. all_eligible uses every non-control compound with matches at all stringency levels.",
)
parser.add_argument(
    "--estimate-only",
    action="store_true",
    help="Only report cpg-compound eligibility counts and exit before computing metrics.",
)
add_cell_count_qc_args(parser)
args = parser.parse_args()

CONFIG = load_yaml(Path(args.config))
PROFILE = args.profile or default_profile(CONFIG)
FEATURES_BASE = args.features_base or str(profile_features_dir(CONFIG, PROFILE))
RESULTS_DIR = args.results_dir or str(profile_results_dir(CONFIG, PROFILE) / "replicate_analysis")
GRAPHICS_DIR = RESULTS_DIR
QC_MASKS = {}
_configured = CONFIG.get("replicate_analysis", {}).get("run", {})
SELECTED_BENCHMARKS = set(parse_csv(args.benchmarks) or enabled_names(_configured))
RUN_KNN_REPLICATE = "knn_replicate" in SELECTED_BENCHMARKS
RUN_MAP = "map" in SELECTED_BENCHMARKS
RUN_NEGCON_MAP = "negcon_map" in SELECTED_BENCHMARKS
if not SELECTED_BENCHMARKS:
    raise ValueError("No replicate-analysis benchmarks selected")
unknown = SELECTED_BENCHMARKS - {"knn_replicate", "map", "negcon_map"}
if unknown:
    raise ValueError(f"Unknown replicate-analysis benchmarks: {sorted(unknown)}")
if args.cpg_compound_selection is None:
    args.cpg_compound_selection = (
        CONFIG.get("replicate_analysis", {})
        .get("cpg_compound_selection", {})
        .get("main", "sample10k")
    )
qc_defaults = CONFIG.get("qc", {})
if qc_defaults.get("enabled", False):
    args.cell_count_qc = True
    args.qc_cell_counts_dir = str(resolve_path(CONFIG["paths"]["qc_cell_counts_dir"]))
    args.qc_cell_count_cutoff = qc_defaults.get("cell_count_cutoff", args.qc_cell_count_cutoff)
    args.qc_sample_wells = qc_defaults.get("sample_wells", args.qc_sample_wells)
    args.qc_seed = qc_defaults.get("seed", args.qc_seed)

MODELS = {
    'cellprofiler':       'cellprofiler_normalized.parquet',
    'cloome':             'cloome_normalized.parquet',
    'dino_v2_cls_token':  'dino_v2_cls_token_normalized.parquet',
    'dino_v2_patch_token':'dino_v2_patch_token_normalized.parquet',
    'open_phenom':        'open_phenom_normalized.parquet',
    'resnet':             'resnet_normalized.parquet',
    'resnet_untrained':   'resnet_untrained_normalized.parquet',
    'subcell':            'subcell_normalized.parquet',
}
MODEL_DISPLAY = {
    'cellprofiler': 'CellProfiler', 'cloome': 'CLOOME',
    'dino_v2_cls_token': 'DINOv2 CLS', 'dino_v2_patch_token': 'DINOv2 Patch',
    'open_phenom': 'OpenPhenom', 'resnet': 'ResNet',
    'resnet_untrained': 'ResNet (UT)', 'subcell': 'SubCell',
}
MODEL_ORDER = ['cellprofiler', 'dino_v2_cls_token', 'dino_v2_patch_token',
               'resnet', 'subcell', 'open_phenom', 'cloome']

DMSO_JCP = 'JCP2022_033924'
EXCLUDE_COMPOUNDS = {'JCP2022_999999', 'JCP2022_UNKNOWN'}
MAX_WELLS_PER_COMPOUND = 200
N_COMPOUNDS = 10000
MAX_WELLS_PER_COMPOUND_PER_SOURCE = 2
SEED = 42
CHUNK_SIZE = 2000


def _l2_normalize(X):
    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (X / norms).astype(np.float32)


def _ap_from_sorted(neg_sorted, rel_sims):
    R = len(rel_sims)
    ranks = np.searchsorted(neg_sorted, -rel_sims, side='left') + 1
    ranks_sorted = np.sort(ranks)
    return float(np.mean(np.arange(1, R + 1, dtype=np.float64) / ranks_sorted))


# ── cpg-compound helpers ─────────────────────────────────────────────────────

def maybe_apply_cell_count_qc(df, dataset_name):
    if not cell_count_qc_enabled(args):
        return df
    if dataset_name not in QC_MASKS:
        QC_MASKS[dataset_name] = make_cell_count_qc_mask(
            df,
            dataset_name=dataset_name,
            cell_counts_dir=args.qc_cell_counts_dir,
            cutoff=args.qc_cell_count_cutoff,
            sample_wells=args.qc_sample_wells,
            seed=args.qc_seed,
            output_dir=os.path.join(RESULTS_DIR, "cell_count_qc"),
        )
    before = len(df)
    df = apply_cell_count_qc_mask(df, QC_MASKS[dataset_name], dataset_name)
    print(f"  [cell-count QC] {dataset_name}: rows {before} -> {len(df)}")
    return df


def cpg_compound_eligible():
    features_dir = os.path.join(FEATURES_BASE, 'cpg-compound')
    per_model = []
    for model, fname in MODELS.items():
        df = pd.read_parquet(
            os.path.join(features_dir, fname),
            columns=[
                'Metadata_JCP2022', 'Metadata_Source', 'Metadata_Batch',
                'Metadata_Plate', 'Metadata_Well',
            ],
        )
        df = maybe_apply_cell_count_qc(df, 'cpg-compound')
        df = df[df['Metadata_JCP2022'] != DMSO_JCP]
        df = df[~df['Metadata_JCP2022'].isin(EXCLUDE_COMPOUNDS)]
        counts = df['Metadata_JCP2022'].value_counts()
        big = set(counts[counts > MAX_WELLS_PER_COMPOUND].index)
        df = df[~df['Metadata_JCP2022'].isin(big)]
        df = subsample_cpg(df, set(df['Metadata_JCP2022'].unique()))
        eligible = {
            compound
            for compound, group in df.groupby('Metadata_JCP2022')
            if _has_match_at_all_stringencies(group)
        }
        per_model.append(eligible)
        print(
            f"    {MODEL_DISPLAY[model]}: {len(eligible)} eligible compounds "
            f"after excluding {len(big)} high-replicate controls"
        )
    shared = sorted(set.intersection(*per_model))
    print(f"    Shared across all models: {len(shared)} compounds")
    return shared


def _has_match_at_all_stringencies(group):
    if len(group) < 2:
        return False
    batches = group['Metadata_Batch'].to_numpy()
    sources = group['Metadata_Source'].to_numpy()
    wells = group['Metadata_Well'].to_numpy()

    has_nr = len(group) >= 2
    has_nsb = any((batches != batches[i]).any() for i in range(len(group)))
    has_nss = any((sources != sources[i]).any() for i in range(len(group)))
    has_nsl = any((wells != wells[i]).any() for i in range(len(group)))
    has_nssl = any(
        ((sources != sources[i]) & (wells != wells[i])).any()
        for i in range(len(group))
    )
    return has_nr and has_nsb and has_nss and has_nsl and has_nssl


def subsample_cpg(df, compounds):
    df = df[df['Metadata_JCP2022'].isin(compounds)].copy()
    rng = np.random.RandomState(SEED)
    keep = []
    for (c, s), g in df.groupby(['Metadata_JCP2022', 'Metadata_Source']):
        if len(g) <= MAX_WELLS_PER_COMPOUND_PER_SOURCE:
            keep.extend(g.index.tolist())
        else:
            chosen = rng.choice(g.index.values, MAX_WELLS_PER_COMPOUND_PER_SOURCE, replace=False)
            keep.extend(chosen.tolist())
    return df.loc[keep].reset_index(drop=True)


# ── Unified mAP computation ─────────────────────────────────────────────────

def run_map(Xn, compounds, batches, sources, wells, paradigms):
    """mAP for specified paradigms. 1 query per compound.
    
    Only counts queries that have ≥1 relevant item in the filtered pool.
    """
    N = len(Xn)
    cmp_to_idx = {}
    for i, c in enumerate(compounds):
        cmp_to_idx.setdefault(c, []).append(i)

    rng = np.random.RandomState(SEED)
    query_comps = sorted([c for c, idxs in cmp_to_idx.items() if len(idxs) >= 2])
    query_idx = np.array([rng.choice(cmp_to_idx[c]) for c in query_comps])
    n_q = len(query_idx)

    q_compounds = compounds[query_idx]
    q_batches = batches[query_idx]
    q_sources = sources[query_idx]
    q_wells = wells[query_idx] if wells is not None else None

    batch_masks = {b: (batches == b) for b in np.unique(batches)}
    source_masks = {s: (sources == s) for s in np.unique(sources)}
    well_masks = {w: (wells == w) for w in np.unique(wells)} if wells is not None else {}

    # Precompute relevant indices
    cmp_rel = {p: {} for p in paradigms}
    for c, idxs in cmp_to_idx.items():
        arr = np.array(idxs, dtype=np.int64)
        if 'No Restriction' in paradigms:
            cmp_rel['No Restriction'][c] = arr
        if 'Not Same Batch' in paradigms:
            for b in np.unique(batches[arr]):
                cmp_rel['Not Same Batch'][(c, b)] = arr[batches[arr] != b]
        if 'Not Same Source' in paradigms:
            for s in np.unique(sources[arr]):
                cmp_rel['Not Same Source'][(c, s)] = arr[sources[arr] != s]
        if 'Not Same Layout' in paradigms and wells is not None:
            for w in np.unique(wells[arr]):
                cmp_rel['Not Same Layout'][(c, w)] = arr[wells[arr] != w]
        if 'Not Same Source Layout' in paradigms and wells is not None:
            for w in np.unique(wells[arr]):
                for s in np.unique(sources[arr]):
                    cmp_rel['Not Same Source Layout'][(c, s, w)] = arr[(sources[arr] != s) & (wells[arr] != w)]

    aps = {p: [] for p in paradigms}

    for ci in range(0, n_q, CHUNK_SIZE):
        ce = min(ci + CHUNK_SIZE, n_q)
        chunk_idx = query_idx[ci:ce]
        cs = ce - ci
        sim_base = Xn[chunk_idx] @ Xn.T

        for p in paradigms:
            sim = sim_base.copy()

            # Apply mask
            if p == 'No Restriction':
                for i in range(cs):
                    sim[i, chunk_idx[i]] = -np.inf
            elif p == 'Not Same Batch':
                for i in range(cs):
                    sim[i, batch_masks[q_batches[ci + i]]] = -np.inf
            elif p == 'Not Same Source':
                for i in range(cs):
                    sim[i, source_masks[q_sources[ci + i]]] = -np.inf
            elif p == 'Not Same Layout':
                for i in range(cs):
                    sim[i, well_masks[q_wells[ci + i]]] = -np.inf
            elif p == 'Not Same Source Layout':
                for i in range(cs):
                    mask = source_masks[q_sources[ci + i]] | well_masks[q_wells[ci + i]]
                    sim[i, mask] = -np.inf

            neg_sorted = np.sort(-sim, axis=1)

            for i in range(cs):
                qi = ci + i
                c = q_compounds[qi]

                if p == 'No Restriction':
                    rel = cmp_rel[p].get(c, np.array([]))
                    rel = rel[rel != query_idx[qi]]
                elif p == 'Not Same Batch':
                    rel = cmp_rel[p].get((c, q_batches[qi]), np.array([]))
                elif p == 'Not Same Source':
                    rel = cmp_rel[p].get((c, q_sources[qi]), np.array([]))
                elif p == 'Not Same Layout':
                    rel = cmp_rel[p].get((c, q_wells[qi]), np.array([]))
                elif p == 'Not Same Source Layout':
                    rel = cmp_rel[p].get((c, q_sources[qi], q_wells[qi]), np.array([]))

                if len(rel) > 0:
                    aps[p].append(_ap_from_sorted(neg_sorted[i], sim[i, rel]))

    result = {}
    for p in paradigms:
        vals = aps[p]
        result[p] = {'mAP': np.mean(vals) * 100 if vals else 0, 'n_queries': len(vals)}
    return result


# ── KNN with proper filtering ───────────────────────────────────────────────

def run_knn(Xn, compounds, batches, sources, wells, paradigms, k=1):
    """KNN Recall@K. Only counts queries with ≥1 possible match in filtered pool."""
    N = len(Xn)
    comp_uniq, comp_int = np.unique(compounds, return_inverse=True)

    batch_indices = {b: np.where(batches != b)[0] for b in np.unique(batches)}
    source_indices = {s: np.where(sources != s)[0] for s in np.unique(sources)}
    well_indices = {w: np.where(wells != w)[0] for w in np.unique(wells)} if wells is not None else {}

    # Precompute: for each query, does it have ≥1 same-compound well in filtered pool?
    # Build compound→indices mapping
    cmp_idx = {}
    for i in range(N):
        cmp_idx.setdefault(comp_int[i], []).append(i)

    results = {}
    for p in paradigms:
        hits = []
        eligible = []

        for ci in range(0, N, CHUNK_SIZE):
            ce = min(ci + CHUNK_SIZE, N)
            cs = ce - ci
            sims = Xn[ci:ce] @ Xn.T

            if p == 'No Restriction':
                for i in range(cs):
                    sims[i, ci + i] = -2.0
                topk = np.argpartition(-sims, k, axis=1)[:, :k]
                for i in range(cs):
                    qi = ci + i
                    qc = comp_int[qi]
                    # Check if there's at least 1 other same-compound well
                    n_reps = len(cmp_idx[qc]) - 1
                    if n_reps > 0:
                        eligible.append(qi)
                        hits.append(int((comp_int[topk[i]] == qc).any()))
            else:
                for i in range(cs):
                    qi = ci + i
                    row = sims[i]
                    qc = comp_int[qi]

                    if p == 'Not Same Batch':
                        pool = batch_indices[batches[qi]]
                    elif p == 'Not Same Source':
                        pool = source_indices[sources[qi]]
                    elif p == 'Not Same Layout':
                        pool = well_indices[wells[qi]]
                    elif p == 'Not Same Source Layout':
                        s_mask = (sources != sources[qi])
                        w_mask = (wells != wells[qi])
                        pool = np.where(s_mask & w_mask)[0]
                    else:
                        continue

                    # Check if any same-compound well exists in pool
                    pool_comps = comp_int[pool]
                    if not (pool_comps == qc).any():
                        continue  # skip — no possible match

                    eligible.append(qi)
                    ek = min(k, len(pool))
                    topk_p = pool[np.argpartition(-row[pool], ek)[:ek]]
                    hits.append(int((comp_int[topk_p] == qc).any()))

        recall = np.mean(hits) * 100 if hits else 0
        results[p] = {'recall': recall, 'n_eligible': len(eligible)}

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHICS_DIR, exist_ok=True)
    print(f"Replicate-analysis benchmarks: {', '.join(sorted(SELECTED_BENCHMARKS))}")
    print(f"Profile: {PROFILE}")
    print(f"Features: {FEATURES_BASE}")
    print(f"Results: {RESULTS_DIR}")
    if args.dry_run:
        print("Dry run complete.")
        return

    if not (RUN_KNN_REPLICATE or RUN_MAP):
        if RUN_NEGCON_MAP:
            run_negative_control_benchmark()
            return
        raise ValueError("Select at least one of knn_replicate, map, or negcon_map")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 1: cpg-tgt2 (4 paradigms)
    # ═══════════════════════════════════════════════════════════════════════
    tgt2_paradigms = ['No Restriction', 'Not Same Batch', 'Not Same Source', 'Not Same Layout', 'Not Same Source Layout']
    tgt2_dir = os.path.join(FEATURES_BASE, 'cpg-tgt2')
    tgt2_knn = {}
    tgt2_map = {}

    if args.only_dataset in ("both", "cpg-tgt2"):
        print("=" * 80)
        print("Panel 1: cpg-tgt2 — 4 paradigms")
        print("=" * 80)

        for model, fname in MODELS.items():
            t0 = time.time()
            path = os.path.join(tgt2_dir, fname)
            df = pd.read_parquet(path)
            df = maybe_apply_cell_count_qc(df, 'cpg-tgt2')
            df = df[df['Metadata_JCP2022'] != DMSO_JCP].reset_index(drop=True)

            feat_cols = [c for c in df.columns if not c.startswith('Metadata_')]
            Xn = _l2_normalize(df[feat_cols].values)
            compounds = df['Metadata_JCP2022'].values
            batches = df['Metadata_Batch'].values
            sources = df['Metadata_Source'].values
            wells = df['Metadata_Well'].values

            display = MODEL_DISPLAY[model]
            print(f"\n  {display}: {len(df)} wells")

            knn = run_knn(Xn, compounds, batches, sources, wells, tgt2_paradigms) if RUN_KNN_REPLICATE else {}
            if RUN_KNN_REPLICATE:
                tgt2_knn[model] = knn

            mp = run_map(Xn, compounds, batches, sources, wells, tgt2_paradigms) if RUN_MAP else {}
            if RUN_MAP:
                tgt2_map[model] = mp

            elapsed = time.time() - t0
            for p in tgt2_paradigms:
                parts = []
                if RUN_KNN_REPLICATE:
                    parts.append(f"KNN={knn[p]['recall']:.1f}% ({knn[p]['n_eligible']}q)")
                if RUN_MAP:
                    parts.append(f"mAP={mp[p]['mAP']:.2f}% ({mp[p]['n_queries']}q)")
                print(f"    {p}: " + "  ".join(parts))
            print(f"    [{elapsed:.0f}s]")

    # ═══════════════════════════════════════════════════════════════════════
    # Panel 2: cpg-compound (4 paradigms)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("Panel 2: cpg-compound — 4 paradigms")
    print("=" * 80)

    cpg_paradigms = ['No Restriction', 'Not Same Batch', 'Not Same Source', 'Not Same Layout', 'Not Same Source Layout']
    cpg_dir = os.path.join(FEATURES_BASE, 'cpg-compound')

    print("\n  Finding eligible compounds...")
    shared = cpg_compound_eligible()
    if args.cpg_compound_selection == "sample10k":
        rng = np.random.RandomState(SEED)
        selected = sorted(rng.choice(shared, N_COMPOUNDS, replace=False)) if len(shared) > N_COMPOUNDS else shared
    else:
        selected = shared
    selected_set = set(selected)
    print(f"  Selected {len(selected)} compounds ({args.cpg_compound_selection})")

    cpg_knn = {}
    cpg_map = {}
    cpg_well_counts = {}

    for model, fname in MODELS.items():
        t0 = time.time()
        path = os.path.join(cpg_dir, fname)
        df = pd.read_parquet(path)
        df = maybe_apply_cell_count_qc(df, 'cpg-compound')
        df = df[df['Metadata_JCP2022'] != DMSO_JCP].reset_index(drop=True)
        df = subsample_cpg(df, selected_set)
        cpg_well_counts[model] = len(df)

        if args.estimate_only:
            print(f"  {MODEL_DISPLAY[model]}: {len(df)} wells after cpg-compound selection")
            continue

        feat_cols = [c for c in df.columns if not c.startswith('Metadata_')]
        Xn = _l2_normalize(df[feat_cols].values)
        compounds = df['Metadata_JCP2022'].values
        batches = df['Metadata_Batch'].values
        sources = df['Metadata_Source'].values
        wells = df['Metadata_Well'].values

        display = MODEL_DISPLAY[model]
        print(f"\n  {display}: {len(df)} wells")

        knn = run_knn(Xn, compounds, batches, sources, wells, cpg_paradigms) if RUN_KNN_REPLICATE else {}
        if RUN_KNN_REPLICATE:
            cpg_knn[model] = knn

        mp = run_map(Xn, compounds, batches, sources, wells, cpg_paradigms) if RUN_MAP else {}
        if RUN_MAP:
            cpg_map[model] = mp

        elapsed = time.time() - t0
        for p in cpg_paradigms:
            parts = []
            if RUN_KNN_REPLICATE:
                parts.append(f"KNN={knn[p]['recall']:.2f}% ({knn[p]['n_eligible']}q)")
            if RUN_MAP:
                parts.append(f"mAP={mp[p]['mAP']:.4f}% ({mp[p]['n_queries']}q)")
            print(f"    {p}: " + "  ".join(parts))
        print(f"    [{elapsed:.0f}s]")

    if args.estimate_only:
        out = {
            'selected_compounds': selected,
            'n_selected_compounds': len(selected),
            'cpg_well_counts': cpg_well_counts,
            'selection': args.cpg_compound_selection,
            'max_wells_per_compound_per_source': MAX_WELLS_PER_COMPOUND_PER_SOURCE,
            'excluded_negative_control': DMSO_JCP,
            'excluded_compounds': sorted(EXCLUDE_COMPOUNDS),
            'max_wells_per_compound_positive_control_threshold': MAX_WELLS_PER_COMPOUND,
            'cell_count_qc': {
                'enabled': cell_count_qc_enabled(args),
                'summaries': {
                    dataset: mask.attrs.get('cell_count_qc_summary', {})
                    for dataset, mask in QC_MASKS.items()
                },
            },
        }
        pkl_path = os.path.join(RESULTS_DIR, 'cpg_compound_all_eligible_selection.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"Selection report: {pkl_path}")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Summary tables
    # ═══════════════════════════════════════════════════════════════════════
    summary_items = []
    if tgt2_knn:
        summary_items.append(("cpg-tgt2", tgt2_paradigms, tgt2_knn, tgt2_map))
    if cpg_knn:
        summary_items.append(("cpg-compound", cpg_paradigms, cpg_knn, cpg_map))
    for dataset, paradigms, knn_res, map_res in summary_items:
        if RUN_MAP:
            print(f"\n{'='*85}")
            print(f"mAP — {dataset}")
            print(f"{'='*85}")
            print(f"{'Model':<16}" + "".join(f"{p:>18}" for p in paradigms))
            print("-" * (16 + 18 * len(paradigms)))
            for m in sorted(map_res, key=lambda m: -map_res[m]['No Restriction']['mAP']):
                line = f"{MODEL_DISPLAY[m]:<16}"
                for p in paradigms:
                    line += f"{map_res[m][p]['mAP']:>17.3f}%"
                print(line)

        if RUN_KNN_REPLICATE:
            print(f"\nKNN Recall@1 — {dataset}")
            print(f"{'Model':<16}" + "".join(f"{p:>18}" for p in paradigms))
            print("-" * (16 + 18 * len(paradigms)))
            for m in sorted(knn_res, key=lambda m: -knn_res[m]['No Restriction']['recall']):
                line = f"{MODEL_DISPLAY[m]:<16}"
                for p in paradigms:
                    line += f"{knn_res[m][p]['recall']:>17.2f}%"
                print(line)

    if args.only_dataset != "both":
        out = {
            'cpg_knn': cpg_knn, 'cpg_map': cpg_map,
            'cpg_paradigms': cpg_paradigms,
            'selected_compounds': selected,
            'n_selected_compounds': len(selected),
            'cpg_well_counts': cpg_well_counts,
            'selection': args.cpg_compound_selection,
            'cell_count_qc': {
                'enabled': cell_count_qc_enabled(args),
                'summaries': {
                    dataset: mask.attrs.get('cell_count_qc_summary', {})
                    for dataset, mask in QC_MASKS.items()
                },
            },
        }
        pkl_name = 'knn_cpg_compound_all_eligible.pkl' if args.cpg_compound_selection == "all_eligible" else 'knn_cpg_compound_sample10k.pkl'
        pkl_path = os.path.join(RESULTS_DIR, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"Results: {pkl_path}")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # 2-Panel heatmap
    # ═══════════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort models by the first selected metric on tgt2 (use consistent order)
    if RUN_MAP and tgt2_map:
        models_sorted = sorted(tgt2_map, key=lambda m: -tgt2_map[m]['No Restriction']['mAP'])
    else:
        models_sorted = sorted(tgt2_knn, key=lambda m: -tgt2_knn[m]['No Restriction']['recall'])
    labels = [MODEL_DISPLAY[m] for m in models_sorted]

    def fmt_2sig(val):
        """Format a number to 2 significant digits."""
        if val == 0:
            return "0"
        import math
        digits = -int(math.floor(math.log10(abs(val)))) + 1
        return f"{val:.{max(digits, 0)}f}"

    metric_specs = []
    if RUN_MAP:
        metric_specs.append(("mAP", lambda res, m, p: res[m][p]['mAP'], 'mAP (%)', 'map'))
    if RUN_KNN_REPLICATE:
        metric_specs.append(("KNN Recall@1", lambda res, m, p: res[m][p]['recall'], 'Recall@1 (%)', 'knn'))
    for metric, get_val, cbar_label, fname_suffix in metric_specs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                        gridspec_kw={'width_ratios': [1, 1]})

        # Panel 1: tgt2
        data1 = np.array([[get_val(tgt2_map if metric == "mAP" else tgt2_knn, m, p)
                           for p in tgt2_paradigms] for m in models_sorted])
        annot1 = np.array([[fmt_2sig(v) for v in row] for row in data1])
        sns.heatmap(data1, annot=annot1, fmt='', cmap='YlOrRd',
                    xticklabels=tgt2_paradigms, yticklabels=labels,
                    ax=ax1, vmin=0, cbar_kws={'label': cbar_label})
        ax1.set_title(f'{metric} — cpg-tgt2\n(301 compounds)', fontsize=11, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        # Panel 2: cpg-compound
        data2 = np.array([[get_val(cpg_map if metric == "mAP" else cpg_knn, m, p)
                           for p in cpg_paradigms] for m in models_sorted])
        annot2 = np.array([[fmt_2sig(v) for v in row] for row in data2])
        sns.heatmap(data2, annot=annot2, fmt='', cmap='YlOrRd',
                    xticklabels=cpg_paradigms, yticklabels=labels,
                    ax=ax2, vmin=0, cbar_kws={'label': cbar_label})
        ax2.set_title(f'{metric} — cpg-compound\n({len(selected)} compounds)', fontsize=11, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fig_path = os.path.join(GRAPHICS_DIR, f'{fname_suffix}_2panel.png')
        plt.savefig(fig_path, dpi=150)
        print(f"\nHeatmap: {fig_path}")
        plt.close()

    # Save all results
    out = {
        'tgt2_knn': tgt2_knn, 'tgt2_map': tgt2_map,
        'cpg_knn': cpg_knn, 'cpg_map': cpg_map,
        'tgt2_paradigms': tgt2_paradigms, 'cpg_paradigms': cpg_paradigms,
        'selected_compounds': selected,
        'n_selected_compounds': len(selected),
        'cpg_well_counts': cpg_well_counts,
        'selection': args.cpg_compound_selection,
        'cell_count_qc': {
            'enabled': cell_count_qc_enabled(args),
            'summaries': {
                dataset: mask.attrs.get('cell_count_qc_summary', {})
                for dataset, mask in QC_MASKS.items()
            },
        },
    }
    pkl_path = os.path.join(RESULTS_DIR, 'knn_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Results: {pkl_path}")

    if RUN_NEGCON_MAP:
        ensure_all_eligible_selection()
        run_negative_control_benchmark()


def ensure_all_eligible_selection() -> None:
    selection_pickle = Path(f"{RESULTS_DIR}_cpg_all_eligible") / "knn_cpg_compound_all_eligible.pkl"
    if selection_pickle.exists():
        return
    benchmarks = [name for name in ("knn_replicate", "map") if name in SELECTED_BENCHMARKS]
    if not benchmarks:
        benchmarks = ["map"]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(Path(args.config)),
        "--profile",
        PROFILE,
        "--benchmarks",
        ",".join(benchmarks),
        "--features-base",
        FEATURES_BASE,
        "--results-dir",
        f"{RESULTS_DIR}_cpg_all_eligible",
        "--only-dataset",
        "cpg-compound",
        "--cpg-compound-selection",
        "all_eligible",
        *cell_count_qc_cli_args(args),
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_negative_control_benchmark() -> None:
    negcon_dir = f"{RESULTS_DIR}_negative_control_map"
    selection_pickle = os.path.join(
        f"{RESULTS_DIR}_cpg_all_eligible",
        "knn_cpg_compound_all_eligible.pkl",
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "negative_control_map.py"),
        "--features-base",
        FEATURES_BASE,
        "--results-dir",
        negcon_dir,
        "--selection-pickle",
        selection_pickle,
        *cell_count_qc_cli_args(args),
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
