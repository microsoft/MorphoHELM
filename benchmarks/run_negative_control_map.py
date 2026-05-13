"""
Negative-control mAP benchmark for the main normalization-sweep setting.

For each non-DMSO query compound, candidates are restricted to DMSO wells plus
same-compound replicate wells. Positives are the same-compound replicates.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc.cell_count_qc import (  # noqa: E402
    add_cell_count_qc_args,
    apply_cell_count_qc_mask,
    cell_count_qc_enabled,
    make_cell_count_qc_mask,
)


MODELS = {
    "cellprofiler": "cellprofiler_normalized.parquet",
    "cloome": "cloome_normalized.parquet",
    "dino_v2_cls_token": "dino_v2_cls_token_normalized.parquet",
    "dino_v2_patch_token": "dino_v2_patch_token_normalized.parquet",
    "open_phenom": "open_phenom_normalized.parquet",
    "resnet": "resnet_normalized.parquet",
    "resnet_untrained": "resnet_untrained_normalized.parquet",
    "subcell": "subcell_normalized.parquet",
}

MODEL_DISPLAY = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls_token": "DINOv2 CLS",
    "dino_v2_patch_token": "DINOv2 Patch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "ResNet (UT)",
    "subcell": "SubCell",
}

PARADIGMS = [
    "No Restriction",
    "Not Same Batch",
    "Not Same Source",
    "Not Same Layout",
    "Not Same Source Layout",
]

DMSO_JCP = "JCP2022_033924"
EXCLUDE_COMPOUNDS = {"JCP2022_999999", "JCP2022_UNKNOWN"}
MAX_WELLS_PER_COMPOUND = 200
MAX_WELLS_PER_COMPOUND_PER_SOURCE = 2
SEED = 42
DEFAULT_CHUNK_SIZE = 512
DEFAULT_SELECTION_PICKLE = (
    "results/normalization_sweep/"
    "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph_cpg_all_eligible/"
    "knn_cpg_compound_all_eligible.pkl"
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--features-base",
    required=True,
    help="Base dir with {cpg-tgt2,cpg-compound}/ normalized parquet subdirs.",
)
parser.add_argument("--results-dir", required=True, help="Output directory.")
parser.add_argument(
    "--selection-pickle",
    default=DEFAULT_SELECTION_PICKLE,
    help="Existing all-eligible cpg-compound result pickle with selected_compounds.",
)
parser.add_argument(
    "--chunk-size",
    type=int,
    default=DEFAULT_CHUNK_SIZE,
    help="Number of query compounds per similarity chunk.",
)
add_cell_count_qc_args(parser)
args = parser.parse_args()

FEATURES_BASE = args.features_base
RESULTS_DIR = args.results_dir
QC_MASKS: dict[str, pd.DataFrame] = {}


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (X / norms).astype(np.float32)


def _summary(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    arr = np.asarray(values)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _ap_from_negative_controls(neg_sorted: np.ndarray, rel_sims: np.ndarray) -> float:
    """AP after inserting relevant scores into sorted DMSO score distribution."""
    rel_desc = np.sort(rel_sims)[::-1]
    n_rel = len(rel_desc)
    n_neg_higher = np.searchsorted(neg_sorted, -rel_desc, side="left")
    positive_ranks = n_neg_higher + np.arange(1, n_rel + 1)
    precision_at_positive = np.arange(1, n_rel + 1, dtype=np.float64) / positive_ranks
    return float(precision_at_positive.mean())


def maybe_apply_cell_count_qc(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
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
    print(f"  [cell-count QC] {dataset_name}: rows {before} -> {len(df)}", flush=True)
    return df


def _has_match_at_all_stringencies(group: pd.DataFrame) -> bool:
    if len(group) < 2:
        return False
    batches = group["Metadata_Batch"].to_numpy()
    sources = group["Metadata_Source"].to_numpy()
    wells = group["Metadata_Well"].to_numpy()
    return (
        len(group) >= 2
        and any((batches != batches[i]).any() for i in range(len(group)))
        and any((sources != sources[i]).any() for i in range(len(group)))
        and any((wells != wells[i]).any() for i in range(len(group)))
        and any(((sources != sources[i]) & (wells != wells[i])).any() for i in range(len(group)))
    )


def subsample_cpg(df: pd.DataFrame, compounds: set[str]) -> pd.DataFrame:
    df = df[df["Metadata_JCP2022"].isin(compounds)].copy()
    rng = np.random.RandomState(SEED)
    keep: list[int] = []
    for (_compound, _source), group in df.groupby(["Metadata_JCP2022", "Metadata_Source"]):
        if len(group) <= MAX_WELLS_PER_COMPOUND_PER_SOURCE:
            keep.extend(group.index.tolist())
        else:
            chosen = rng.choice(
                group.index.values,
                MAX_WELLS_PER_COMPOUND_PER_SOURCE,
                replace=False,
            )
            keep.extend(chosen.tolist())
    return df.loc[keep].reset_index(drop=True)


def cpg_compound_eligible() -> list[str]:
    features_dir = os.path.join(FEATURES_BASE, "cpg-compound")
    per_model = []
    columns = [
        "Metadata_JCP2022",
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
    ]
    for model, fname in MODELS.items():
        df = pd.read_parquet(os.path.join(features_dir, fname), columns=columns)
        df = maybe_apply_cell_count_qc(df, "cpg-compound")
        df = df[df["Metadata_JCP2022"] != DMSO_JCP]
        df = df[~df["Metadata_JCP2022"].isin(EXCLUDE_COMPOUNDS)]
        counts = df["Metadata_JCP2022"].value_counts()
        high_replicate = set(counts[counts > MAX_WELLS_PER_COMPOUND].index)
        df = df[~df["Metadata_JCP2022"].isin(high_replicate)]
        df = subsample_cpg(df, set(df["Metadata_JCP2022"].unique()))
        eligible = {
            compound
            for compound, group in df.groupby("Metadata_JCP2022")
            if _has_match_at_all_stringencies(group)
        }
        per_model.append(eligible)
        print(
            f"    {MODEL_DISPLAY[model]}: {len(eligible)} eligible compounds "
            f"after excluding {len(high_replicate)} high-replicate controls",
            flush=True,
        )
    shared = sorted(set.intersection(*per_model))
    print(f"    Shared across all models: {len(shared)} compounds", flush=True)
    return shared


def load_selected_cpg_compounds() -> list[str]:
    path = Path(args.selection_pickle)
    if path.exists():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        selected = sorted(payload["selected_compounds"])
        print(f"Loaded {len(selected)} cpg-compound selected compounds from {path}", flush=True)
        return selected
    print(f"Selection pickle not found at {path}; recomputing eligibility.", flush=True)
    return cpg_compound_eligible()


def _build_cmp_to_idx(compounds: np.ndarray) -> dict[str, np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for idx, compound in enumerate(compounds):
        buckets.setdefault(str(compound), []).append(idx)
    return {compound: np.asarray(indices, dtype=np.int64) for compound, indices in buckets.items()}


def _choose_queries(compounds: np.ndarray, cmp_to_idx: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    rng = np.random.RandomState(SEED)
    query_compounds = sorted(
        compound
        for compound, indices in cmp_to_idx.items()
        if compound != DMSO_JCP and len(indices) >= 2
    )
    query_idx = np.asarray([rng.choice(cmp_to_idx[compound]) for compound in query_compounds], dtype=np.int64)
    return query_compounds, query_idx


def _build_relevant_by_query(
    compounds: np.ndarray,
    batches: np.ndarray,
    sources: np.ndarray,
    wells: np.ndarray,
    cmp_to_idx: dict[str, np.ndarray],
    query_compounds: list[str],
    query_idx: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    rels = {paradigm: [] for paradigm in PARADIGMS}
    for compound, qidx in zip(query_compounds, query_idx, strict=True):
        candidates = cmp_to_idx[compound]
        rels["No Restriction"].append(candidates[candidates != qidx])
        rels["Not Same Batch"].append(candidates[batches[candidates] != batches[qidx]])
        rels["Not Same Source"].append(candidates[sources[candidates] != sources[qidx]])
        rels["Not Same Layout"].append(candidates[wells[candidates] != wells[qidx]])
        rels["Not Same Source Layout"].append(
            candidates[(sources[candidates] != sources[qidx]) & (wells[candidates] != wells[qidx])]
        )
    return rels


class DmsoPoolCache:
    def __init__(self, batches: np.ndarray, sources: np.ndarray, wells: np.ndarray):
        self.batches = batches
        self.sources = sources
        self.wells = wells
        all_idx = np.arange(len(batches), dtype=np.int64)
        self.cache: dict[str, dict[Any, np.ndarray]] = {
            "No Restriction": {None: all_idx},
            "Not Same Batch": {},
            "Not Same Source": {},
            "Not Same Layout": {},
            "Not Same Source Layout": {},
        }

    def get(self, paradigm: str, batch: Any, source: Any, well: Any) -> np.ndarray:
        if paradigm == "No Restriction":
            return self.cache[paradigm][None]
        if paradigm == "Not Same Batch":
            key = batch
            if key not in self.cache[paradigm]:
                self.cache[paradigm][key] = np.where(self.batches != batch)[0]
            return self.cache[paradigm][key]
        if paradigm == "Not Same Source":
            key = source
            if key not in self.cache[paradigm]:
                self.cache[paradigm][key] = np.where(self.sources != source)[0]
            return self.cache[paradigm][key]
        if paradigm == "Not Same Layout":
            key = well
            if key not in self.cache[paradigm]:
                self.cache[paradigm][key] = np.where(self.wells != well)[0]
            return self.cache[paradigm][key]
        if paradigm == "Not Same Source Layout":
            key = (source, well)
            if key not in self.cache[paradigm]:
                self.cache[paradigm][key] = np.where((self.sources != source) & (self.wells != well))[0]
            return self.cache[paradigm][key]
        raise ValueError(f"Unknown paradigm: {paradigm}")


def run_negative_control_map(
    Xn: np.ndarray,
    compounds: np.ndarray,
    batches: np.ndarray,
    sources: np.ndarray,
    wells: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    cmp_to_idx = _build_cmp_to_idx(compounds)
    query_compounds, query_idx = _choose_queries(compounds, cmp_to_idx)
    rels_by_query = _build_relevant_by_query(
        compounds,
        batches,
        sources,
        wells,
        cmp_to_idx,
        query_compounds,
        query_idx,
    )

    dmso_idx = np.where(compounds == DMSO_JCP)[0]
    if len(dmso_idx) == 0:
        raise ValueError("No DMSO wells available after filtering/QC.")
    dmso_X = Xn[dmso_idx]
    dmso_cache = DmsoPoolCache(batches[dmso_idx], sources[dmso_idx], wells[dmso_idx])

    aps = {paradigm: [] for paradigm in PARADIGMS}
    neg_counts = {paradigm: [] for paradigm in PARADIGMS}
    rel_counts = {paradigm: [] for paradigm in PARADIGMS}
    skipped_no_relevant = {paradigm: 0 for paradigm in PARADIGMS}
    skipped_no_dmso = {paradigm: 0 for paradigm in PARADIGMS}

    n_queries = len(query_idx)
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk_query_idx = query_idx[start:end]
        qX = Xn[chunk_query_idx]
        dmso_sims = qX @ dmso_X.T

        rel_union_parts = []
        for paradigm in PARADIGMS:
            rel_union_parts.extend(rels_by_query[paradigm][start:end])
        if rel_union_parts:
            rel_union = np.unique(np.concatenate([part for part in rel_union_parts if len(part)]))
        else:
            rel_union = np.array([], dtype=np.int64)
        if len(rel_union):
            rel_sims_all = qX @ Xn[rel_union].T
            rel_col = {int(idx): pos for pos, idx in enumerate(rel_union)}
        else:
            rel_sims_all = np.empty((len(chunk_query_idx), 0), dtype=np.float32)
            rel_col = {}

        for local_i, global_qi in enumerate(range(start, end)):
            qidx = query_idx[global_qi]
            for paradigm in PARADIGMS:
                rel_idx = rels_by_query[paradigm][global_qi]
                if len(rel_idx) == 0:
                    skipped_no_relevant[paradigm] += 1
                    continue
                dmso_pool = dmso_cache.get(paradigm, batches[qidx], sources[qidx], wells[qidx])
                if len(dmso_pool) == 0:
                    skipped_no_dmso[paradigm] += 1
                    continue

                neg_sorted = np.sort(-dmso_sims[local_i, dmso_pool])
                rel_columns = [rel_col[int(idx)] for idx in rel_idx]
                rel_sims = rel_sims_all[local_i, rel_columns]
                aps[paradigm].append(_ap_from_negative_controls(neg_sorted, rel_sims))
                neg_counts[paradigm].append(int(len(dmso_pool)))
                rel_counts[paradigm].append(int(len(rel_idx)))

        if end == n_queries or end % max(chunk_size * 10, 1) == 0:
            print(f"    processed {end}/{n_queries} query compounds", flush=True)

    results = {}
    for paradigm in PARADIGMS:
        vals = aps[paradigm]
        results[paradigm] = {
            "mAP": float(np.mean(vals) * 100.0) if vals else 0.0,
            "mAP_percent": float(np.mean(vals) * 100.0) if vals else 0.0,
            "n_queries": int(len(vals)),
            "n_query_compounds_total": int(n_queries),
            "n_negative_control_candidates": _summary(neg_counts[paradigm]),
            "n_relevant_replicates": _summary(rel_counts[paradigm]),
            "n_skipped_no_relevant": int(skipped_no_relevant[paradigm]),
            "n_skipped_no_dmso": int(skipped_no_dmso[paradigm]),
        }
    return {
        "paradigm_results": results,
        "n_query_compounds_total": int(n_queries),
        "n_dmso_wells": int(len(dmso_idx)),
    }


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if not column.startswith("Metadata_")]


def prepare_tgt2_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def prepare_cpg_compound_df(df: pd.DataFrame, selected_set: set[str]) -> pd.DataFrame:
    dmso = df[df["Metadata_JCP2022"] == DMSO_JCP].copy()
    selected = subsample_cpg(df[df["Metadata_JCP2022"] != DMSO_JCP], selected_set)
    return pd.concat([selected, dmso], ignore_index=True)


def run_dataset(dataset_name: str, selected_set: set[str] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    dataset_dir = os.path.join(FEATURES_BASE, dataset_name)
    dataset_results: dict[str, Any] = {}
    well_counts: dict[str, int] = {}

    print(f"\n{'=' * 80}", flush=True)
    print(f"{dataset_name} negative-control mAP", flush=True)
    print("=" * 80, flush=True)

    for model, fname in MODELS.items():
        t0 = time.time()
        print(f"\n  {MODEL_DISPLAY[model]}", flush=True)
        df = pd.read_parquet(os.path.join(dataset_dir, fname))
        df = maybe_apply_cell_count_qc(df, dataset_name)
        if dataset_name == "cpg-compound":
            if selected_set is None:
                raise ValueError("cpg-compound requires selected compounds.")
            df = prepare_cpg_compound_df(df, selected_set)
        else:
            df = prepare_tgt2_df(df)

        well_counts[model] = int(len(df))
        n_dmso = int((df["Metadata_JCP2022"] == DMSO_JCP).sum())
        n_non_dmso = int((df["Metadata_JCP2022"] != DMSO_JCP).sum())
        print(f"    wells={len(df)} non-DMSO={n_non_dmso} DMSO={n_dmso}", flush=True)

        feat_cols = _feature_columns(df)
        Xn = _l2_normalize(df[feat_cols].to_numpy())
        result = run_negative_control_map(
            Xn,
            df["Metadata_JCP2022"].to_numpy(),
            df["Metadata_Batch"].to_numpy(),
            df["Metadata_Source"].to_numpy(),
            df["Metadata_Well"].to_numpy(),
            chunk_size=args.chunk_size,
        )
        dataset_results[model] = {
            **result["paradigm_results"],
            "model_display": MODEL_DISPLAY[model],
            "n_wells": int(len(df)),
            "n_non_dmso_wells": n_non_dmso,
            "n_dmso_wells": result["n_dmso_wells"],
            "n_query_compounds_total": result["n_query_compounds_total"],
        }

        for paradigm in PARADIGMS:
            metrics = dataset_results[model][paradigm]
            print(
                f"    {paradigm}: mAP={metrics['mAP_percent']:.6f}% "
                f"({metrics['n_queries']}q, DMSO mean={metrics['n_negative_control_candidates']['mean']:.1f})",
                flush=True,
            )
        print(f"    elapsed={time.time() - t0:.1f}s", flush=True)

    return dataset_results, well_counts


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    selected_compounds = load_selected_cpg_compounds()
    selected_set = set(selected_compounds)
    print(f"Selected cpg-compound compounds: {len(selected_compounds)}", flush=True)

    tgt2_results, tgt2_well_counts = run_dataset("cpg-tgt2")
    cpg_results, cpg_well_counts = run_dataset("cpg-compound", selected_set)

    negative_control_candidate_counts = {
        "cpg-tgt2": {
            model: {
                paradigm: metrics["n_negative_control_candidates"]
                for paradigm, metrics in model_results.items()
                if paradigm in PARADIGMS
            }
            for model, model_results in tgt2_results.items()
        },
        "cpg-compound": {
            model: {
                paradigm: metrics["n_negative_control_candidates"]
                for paradigm, metrics in model_results.items()
                if paradigm in PARADIGMS
            }
            for model, model_results in cpg_results.items()
        },
    }

    out = {
        "tgt2_negative_control_map": tgt2_results,
        "cpg_negative_control_map": cpg_results,
        "paradigms": PARADIGMS,
        "selected_compounds": selected_compounds,
        "cpg_selected_compounds": selected_compounds,
        "n_selected_compounds": len(selected_compounds),
        "models": MODELS,
        "model_display": MODEL_DISPLAY,
        "tgt2_well_counts": tgt2_well_counts,
        "cpg_well_counts": cpg_well_counts,
        "negative_control": {
            "compound_id": DMSO_JCP,
            "role": "DMSO distractor wells",
        },
        "negative_control_candidate_counts": negative_control_candidate_counts,
        "cell_count_qc": {
            "enabled": cell_count_qc_enabled(args),
            "summaries": {
                dataset: mask.attrs.get("cell_count_qc_summary", {})
                for dataset, mask in QC_MASKS.items()
            },
        },
        "setting": {
            "name": "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph",
            "features_base": FEATURES_BASE,
            "results_dir": RESULTS_DIR,
            "benchmark": "negative_control_map",
            "seed": SEED,
        },
        "parameters": {
            "max_wells_per_compound_per_source": MAX_WELLS_PER_COMPOUND_PER_SOURCE,
            "negative_control_compound": DMSO_JCP,
            "excluded_compounds": sorted(EXCLUDE_COMPOUNDS),
            "max_wells_per_compound_positive_control_threshold": MAX_WELLS_PER_COMPOUND,
            "chunk_size": args.chunk_size,
            "selection_pickle": args.selection_pickle,
        },
    }

    pkl_path = os.path.join(RESULTS_DIR, "negative_control_map.pkl")
    with open(pkl_path, "wb") as handle:
        pickle.dump(out, handle)
    print(f"\nResults: {pkl_path}", flush=True)


if __name__ == "__main__":
    main()
