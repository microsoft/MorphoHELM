#!/usr/bin/env python3
"""Export discovery-set Jaccard matrices from benchmark result pickles."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parents[1]
RESULTS_ROOT = PROJECT_DIR / "results" / "normalization_sweep"
OUT_DIR = PAPER_DIR / "data" / "jaccard"
PIPELINE_NAME = "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph"

MODELS = [
    "cellprofiler",
    "cloome",
    "dino_v2_cls_token",
    "dino_v2_patch_token",
    "open_phenom",
    "resnet",
    "resnet_untrained",
    "subcell",
]

CPG_MOA_KEYS = {
    "cellprofiler": "cellprofiler",
    "cloome": "cloome",
    "dino_v2_cls_token": "dino_v2",
    "dino_v2_patch_token": "dino_v2_patch",
    "open_phenom": "openphenom",
    "resnet": "resnet",
    "resnet_untrained": "resnet_untrained",
    "subcell": "subcell",
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_matrix(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("x y C\n")
        for y, row in enumerate(rows):
            for x, value in enumerate(row):
                handle.write(f"{x} {y} {value:.10g}\n")


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def matrix_from_sets(discovery_sets: dict[str, set[str]]) -> list[list[float]]:
    return [
        [jaccard(discovery_sets[left], discovery_sets[right]) for right in MODELS]
        for left in MODELS
    ]


def significant_from_pvalues(result: dict) -> set[str]:
    if "significant_compounds" in result:
        return {str(item) for item in result["significant_compounds"]}
    if "processed_compounds" not in result or "all_pvals" not in result:
        raise KeyError(
            "Result is missing significant_compounds and cannot be reconstructed "
            "from processed_compounds/all_pvals. Re-run the benchmark script."
        )
    return {
        str(compound)
        for compound, pvalue in zip(result["processed_compounds"], result["all_pvals"], strict=True)
        if float(pvalue) < 0.05
    }


def export_bbbc036_moa(result_dir: Path, out_dir: Path) -> None:
    raw = load_pickle(result_dir / "bbbc036_moa_enrichment.pkl")
    discovery_sets = {model: significant_from_pvalues(raw[model]) for model in MODELS}
    write_matrix(out_dir / "bbbc036-moa-jaccard.dat", matrix_from_sets(discovery_sets))


def export_cpg_moa(result_dir: Path, out_dir: Path) -> None:
    raw = load_pickle(result_dir / "moa_enrichment_results.pkl")["global"]
    discovery_sets = {
        model: significant_from_pvalues(raw[CPG_MOA_KEYS[model]])
        for model in MODELS
    }
    write_matrix(out_dir / "cpg-moa-jaccard.dat", matrix_from_sets(discovery_sets))


def export_crispr(result_dir: Path, out_dir: Path) -> None:
    raw = load_pickle(result_dir / "crispr_enrichment_no_restriction.pkl")
    discovery_sets: dict[str, set[str]] = {}
    for model in MODELS:
        genes: set[str] = set()
        for database_result in raw[model].values():
            genes.update(str(gene) for gene in database_result.get("significant_genes", []))
        discovery_sets[model] = genes
    write_matrix(out_dir / "crispr-jaccard.dat", matrix_from_sets(discovery_sets))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pairwise Jaccard matrices for least-restrictive discovery sets."
    )
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--pipeline-name", default=PIPELINE_NAME)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip matrices whose required result pickle is missing.",
    )
    args = parser.parse_args()

    result_dir = args.results_root / args.pipeline_name
    jobs = [
        (result_dir / "bbbc036_moa_enrichment.pkl", export_bbbc036_moa),
        (result_dir / "moa_enrichment_results.pkl", export_cpg_moa),
        (result_dir / "crispr_enrichment_no_restriction.pkl", export_crispr),
    ]
    for required_path, exporter in jobs:
        if not required_path.exists():
            if args.skip_missing:
                print(f"Skipping missing input: {required_path}")
                continue
            raise FileNotFoundError(required_path)
        exporter(result_dir, args.out_dir)
    print(f"Exported Jaccard matrices to {args.out_dir}")


if __name__ == "__main__":
    main()
