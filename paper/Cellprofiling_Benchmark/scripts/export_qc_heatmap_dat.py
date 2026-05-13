#!/usr/bin/env python3
"""Export QC benchmark result pickles to PGFPlots heatmap .dat files."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "normalization_sweep"
OUT_DIR = PAPER_DIR / "data" / "qc"

PIPELINES = {
    "csall-plate-pca64-madctrl-plate-nosph": "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph",
    "csall-plate-pca64-madctrl-plate-sphctrl-batch": "QC_CSAll_Plate__PCA64__MADCtrl_Plate__SphCtrl_Batch",
    "nocs-pca64-madctrl-plate-nosph": "QC_NoCS__PCA64__MADCtrl_Plate__NoSph",
    "csall-plate-nopca-madctrl-plate-nosph": "QC_CSAll_Plate__NoPCA__MADCtrl_Plate__NoSph",
}

CPG_MODELS = [
    "cellprofiler", "cloome", "dino_v2_cls", "dino_v2_patch",
    "open_phenom", "resnet", "resnet_untrained", "subcell",
]
TOKEN_MODELS = [
    "cellprofiler", "cloome", "dino_v2_cls_token", "dino_v2_patch_token",
    "open_phenom", "resnet", "resnet_untrained", "subcell",
]
MOA_CPG_COLS = ["CPJump-Global", "CPJump-Not Same Batch", "CPJump-Not Same Source"]

BBBC_TO_CANONICAL = {
    "dino_v2_cls": "dino_v2_cls_token",
    "dino_v2_patch": "dino_v2_patch_token",
    "open_phenom": "open_phenom",
}

CRISPR_DBS = ["CORUM", "HuMAP", "REACTOME", "SIGNOR", "StringDB"]

KNN_PARADIGMS = ["No Restriction", "Not Same Batch", "Not Same Source", "Not Same Layout"]
CPG_ALL_ELIGIBLE_DIRS = {
    slug: f"{dirname}_cpg_all_eligible"
    for slug, dirname in PIPELINES.items()
}

DISPLAY_NAMES = {
    "cellprofiler": "CellProfiler",
    "cloome": "CLOOME",
    "dino_v2_cls": "DINOv2 CLS",
    "dino_v2_cls_token": "DINOv2 CLS",
    "dino_v2_patch": "DINOv2 Patch",
    "dino_v2_patch_token": "DINOv2 Patch",
    "open_phenom": "OpenPhenom",
    "resnet": "ResNet",
    "resnet_untrained": "{ResNet\\\\(Untrained)}",
    "subcell": "SubCell",
}

LABEL_MACROS: dict[tuple[str, str], str] = {}


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


def mean_score(values: list[float]) -> float:
    finite_values = [v for v in values if not math.isnan(v)]
    if not finite_values:
        return float("-inf")
    return sum(finite_values) / len(finite_values)


def ranked_rows(models: list[str], value_fn) -> list[tuple[str, list[float]]]:
    rows = [(model, value_fn(model)) for model in models]
    return sorted(rows, key=lambda item: (-mean_score(item[1]), DISPLAY_NAMES[item[0]]))


def ranked_rows_by_column(models: list[str], value_fn, column_index: int) -> list[tuple[str, list[float]]]:
    rows = [(model, value_fn(model)) for model in models]
    return sorted(rows, key=lambda item: (-item[1][column_index], DISPLAY_NAMES[item[0]]))


def labels_for(models: list[str]) -> str:
    return ", ".join(DISPLAY_NAMES[model] for model in models)


def remember_labels(slug: str, suffix: str, models: list[str]) -> None:
    LABEL_MACROS[(slug, suffix)] = labels_for(models)


def geometric_mean(values: list[float]) -> float:
    values = [v for v in values if v > 0 and not math.isnan(v)]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bbbc_key(canonical_model: str) -> str:
    return BBBC_TO_CANONICAL.get(canonical_model, canonical_model)


def export_moa(slug: str, result_dir: Path) -> None:
    cpg = load_pickle(result_dir / "moa_enrichment_results.pkl")["summary"]
    bbbc = load_pickle(result_dir / "bbbc036_moa_enrichment.pkl")

    fs_bbbc_rows = ranked_rows(
        CPG_MODELS,
        lambda model: [float(bbbc[bbbc_key(model)]["fraction_significant"])] * 2,
    )
    or_bbbc_rows = ranked_rows(
        CPG_MODELS,
        lambda model: [float(bbbc[bbbc_key(model)]["geometric_mean_or"])] * 2,
    )
    fs_cpg_rows = ranked_rows(
        CPG_MODELS,
        lambda model: [float(cpg[(model, col)]["frac_sig"]) for col in MOA_CPG_COLS],
    )
    or_cpg_rows = ranked_rows(
        CPG_MODELS,
        lambda model: [float(cpg[(model, col)]["geom_or"]) for col in MOA_CPG_COLS],
    )

    fs_bbbc = [values for _, values in fs_bbbc_rows]
    or_bbbc = [values for _, values in or_bbbc_rows]
    fs_cpg = [values for _, values in fs_cpg_rows]
    or_cpg = [values for _, values in or_cpg_rows]

    remember_labels(slug, "moa-fs-bbbc036-2col", [model for model, _ in fs_bbbc_rows])
    remember_labels(slug, "moa-or-bbbc036-2col", [model for model, _ in or_bbbc_rows])
    remember_labels(slug, "moa-fs-cpgmoa", [model for model, _ in fs_cpg_rows])
    remember_labels(slug, "moa-or-cpgmoa", [model for model, _ in or_cpg_rows])

    write_matrix(OUT_DIR / f"{slug}-moa-fs-bbbc036-2col.dat", fs_bbbc)
    write_matrix(OUT_DIR / f"{slug}-moa-or-bbbc036-2col.dat", or_bbbc)
    write_matrix(OUT_DIR / f"{slug}-moa-fs-cpgmoa.dat", fs_cpg)
    write_matrix(OUT_DIR / f"{slug}-moa-or-cpgmoa.dat", or_cpg)


def summarize_crispr(raw: dict, model: str, metric: str) -> float:
    values = [float(raw[model][db][metric]) for db in CRISPR_DBS if db in raw[model]]
    if metric == "geometric_mean_odds":
        return geometric_mean(values)
    return sum(values) / len(values)


def export_crispr(slug: str, result_dir: Path) -> None:
    nr = load_pickle(result_dir / "crispr_enrichment_no_restriction.pkl")
    nsb = load_pickle(result_dir / "crispr_enrichment_not_same_batch.pkl")

    fs_ranked = ranked_rows_by_column(
        TOKEN_MODELS,
        lambda model: [
            summarize_crispr(nr, model, "fraction_significant"),
            summarize_crispr(nsb, model, "fraction_significant"),
        ],
        0,
    )
    or_ranked = ranked_rows_by_column(
        TOKEN_MODELS,
        lambda model: [
            summarize_crispr(nr, model, "geometric_mean_odds"),
            summarize_crispr(nsb, model, "geometric_mean_odds"),
        ],
        0,
    )

    fs_rows = [values for _, values in fs_ranked]
    or_rows = [values for _, values in or_ranked]

    remember_labels(slug, "crispr-fs", [model for model, _ in fs_ranked])
    remember_labels(slug, "crispr-or", [model for model, _ in or_ranked])

    write_matrix(OUT_DIR / f"{slug}-crispr-fs.dat", fs_rows)
    write_matrix(OUT_DIR / f"{slug}-crispr-or.dat", or_rows)


def export_knn(slug: str, result_dir: Path, cpg_override_dir: str | None = None) -> None:
    knn = load_pickle(result_dir / "knn_results.pkl")
    override_dirname = cpg_override_dir or CPG_ALL_ELIGIBLE_DIRS[slug]
    cpg_override = load_pickle(
        RESULTS_DIR / override_dirname / "knn_cpg_compound_all_eligible.pkl"
    )

    outputs = [
        ("knn-recall-tgt2", "tgt2_knn", "recall"),
        ("knn-recall-compound", "cpg_knn", "recall"),
        ("knn-map-tgt2", "tgt2_map", "mAP"),
        ("knn-map-compound", "cpg_map", "mAP"),
    ]
    for suffix, result_key, metric in outputs:
        result_data = cpg_override if cpg_override is not None and result_key.startswith("cpg_") else knn
        value_fn = lambda model: [
            float(result_data[result_key][model][paradigm][metric])
            for paradigm in KNN_PARADIGMS
        ]
        if suffix == "knn-map-compound":
            ranked = ranked_rows_by_column(TOKEN_MODELS, value_fn, 0)
        else:
            ranked = ranked_rows(TOKEN_MODELS, value_fn)
        rows = [values for _, values in ranked]
        remember_labels(slug, suffix, [model for model, _ in ranked])
        write_matrix(OUT_DIR / f"{slug}-{suffix}.dat", rows)


def export_negative_control_map(slug: str, dirname: str) -> None:
    result_path = RESULTS_DIR / f"{dirname}_negative_control_map" / "negative_control_map.pkl"
    negative = load_pickle(result_path)
    paradigms = ["No Restriction", "Not Same Batch", "Not Same Source", "Not Same Layout"]
    sections = [
        ("negative-control-map-tgt2", "tgt2_negative_control_map"),
        ("negative-control-map-compound", "cpg_negative_control_map"),
    ]

    for suffix, result_key in sections:
        value_fn = lambda model: [
            float(negative[result_key][model][paradigm]["mAP_percent"])
            for paradigm in paradigms
        ]
        if suffix == "negative-control-map-compound":
            ranked = ranked_rows_by_column(TOKEN_MODELS, value_fn, 0)
        else:
            ranked = ranked_rows(TOKEN_MODELS, value_fn)
        rows = [values for _, values in ranked]
        remember_labels(slug, suffix, [model for model, _ in ranked])
        write_matrix(OUT_DIR / f"{slug}-{suffix}.dat", rows)


def parse_pipeline_mapping(values: list[str] | None) -> dict[str, str]:
    if not values:
        return {}
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Pipeline mapping must be slug=result_dir, got: {value}")
        slug, dirname = value.split("=", 1)
        slug = slug.strip()
        dirname = dirname.strip()
        if not slug or not dirname:
            raise ValueError(f"Invalid pipeline mapping: {value}")
        parsed[slug] = dirname
    return parsed


def write_label_macros(merge_existing: bool = False) -> None:
    path = OUT_DIR / "qc_heatmap_labels.tex"
    existing_lines: list[str] = []
    current_keys = {
        f"qclabels@{slug}@{suffix}"
        for slug, suffix in LABEL_MACROS
    }
    if merge_existing and path.exists():
        for line in path.read_text().splitlines():
            if any(key in line for key in current_keys):
                continue
            if line.startswith("% Auto-generated"):
                continue
            if line.strip():
                existing_lines.append(line)

    with path.open("w") as handle:
        handle.write("% Auto-generated by scripts/export_qc_heatmap_dat.py. Do not edit by hand.\n")
        for line in existing_lines:
            handle.write(f"{line}\n")
        for (slug, suffix), labels in sorted(LABEL_MACROS.items()):
            handle.write(
                f"\\expandafter\\def\\csname qclabels@{slug}@{suffix}\\endcsname"
                f"{{{labels}}}\n"
            )


def main() -> None:
    global RESULTS_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="Export benchmark result pickles to PGFPlots heatmap .dat files.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Root directory containing normalization_sweep-style result folders.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="Directory where .dat files and qc_heatmap_labels.tex are written.")
    parser.add_argument("--only-main", action="store_true",
                        help="Export only the main-paper QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph pipeline.")
    parser.add_argument("--include-negative-control", action="store_true",
                        help="Also export negative-control mAP files in --only-main mode.")
    parser.add_argument("--merge-labels", action="store_true",
                        help="Preserve unrelated label macros already present in qc_heatmap_labels.tex.")
    parser.add_argument("--pipeline", action="append",
                        help="Pipeline mapping as slug=result_dir. May be repeated. Overrides built-in mappings.")
    parser.add_argument("--negative-control-pipeline",
                        help="Pipeline mapping as slug=result_dir for negative-control mAP export.")
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    OUT_DIR = args.out_dir

    pipeline_override = parse_pipeline_mapping(args.pipeline)
    pipelines = pipeline_override or PIPELINES
    if args.only_main:
        main_slug = "csall-plate-pca64-madctrl-plate-nosph"
        pipelines = {main_slug: pipelines.get(main_slug, PIPELINES[main_slug])}

    for slug, dirname in pipelines.items():
        result_dir = RESULTS_DIR / dirname
        if not result_dir.exists():
            raise FileNotFoundError(result_dir)
        export_moa(slug, result_dir)
        export_crispr(slug, result_dir)
        export_knn(slug, result_dir, cpg_override_dir=f"{dirname}_cpg_all_eligible")
        print(f"Exported QC heatmap data for {dirname}")
    if args.include_negative_control or (not args.only_main and not pipeline_override):
        negative_mapping = parse_pipeline_mapping([args.negative_control_pipeline]) if args.negative_control_pipeline else {}
        if negative_mapping:
            negative_slug, negative_dirname = next(iter(negative_mapping.items()))
        else:
            negative_slug = "csall-plate-pca64-madctrl-plate-nosph"
            negative_dirname = pipelines.get(negative_slug, PIPELINES[negative_slug])
        export_negative_control_map(
            negative_slug,
            negative_dirname,
        )
    write_label_macros(merge_existing=args.merge_labels)


if __name__ == "__main__":
    main()
