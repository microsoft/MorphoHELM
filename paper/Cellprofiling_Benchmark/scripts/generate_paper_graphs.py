#!/usr/bin/env python3
"""Config-driven regeneration of paper benchmark results and graph inputs."""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "paper" / "Cellprofiling_Benchmark" / "configs" / "paper_graphs.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_path(value: str | Path, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_normalize_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalize_config = resolve_path(config["paths"]["normalize_config"])
    raw = load_yaml(normalize_config)
    return raw.get("profiles", {})


def selected_profile_names(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    normalization = config["normalization"]
    if args.profile:
        return args.profile
    if args.profiles == "all":
        return [normalization["main_profile"], *normalization.get("appendix_profiles", [])]
    if args.profiles == "main" or args.profiles is None:
        return [normalization["main_profile"]]
    parsed = parse_csv(args.profiles)
    if parsed:
        return parsed
    raise ValueError(f"Could not parse --profiles={args.profiles!r}")


def profile_info(profile_name: str, profiles: dict[str, dict[str, Any]]) -> dict[str, str]:
    if profile_name not in profiles:
        raise ValueError(f"Unknown normalization profile {profile_name!r}. Available: {sorted(profiles)}")
    profile = profiles[profile_name]
    return {
        "name": profile_name,
        "slug": profile.get("slug", profile_name.lower().replace("__", "-").replace("_", "-")),
        "result_dir": profile.get("result_dir", f"QC_{profile_name}"),
    }


def qc_args(config: dict[str, Any]) -> list[str]:
    qc = config.get("qc", {})
    if not qc.get("enabled", False):
        return []
    return [
        "--cell-count-qc",
        "--qc-cell-counts-dir",
        str(resolve_path(config["paths"]["qc_cell_counts_dir"])),
        "--qc-cell-count-cutoff",
        str(qc.get("cell_count_cutoff", 0.05)),
        "--qc-sample-wells",
        str(qc.get("sample_wells", 5000)),
        "--qc-seed",
        str(qc.get("seed", 42)),
    ]


def write_runtime_normalize_config(config: dict[str, Any], results_root: Path) -> Path:
    normalize_config_path = resolve_path(config["paths"]["normalize_config"])
    normalize_cfg = load_yaml(normalize_config_path)
    normalize_cfg.setdefault("execution", {})["mode"] = "local"
    normalize_cfg["aggregated_root"] = {"local": str(resolve_path(config["paths"]["aggregated_root"]))}
    normalize_cfg["normalized_root"] = {"local": str(resolve_path(config["paths"]["normalized_root"]))}
    normalize_cfg["metadata_root"] = {"local": str(ROOT)}
    out_path = results_root / "runtime_normalize_splits.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        yaml.safe_dump(normalize_cfg, handle, sort_keys=False)
    return out_path


def validate_normalized_inputs(profile: dict[str, str], config: dict[str, Any], normalized_dir: Path | None = None) -> Path:
    normalized_dir = normalized_dir or resolve_path(config["paths"]["normalized_root"]) / profile["name"]
    missing: list[Path] = []
    for split in config["data"]["splits"]:
        for model in config["data"]["models"]:
            path = normalized_dir / split / f"{model}_normalized.parquet"
            if not path.exists():
                missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
        raise FileNotFoundError(
            f"Missing normalized inputs for profile {profile['name']}:\n{preview}{extra}\n"
            "Run with --run-normalization or point paths.normalized_root at existing outputs."
        )
    return normalized_dir


def run_normalization(profile: dict[str, str], config: dict[str, Any], results_root: Path, dry_run: bool) -> None:
    runtime_config = write_runtime_normalize_config(config, results_root)
    run_cmd(
        [
            sys.executable,
            str(ROOT / "data-preprocessing" / "src" / "normalize_splits.py"),
            "--config",
            str(runtime_config),
            "--profile",
            profile["name"],
            "--splits",
            ",".join(config["data"]["splits"]),
            "--model",
            "all",
        ],
        dry_run=dry_run,
    )


def run_benchmarks(profile: dict[str, str], features_dir: Path, config: dict[str, Any], dry_run: bool) -> None:
    paths = config["paths"]
    results_root = resolve_path(paths["results_root"])
    result_dir = results_root / profile["result_dir"]
    result_dir.mkdir(parents=True, exist_ok=True)
    n_resamples = str(config["benchmarks"].get("n_resamples", 100))
    run_flags = config.get("benchmarks", {}).get("run", {})

    enrichment = []
    if run_flags.get("crispr", True):
        enrichment.append("crispr")
    if run_flags.get("cpg_moa", run_flags.get("moa", True)):
        enrichment.append("cpg_moa")
    if run_flags.get("bbbc036_moa", run_flags.get("moa", True)):
        enrichment.append("bbbc036_moa")
    if enrichment:
        run_cmd(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "enrichment" / "run_enrichment_sweep.py"),
                "--config",
                str(config["_config_path"]),
                "--profile",
                profile["name"],
                "--benchmarks",
                ",".join(enrichment),
                "--features-base",
                str(features_dir),
                "--results-dir",
                str(result_dir),
                "--n-resamples",
                n_resamples,
            ],
            dry_run=dry_run,
        )

    replicate = []
    if run_flags.get("knn_replicate", run_flags.get("knn", True)):
        replicate.append("knn_replicate")
    if run_flags.get("map", run_flags.get("knn", True)):
        replicate.append("map")
    if run_flags.get("negcon_map", run_flags.get("negative_control_map", True)):
        replicate.append("negcon_map")
    if replicate:
        run_cmd(
            [
                sys.executable,
                str(ROOT / "benchmarks" / "replicate_analysis" / "run_replicate_analysis_sweep.py"),
                "--config",
                str(config["_config_path"]),
                "--profile",
                profile["name"],
                "--benchmarks",
                ",".join(replicate),
                "--features-base",
                str(features_dir),
                "--results-dir",
                str(result_dir),
            ],
            dry_run=dry_run,
        )


def export_graphs(profiles: list[dict[str, str]], config: dict[str, Any], dry_run: bool) -> None:
    paths = config["paths"]
    results_root = resolve_path(paths["results_root"])
    paper_dir = resolve_path(paths["paper_dir"])
    exports = config.get("exports", {})
    main_profile = config["normalization"]["main_profile"]
    main = next((profile for profile in profiles if profile["name"] == main_profile), profiles[0])

    if exports.get("qc_heatmaps", True):
        pipeline_args = []
        for profile in profiles:
            pipeline_args.extend(["--pipeline", f"{profile['slug']}={profile['result_dir']}"])
        run_cmd(
            [
                sys.executable,
                str(ROOT / "paper" / "Cellprofiling_Benchmark" / "scripts" / "export_qc_heatmap_dat.py"),
                "--results-dir",
                str(results_root),
                "--out-dir",
                str(paper_dir / "data" / "qc"),
                "--merge-labels",
                "--include-negative-control",
                "--negative-control-pipeline",
                f"{main['slug']}={main['result_dir']}",
                *pipeline_args,
            ],
            dry_run=dry_run,
        )

    if exports.get("jaccard", True):
        run_cmd(
            [
                sys.executable,
                str(ROOT / "paper" / "Cellprofiling_Benchmark" / "scripts" / "export_jaccard_dat.py"),
                "--results-root",
                str(results_root),
                "--pipeline-name",
                main["result_dir"],
                "--out-dir",
                str(paper_dir / "data" / "jaccard"),
            ],
            dry_run=dry_run,
        )

    if exports.get("crispr_database_barplot", True):
        run_cmd(
            [
                sys.executable,
                str(ROOT / "paper" / "Cellprofiling_Benchmark" / "scripts" / "plot_crispr_database_barplot.py"),
                "--result-dir",
                str(results_root / main["result_dir"]),
                "--out-dir",
                str(paper_dir / "graphs" / "Results" / "CRISPR_Enrichment"),
            ],
            dry_run=dry_run,
        )

    if exports.get("haldane_diagnostic", True):
        haldane = paths.get("haldane", {})
        old_path = resolve_path(haldane["old_path"])
        haldane_path = resolve_path(haldane["haldane_path"])
        if old_path.exists() and haldane_path.exists():
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "benchmarks" / "enrichment" / "moa" / "plot_moa_pooled_or_skewness.py"),
                    "--old-path",
                    str(old_path),
                    "--haldane-path",
                    str(haldane_path),
                    "--paper-out-dir",
                    str(paper_dir / "all_graphs" / "MoA" / "BBBC036"),
                    "--out-dir",
                    str(results_root / "haldane_diagnostic"),
                ],
                dry_run=dry_run,
            )
        else:
            print(f"Skipping Haldane diagnostic; missing {old_path} or {haldane_path}", file=sys.stderr)


def make_mock_results(results_root: Path, profiles: list[dict[str, str]]) -> None:
    cpg_models = [
        "cellprofiler", "cloome", "dino_v2_cls", "dino_v2_patch",
        "open_phenom", "resnet", "resnet_untrained", "subcell",
    ]
    token_models = [
        "cellprofiler", "cloome", "dino_v2_cls_token", "dino_v2_patch_token",
        "open_phenom", "resnet", "resnet_untrained", "subcell",
    ]
    moa_key = {
        "open_phenom": "openphenom",
        "dino_v2_cls": "dino_v2",
        "dino_v2_patch": "dino_v2_patch",
    }
    moa_cols = ["CPJump-Global", "CPJump-Not Same Batch", "CPJump-Not Same Source"]
    dbs = ["CORUM", "HuMAP", "REACTOME", "SIGNOR", "StringDB"]
    paradigms = ["No Restriction", "Not Same Batch", "Not Same Source", "Not Same Layout"]

    for profile in profiles:
        result_dir = results_root / profile["result_dir"]
        override_dir = results_root / f"{profile['result_dir']}_cpg_all_eligible"
        negative_dir = results_root / f"{profile['result_dir']}_negative_control_map"
        result_dir.mkdir(parents=True, exist_ok=True)
        override_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)

        summary = {}
        for i, model in enumerate(cpg_models):
            for j, col in enumerate(moa_cols):
                summary[(model, col)] = {"frac_sig": 0.02 + 0.004 * i + 0.001 * j, "geom_or": 8.0 + 0.2 * i}
        global_moa = {
            moa_key.get(model, model): {
                "processed_compounds": [f"cmp_{idx}" for idx in range(8)],
                "significant_compounds": [f"cmp_{idx}" for idx in range(i % 5)],
                "all_pvals": [0.01 if idx < i % 5 else 0.5 for idx in range(8)],
            }
            for i, model in enumerate(cpg_models)
        }
        (result_dir / "moa_enrichment_results.pkl").write_bytes(
            pickle.dumps({"summary": summary, "global": global_moa})
        )

        bbbc = {
            model: {
                "fraction_significant": 0.04 + 0.003 * i,
                "geometric_mean_or": 9.0 + 0.15 * i,
                "processed_compounds": [f"bbbc_{idx}" for idx in range(8)],
                "significant_compounds": [f"bbbc_{idx}" for idx in range((i + 1) % 5)],
                "all_pvals": [0.01 if idx < (i + 1) % 5 else 0.5 for idx in range(8)],
            }
            for i, model in enumerate(token_models)
        }
        (result_dir / "bbbc036_moa_enrichment.pkl").write_bytes(pickle.dumps(bbbc))

        for mode, offset in [("no_restriction", 0.0), ("not_same_batch", -0.01)]:
            crispr = {}
            for i, model in enumerate(token_models):
                crispr[model] = {}
                for j, db in enumerate(dbs):
                    crispr[model][db] = {
                        "fraction_significant": max(0.0, 0.03 + 0.003 * i + 0.001 * j + offset),
                        "geometric_mean_odds": 11.0 + 0.2 * i + 0.05 * j + offset,
                        "significant_genes": [f"gene_{idx}" for idx in range((i + j) % 5)],
                    }
            (result_dir / f"crispr_enrichment_{mode}.pkl").write_bytes(pickle.dumps(crispr))

        def knn_payload() -> dict[str, Any]:
            out = {"tgt2_knn": {}, "tgt2_map": {}, "cpg_knn": {}, "cpg_map": {}}
            for i, model in enumerate(token_models):
                for section in out.values():
                    section[model] = {}
                for j, paradigm in enumerate(paradigms):
                    out["tgt2_knn"][model][paradigm] = {"recall": 5.0 + i + 0.1 * j}
                    out["tgt2_map"][model][paradigm] = {"mAP": 1.0 + 0.2 * i + 0.05 * j}
                    out["cpg_knn"][model][paradigm] = {"recall": 0.5 + 0.2 * i + 0.02 * j}
                    out["cpg_map"][model][paradigm] = {"mAP": 0.1 + 0.02 * i + 0.005 * j}
            return out

        (result_dir / "knn_results.pkl").write_bytes(pickle.dumps(knn_payload()))
        (override_dir / "knn_cpg_compound_all_eligible.pkl").write_bytes(pickle.dumps(knn_payload()))

        negative = {"tgt2_negative_control_map": {}, "cpg_negative_control_map": {}}
        for i, model in enumerate(token_models):
            negative["tgt2_negative_control_map"][model] = {}
            negative["cpg_negative_control_map"][model] = {}
            for j, paradigm in enumerate(paradigms):
                negative["tgt2_negative_control_map"][model][paradigm] = {"mAP_percent": 1.0 + i + 0.1 * j}
                negative["cpg_negative_control_map"][model][paradigm] = {"mAP_percent": 0.1 + 0.2 * i + 0.02 * j}
        (negative_dir / "negative_control_map.pkl").write_bytes(pickle.dumps(negative))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate paper benchmark graphs from normalized embeddings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", action="append", help="Specific normalization profile to run. May be repeated.")
    parser.add_argument("--profiles", default="main", help="'main', 'all', or comma-separated profile names.")
    parser.add_argument("--features-dir", type=Path, help="Override normalized feature directory for a single selected profile.")
    parser.add_argument("--normalized-root", type=Path, help="Override paths.normalized_root from the graph config.")
    parser.add_argument("--results-root", type=Path, help="Override paths.results_root from the graph config.")
    parser.add_argument("--paper-dir", "--final-paper-dir", dest="paper_dir", type=Path,
                        help="Override paths.paper_dir from the graph config.")
    parser.add_argument("--n-resamples", type=int, help="Override benchmarks.n_resamples from the graph config.")
    parser.add_argument("--run-normalization", action="store_true", help="Run normalization before benchmarks.")
    parser.add_argument("--skip-normalization", action="store_true", help="Do not run normalization, even if config enables it.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark recomputation and only export graph inputs.")
    parser.add_argument("--only-export-graphs", action="store_true", help="Alias for --skip-normalization --skip-benchmarks.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--mock", action="store_true", help="Run a tiny synthetic graph-export smoke test.")
    parser.add_argument("--mock-dir", type=Path, default=ROOT / "tmp" / "mock_paper_graphs")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config))
    config["_config_path"] = str(resolve_path(args.config))
    if args.normalized_root:
        config["paths"]["normalized_root"] = str(args.normalized_root)
    if args.results_root:
        config["paths"]["results_root"] = str(args.results_root)
    if args.paper_dir:
        config["paths"]["paper_dir"] = str(args.paper_dir)
    if args.n_resamples is not None:
        config.setdefault("benchmarks", {})["n_resamples"] = args.n_resamples
    normalize_profiles = load_normalize_profiles(config)
    profiles = [profile_info(name, normalize_profiles) for name in selected_profile_names(args, config)]
    results_root = resolve_path(config["paths"]["results_root"])
    if args.features_dir and len(profiles) != 1:
        raise ValueError("--features-dir can only be used when one profile is selected")

    if args.only_export_graphs:
        args.skip_normalization = True
        args.skip_benchmarks = True

    if args.mock:
        if args.mock_dir.exists():
            shutil.rmtree(args.mock_dir)
        mock_results = args.mock_dir / "results"
        mock_paper = args.mock_dir / "paper"
        make_mock_results(mock_results, profiles)
        config["paths"]["results_root"] = str(mock_results)
        config["paths"]["paper_dir"] = str(mock_paper)
        config["exports"]["haldane_diagnostic"] = False
        export_graphs(profiles, config, dry_run=args.dry_run)
        print(f"Mock graph generation succeeded: {args.mock_dir}")
        return

    run_norm = config.get("normalization", {}).get("run_by_default", False)
    if args.run_normalization:
        run_norm = True
    if args.skip_normalization:
        run_norm = False

    for profile in profiles:
        if run_norm:
            if args.features_dir:
                raise ValueError("--features-dir cannot be combined with --run-normalization")
            run_normalization(profile, config, results_root, dry_run=args.dry_run)
        features_dir = resolve_path(args.features_dir) if args.features_dir else resolve_path(config["paths"]["normalized_root"]) / profile["name"]
        if not args.dry_run:
            features_dir = validate_normalized_inputs(profile, config, features_dir)
        if not args.skip_benchmarks:
            run_benchmarks(profile, features_dir, config, dry_run=args.dry_run)

    export_graphs(profiles, config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
