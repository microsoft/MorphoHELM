from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmarks.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def enabled_names(run_config: dict[str, Any]) -> list[str]:
    return [name for name, enabled in run_config.items() if bool(enabled)]


def qc_cli_args(config: dict[str, Any]) -> list[str]:
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


def default_profile(config: dict[str, Any]) -> str:
    return str(config.get("profiles", {}).get("default", "CSAll_Plate__PCA64__MADCtrl_Plate__NoSph"))


def profile_features_dir(config: dict[str, Any], profile: str) -> Path:
    return resolve_path(config["paths"]["normalized_root"]) / profile


def profile_results_dir(config: dict[str, Any], profile: str) -> Path:
    return resolve_path(config["paths"]["results_root"]) / profile
