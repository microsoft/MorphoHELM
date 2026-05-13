#!/usr/bin/env python3
"""Normalize aggregated well-level feature parquets for one or more splits."""

import argparse
import copy
import glob
import json
import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from normalize_utils import normalize_single_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s [normalize-splits] %(levelname)s: %(message)s")
logger = logging.getLogger("normalize-splits")


DEFAULT_PROFILES = {
    "CSAll_Plate__PCA64__MADCtrl_Plate__NoSph": {
        "result_dir": "QC_CSAll_Plate__PCA64__MADCtrl_Plate__NoSph",
        "slug": "csall-plate-pca64-madctrl-plate-nosph",
        "centerscale": {"enabled": True, "scope": "plate", "position": "pre_pca", "fit_on": "all"},
        "pca": {"enabled": True, "n_components": 64},
        "mad": {"scope": "plate"},
        "spherize": {"enabled": False},
    },
    "CSAll_Plate__PCA64__MADCtrl_Plate__SphCtrl_Batch": {
        "result_dir": "QC_CSAll_Plate__PCA64__MADCtrl_Plate__SphCtrl_Batch",
        "slug": "csall-plate-pca64-madctrl-plate-sphctrl-batch",
        "centerscale": {"enabled": True, "scope": "plate", "position": "pre_pca", "fit_on": "all"},
        "pca": {"enabled": True, "n_components": 64},
        "mad": {"scope": "plate"},
        "spherize": {"enabled": True, "scope": "batch"},
    },
    "NoCS__PCA64__MADCtrl_Plate__NoSph": {
        "result_dir": "QC_NoCS__PCA64__MADCtrl_Plate__NoSph",
        "slug": "nocs-pca64-madctrl-plate-nosph",
        "centerscale": {"enabled": False, "scope": "plate", "position": "none", "fit_on": "all"},
        "pca": {"enabled": True, "n_components": 64},
        "mad": {"scope": "plate"},
        "spherize": {"enabled": False},
    },
    "CSAll_Plate__NoPCA__MADCtrl_Plate__NoSph": {
        "result_dir": "QC_CSAll_Plate__NoPCA__MADCtrl_Plate__NoSph",
        "slug": "csall-plate-nopca-madctrl-plate-nosph",
        "centerscale": {"enabled": True, "scope": "plate", "position": "pre_pca", "fit_on": "all"},
        "pca": {"enabled": False, "n_components": 0},
        "mad": {"scope": "plate"},
        "spherize": {"enabled": False},
    },
    "CSAll_Plate__PCA8__MADCtrl_Plate__NoSph": {
        "result_dir": "QC_CSAll_Plate__PCA8__MADCtrl_Plate__NoSph",
        "slug": "csall-plate-pca8-madctrl-plate-nosph",
        "centerscale": {"enabled": True, "scope": "plate", "position": "pre_pca", "fit_on": "all"},
        "pca": {"enabled": True, "n_components": 8},
        "mad": {"scope": "plate"},
        "spherize": {"enabled": False},
    },
}

VALID_SCOPES = {"plate", "batch", "source"}
VALID_CENTERSCALE_POSITIONS = {"pre_pca", "post_pca", "none"}
VALID_CENTERSCALE_FIT = {"all", "controls"}


def parse_list(values: Iterable[str]) -> List[str]:
    parsed = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    if not parsed:
        raise ValueError("At least one value is required.")
    return parsed


def _mode_value(value, mode: str):
    if isinstance(value, Mapping) and mode in value:
        return value[mode]
    return value


def _root(cfg: Dict, name: str, mode: str) -> str:
    value = cfg.get(name)
    if value is None:
        raise ValueError(f"Missing required config key: {name}")
    value = _mode_value(value, mode)
    if value is None:
        raise ValueError(f"Missing {mode} value for config key: {name}")
    return value


def resolve_profile(cfg: Dict, profile_name: str) -> Dict:
    profiles = copy.deepcopy(DEFAULT_PROFILES)
    profiles.update(cfg.get("profiles", {}))
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {sorted(profiles)}")
    profile = profiles[profile_name]
    validate_profile(profile_name, profile)
    return profile


def validate_profile(profile_name: str, profile: Dict) -> None:
    centerscale = profile.get("centerscale", {})
    if not bool(centerscale.get("enabled", True)):
        centerscale["position"] = "none"
    position = centerscale.get("position", "pre_pca")
    fit_on = centerscale.get("fit_on", "all")
    if position not in VALID_CENTERSCALE_POSITIONS:
        raise ValueError(
            f"{profile_name}: centerscale.position must be one of "
            f"{sorted(VALID_CENTERSCALE_POSITIONS)}, got {position!r}"
        )
    if fit_on not in VALID_CENTERSCALE_FIT:
        raise ValueError(
            f"{profile_name}: centerscale.fit_on must be one of "
            f"{sorted(VALID_CENTERSCALE_FIT)}, got {fit_on!r}"
        )

    for section in ["centerscale", "mad", "spherize"]:
        section_cfg = profile.get(section, {})
        if section == "centerscale" and not bool(section_cfg.get("enabled", True)):
            continue
        if section == "spherize" and not bool(section_cfg.get("enabled", False)):
            continue
        scope = section_cfg.get("scope", "plate")
        if scope not in VALID_SCOPES:
            raise ValueError(f"{profile_name}: {section}.scope must be one of {sorted(VALID_SCOPES)}, got {scope!r}")

    pca_cfg = profile.get("pca", {})
    if bool(pca_cfg.get("enabled", True)):
        n_components = int(pca_cfg.get("n_components", 64))
        if n_components < 1:
            raise ValueError(f"{profile_name}: enabled PCA requires n_components >= 1")
    else:
        n_components = int(pca_cfg.get("n_components", 0))
        if n_components != 0:
            raise ValueError(f"{profile_name}: disabled PCA must set n_components: 0")


def resolve_pca_components(profile_cfg: Dict, n_samples: int, n_features: int) -> int:
    pca_cfg = profile_cfg.get("pca", {})
    if not bool(pca_cfg.get("enabled", True)):
        return 0
    requested = int(pca_cfg.get("n_components", 64))
    return min(requested, n_samples, n_features)


def labels_for_scope(df: pd.DataFrame, split_cfg: Dict, scope: str) -> np.ndarray:
    if scope == "plate":
        column = split_cfg.get("plate_column", "Metadata_Plate")
    elif scope == "batch":
        column = split_cfg.get("batch_column", "Metadata_Batch")
    elif scope == "source":
        column = split_cfg.get("source_column", "Metadata_Source")
    else:
        raise ValueError(f"Unknown normalization scope: {scope}")
    if column not in df.columns:
        raise ValueError(f"Normalization scope '{scope}' requires missing column '{column}'")
    return df[column].fillna(f"__missing_{scope}__").astype(str).to_numpy()


def resolve_models(cfg: Dict, model_arg: str) -> List[str]:
    if model_arg == "all":
        return list(cfg["models"])
    requested = parse_list([model_arg])
    unknown = [m for m in requested if m not in cfg["models"]]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {cfg['models']}")
    return requested


def _read_bbbc036_metadata(split_cfg: Dict, metadata_root: str) -> pd.DataFrame:
    combined = split_cfg.get("combined_metadata_path")
    if combined and os.path.exists(combined):
        meta = pd.read_parquet(combined)
    else:
        pattern = split_cfg.get(
            "per_plate_metadata_glob",
            os.path.join(metadata_root, "bbbc036", "metadata", "per_plate", "*_metadata.parquet"),
        )
        files = sorted(glob.glob(pattern))
        if not files:
            blob_path = split_cfg.get("bbbc036_metadata_path")
            if blob_path:
                path = blob_path if os.path.isabs(blob_path) else os.path.join(metadata_root, blob_path)
                if os.path.exists(path):
                    meta = pd.read_parquet(path)
                else:
                    raise FileNotFoundError(f"BBBC036 metadata not found: {path}")
            else:
                raise FileNotFoundError(f"No BBBC036 per-plate metadata files found: {pattern}")
        else:
            meta = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    meta["Metadata_Plate"] = meta["Metadata_Plate"].astype(str)
    keep = ["Metadata_Plate", "Metadata_Well"]
    for col in ["Metadata_ASSAY_WELL_ROLE", "Metadata_broad_sample", "Metadata_moa", "Metadata_target"]:
        if col in meta.columns:
            keep.append(col)
    return meta[keep].drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"])


def _read_cpg_metadata(split_cfg: Dict, metadata_root: str) -> pd.DataFrame:
    combined = split_cfg.get("combined_metadata_path")
    candidates = []
    if combined:
        candidates.append(combined if os.path.isabs(combined) else os.path.join(metadata_root, combined))
    candidates.extend([
        os.path.join(metadata_root, "metadata", "plate_metadata", "metadata.parquet"),
        os.path.join(metadata_root, "plate_metadata", "metadata.parquet"),
        os.path.join(metadata_root, "cpg0016", "metadata", "metadata.parquet"),
    ])
    for path in candidates:
        if os.path.exists(path):
            meta = pd.read_parquet(path)
            meta["Metadata_Plate"] = meta["Metadata_Plate"].astype(str)
            keep = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
            for col in ["Metadata_JCP2022", "Metadata_Symbol", "Metadata_Batch", "Metadata_PlateType"]:
                if col in meta.columns:
                    keep.append(col)
            return meta[keep].drop_duplicates()
    raise FileNotFoundError(
        "CPG normalization needs combined well-level metadata with Metadata_JCP2022 "
        "or Metadata_Symbol. Configure combined_metadata_path."
    )


def attach_metadata(df: pd.DataFrame, split: str, split_cfg: Dict, metadata_root: str) -> pd.DataFrame:
    df = df.copy()
    df["Metadata_Plate"] = df["Metadata_Plate"].astype(str)
    if split == "bbbc036":
        meta = _read_bbbc036_metadata(split_cfg, metadata_root)
        merged = df.merge(meta, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    else:
        meta = _read_cpg_metadata(split_cfg, metadata_root)
        keys = ["Metadata_Plate", "Metadata_Well"]
        if "Metadata_Source" in df.columns and "Metadata_Source" in meta.columns:
            keys.insert(0, "Metadata_Source")
        duplicate_meta_cols = [c for c in meta.columns if c in df.columns and c not in keys]
        if duplicate_meta_cols:
            meta = meta.drop(columns=duplicate_meta_cols)
        merged = df.merge(meta, on=keys, how="inner")
    merged = merged.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"], keep="first")
    if merged.empty:
        raise RuntimeError(f"{split}: metadata join produced 0 rows")
    return merged


def control_mask(df: pd.DataFrame, split_cfg: Dict) -> np.ndarray:
    column = split_cfg["control_column"]
    value = split_cfg["control_value"]
    if column not in df.columns:
        raise ValueError(f"Missing control column '{column}'")
    mask = (df[column] == value).to_numpy()
    if mask.sum() == 0:
        raise RuntimeError(f"No controls found for {column} == {value}")
    return mask


def validate_normalized(df: pd.DataFrame, feature_cols: List[str], label: str):
    if df.duplicated(subset=["Metadata_Plate", "Metadata_Well"]).sum():
        raise ValueError(f"{label}: duplicate plate/well rows")
    values = df[feature_cols].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: normalized features contain NaN or Inf")
    return {
        "rows": int(len(df)),
        "feature_columns": int(len(feature_cols)),
        "plates": int(df["Metadata_Plate"].nunique()),
    }


def normalize_one(
    split: str,
    model: str,
    cfg: Dict,
    profile_name: str,
    profile_cfg: Dict,
    mode: str,
) -> Dict[str, object]:
    aggregated_root = _root(cfg, "aggregated_root", mode)
    normalized_root = _root(cfg, "normalized_root", mode)
    metadata_root = _root(cfg, "metadata_root", mode)
    split_cfg = cfg["splits"][split]

    input_path = os.path.join(aggregated_root, split, f"{model}_aggregated.parquet")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing aggregated input: {input_path}")
    df = pd.read_parquet(input_path)
    df = attach_metadata(df, split, split_cfg, metadata_root)

    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    if not feature_cols:
        raise ValueError(f"{split}/{model}: no feature columns")

    mask = control_mask(df, split_cfg)
    plate_labels = df[split_cfg.get("plate_column", "Metadata_Plate")].to_numpy()
    features = df[feature_cols].to_numpy(dtype=np.float32)

    pca_components = resolve_pca_components(profile_cfg, features.shape[0], features.shape[1])
    if pca_components < 0:
        raise ValueError(f"{split}/{model}: invalid PCA components after bounds: {pca_components}")

    centerscale_cfg = profile_cfg.get("centerscale", {})
    mad_cfg = profile_cfg.get("mad", {})
    spherize_cfg = profile_cfg.get("spherize", {})

    normalized = normalize_single_model(
        features=features,
        plate_labels=plate_labels,
        control_mask=mask,
        n_pca_components=pca_components,
        min_controls_for_mad=int(cfg.get("min_controls_for_mad", 5)),
        min_controls_for_spherize=int(cfg.get("min_controls_for_spherize", 20)),
        centerscale_position=centerscale_cfg.get("position", "pre_pca"),
        centerscale_fit_on=centerscale_cfg.get("fit_on", "all"),
        centerscale_enabled=bool(centerscale_cfg.get("enabled", True)),
        centerscale_labels=labels_for_scope(df, split_cfg, centerscale_cfg.get("scope", "plate")),
        mad_labels=labels_for_scope(df, split_cfg, mad_cfg.get("scope", "plate")),
        spherize_labels=labels_for_scope(df, split_cfg, spherize_cfg.get("scope", "plate")),
        do_spherize=bool(spherize_cfg.get("enabled", False)),
    )

    feature_prefix = "PC" if pca_components > 0 else "F"
    out_feature_cols = [f"{feature_prefix}_{i}" for i in range(normalized.shape[1])]
    out = pd.concat(
        [
            df[metadata_cols].reset_index(drop=True),
            pd.DataFrame(normalized, columns=out_feature_cols),
        ],
        axis=1,
    )
    summary = validate_normalized(out, out_feature_cols, f"{split}/{model}")

    out_dir = os.path.join(normalized_root, profile_name, split)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{model}_normalized.parquet")
    out.to_parquet(output_path, index=False)

    summary.update({
        "split": split,
        "model": model,
        "input_path": input_path,
        "output_path": output_path,
        "controls": int(mask.sum()),
        "pca_components": int(pca_components),
        "output_feature_columns": int(normalized.shape[1]),
        "profile_config": profile_cfg,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Normalize aggregated features for one or more splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="CSAll_Plate__PCA64__MADCtrl_Plate__NoSph")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--model", default="all")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get("execution", {}).get("mode", "local")
    profile_cfg = resolve_profile(cfg, args.profile)
    splits = parse_list(args.splits)
    models = resolve_models(cfg, args.model)

    summaries = []
    failures = []
    for split in splits:
        if split not in cfg["splits"]:
            raise ValueError(f"Unknown split '{split}'. Available: {sorted(cfg['splits'])}")
        for model in models:
            try:
                logger.info("Normalizing %s / %s / %s", args.profile, split, model)
                summaries.append(normalize_one(split, model, cfg, args.profile, profile_cfg, mode))
            except Exception:
                logger.exception("Failed: %s / %s", split, model)
                failures.append(f"{split}/{model}")
                if not args.continue_on_error:
                    raise

    normalized_root = _root(cfg, "normalized_root", mode)
    manifest_dir = os.path.join(normalized_root, args.profile)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "normalization_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "profile": args.profile,
                "profile_config": profile_cfg,
                "outputs": summaries,
                "failures": failures,
            },
            f,
            indent=2,
        )

    if failures:
        raise RuntimeError(f"Normalization failed for: {failures}")
    logger.info("Normalization complete. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
