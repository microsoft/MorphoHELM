import argparse
import copy
import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional

import config
from run_inference import get_supported_splits, run_inference_for_config


CPG_SPLIT_DEFAULTS = {
    "cpg-crispr": {"dataset_name": "cpg0016", "bit_depth": 8, "split": "cpg-crispr"},
    "cpg-tgt2": {"dataset_name": "cpg0016", "bit_depth": 8, "split": "cpg-tgt2"},
    "cpg-compound": {"dataset_name": "cpg0016", "bit_depth": 8, "split": "cpg-compound"},
    "cpg-moa": {"dataset_name": "cpg0016", "bit_depth": 8, "split": "cpg-moa"},
}
SPLIT_DEFAULTS = {
    **CPG_SPLIT_DEFAULTS,
    "bbbc036": {"dataset_name": "bbbc036", "bit_depth": 8, "split": None},
}


def parse_splits(values: Iterable[str]) -> List[str]:
    """Parse comma-separated and repeated --splits values."""
    splits = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                splits.append(item)
    if not splits:
        raise ValueError("At least one split must be provided.")

    supported = set(get_supported_splits())
    unknown = [split for split in splits if split not in supported]
    if unknown:
        raise ValueError(
            f"Unknown split(s): {unknown}. Supported splits: {sorted(supported)}"
        )
    return splits


def _mode_value(value, mode: str):
    """Return a mode-specific value when the config stores values by mode."""
    if isinstance(value, Mapping) and mode in value:
        return value[mode]
    return value


def _read_split_settings(split_inference: Dict, split: str, mode: str) -> Dict:
    """Merge common and mode-specific settings for one split."""
    defaults = SPLIT_DEFAULTS[split]
    split_settings = copy.deepcopy(defaults)
    configured = copy.deepcopy(split_inference.get("splits", {}).get(split, {}))

    for key, value in configured.items():
        if key in ("local", "amlt"):
            continue
        split_settings[key] = value
    split_settings.update(configured.get(mode, {}))
    return split_settings


def _output_root_for_mode(cfg: Dict, mode: str) -> Optional[str]:
    split_inference = cfg.get("split_inference", {})
    output_root = split_inference.get("output_root")
    if output_root is None:
        output_root = cfg.get("inference", {}).get("output_root")
    if output_root is not None:
        return _mode_value(output_root, mode)

    current_output = cfg.get(mode, {}).get("output_path")
    if not current_output:
        return None

    final_component = os.path.basename(os.path.normpath(current_output))
    if final_component in get_supported_splits():
        return os.path.dirname(os.path.normpath(current_output))
    return current_output


def build_split_config(base_cfg: Dict, split: str) -> Dict:
    """Return a deep-copied config specialized to one split."""
    cfg = copy.deepcopy(base_cfg)
    mode = cfg["execution"]["mode"]
    split_inference = cfg.get("split_inference", {})
    split_settings = _read_split_settings(split_inference, split, mode)
    mode_section = cfg.setdefault(mode, {})

    dataset_name = split_settings["dataset_name"]
    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_cfg["name"] = dataset_name
    dataset_cfg["bit_depth"] = split_settings["bit_depth"]

    if split_settings.get("split"):
        dataset_cfg["split"] = split_settings["split"]
    else:
        dataset_cfg.pop("split", None)

    for key in ("exclude_plates", "test_plates", "max_plates", "moa_labels_csv"):
        dataset_cfg.pop(key, None)
        if key in split_settings:
            dataset_cfg[key] = split_settings[key]

    for path_key in ("data_path", "metadata_path", "log_path", "git_clone_dir"):
        if path_key in split_settings:
            mode_section[path_key] = split_settings[path_key]

    if "output_path" in split_settings:
        mode_section["output_path"] = split_settings["output_path"]
    else:
        output_root = _output_root_for_mode(cfg, mode)
        if not output_root:
            raise ValueError(
                f"No output path/root configured for split '{split}'. Set "
                f"{mode}.output_path or split_inference.output_root."
            )
        mode_section["output_path"] = os.path.join(output_root, split)

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run HuggingFace inference for one or more dataset splits consecutively"
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML config path")
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Split names to run. Accepts space-separated and comma-separated values.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override: run only this model for every split",
    )
    parser.add_argument(
        "--max-plates",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to every split",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve plates/models/output paths without running GPU inference",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next split if one split fails. Default: stop on first failure.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [multi-split] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("multi-split")

    base_cfg = config.load_config(args.config)
    splits = parse_splits(args.splits)
    logger.info(f"Running splits in order: {splits}")

    failures = []
    for split in splits:
        logger.info(f"Starting split: {split}")
        split_cfg = build_split_config(base_cfg, split)
        try:
            summary = run_inference_for_config(
                split_cfg,
                model_override=args.model,
                max_plates=args.max_plates,
                dry_run=args.dry_run,
                logger=logger,
            )
            logger.info(
                f"Finished split {split}: {summary['num_plates']} plates, "
                f"models={summary['models']}, output={summary['output_path']}"
            )
        except Exception as exc:
            logger.exception(f"Split '{split}' failed: {exc}")
            failures.append(split)
            if not args.continue_on_error:
                raise

    if failures:
        raise RuntimeError(f"One or more splits failed: {failures}")
    logger.info("All requested splits complete")


if __name__ == "__main__":
    main()
