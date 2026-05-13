import argparse
import copy
import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

from aggregate_features import run_aggregation_for_config
from run_inference import get_supported_splits
from run_inference_splits import build_split_config, parse_splits


def _mode_value(value, mode: str):
    if isinstance(value, Mapping) and mode in value:
        return value[mode]
    return value


def _read_split_settings(base_cfg: Dict, split: str, mode: str) -> Dict:
    configured = copy.deepcopy(base_cfg.get("split_inference", {}).get("splits", {}).get(split, {}))
    split_settings = {}
    for key, value in configured.items():
        if key in ("local", "amlt"):
            continue
        split_settings[key] = value
    split_settings.update(configured.get(mode, {}))
    return split_settings


def _aggregation_root_for_mode(cfg: Dict, mode: str) -> Optional[str]:
    agg_cfg = cfg.get("aggregation", {})
    for key in ("output_root", "aggregated_output_root"):
        if key in agg_cfg:
            return _mode_value(agg_cfg[key], mode)
    split_inference = cfg.get("split_inference", {})
    for key in ("aggregation_output_root", "aggregated_output_root"):
        if key in split_inference:
            return _mode_value(split_inference[key], mode)
    current_output = cfg.get(mode, {}).get("output_path")
    if current_output:
        final_component = os.path.basename(os.path.normpath(current_output))
        if final_component in get_supported_splits():
            return os.path.dirname(os.path.dirname(os.path.normpath(current_output)))
    return None


def _split_aggregation_output(base_cfg: Dict, split_cfg: Dict, split: str) -> str:
    mode = split_cfg["execution"]["mode"]
    split_settings = _read_split_settings(base_cfg, split, mode)
    for key in ("aggregation_output_path", "aggregated_output_path"):
        if key in split_settings:
            return split_settings[key]

    root = _aggregation_root_for_mode(base_cfg, mode)
    if root:
        return os.path.join(root, split)

    agg_cfg = split_cfg.get("aggregation", {})
    if agg_cfg.get("output_path"):
        return agg_cfg["output_path"]
    return split_cfg[mode]["output_path"]


def _direct_inference_outputs(base_cfg: Dict, split: str, mode: str) -> List[str]:
    split_settings = _read_split_settings(base_cfg, split, mode)
    direct_paths = []
    for key in ("direct_inference_output_path", "direct_output_path"):
        value = split_settings.get(key)
        if value:
            direct_paths.append(value)
    value = split_settings.get("direct_inference_output_paths")
    if value:
        direct_paths.extend(value if isinstance(value, list) else [value])
    return direct_paths


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate inference outputs for one or more dataset splits"
    )
    parser.add_argument("--config", required=True, help="Base inference_splits.yaml config")
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="Split names to aggregate. Accepts space-separated and comma-separated values.",
    )
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers for pkl loading per model")
    parser.add_argument("--parallel-models", action="store_true", help="Aggregate models in parallel")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after split/model failures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [aggregate-splits] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("aggregate-splits")

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    splits = parse_splits(args.splits)
    failures = []
    for split in splits:
        logger.info("Aggregating split: %s", split)
        split_cfg = build_split_config(base_cfg, split)
        mode = split_cfg["execution"]["mode"]
        output_path = _split_aggregation_output(base_cfg, split_cfg, split)

        try:
            run_aggregation_for_config(
                split_cfg,
                model_override=args.model,
                output_path=output_path,
                num_workers=args.num_workers,
                parallel_models=args.parallel_models,
                continue_on_error=args.continue_on_error,
                logger=logger,
            )

            for direct_path in _direct_inference_outputs(base_cfg, split, mode):
                logger.info("Aggregating direct inference output for %s: %s", split, direct_path)
                run_aggregation_for_config(
                    split_cfg,
                    model_override=args.model,
                    output_path=output_path,
                    inference_output_path=direct_path,
                    num_workers=args.num_workers,
                    parallel_models=args.parallel_models,
                    continue_on_error=args.continue_on_error,
                    logger=logger,
                )
        except Exception:
            logger.exception("Aggregation failed for split %s", split)
            failures.append(split)
            if not args.continue_on_error:
                raise

    if failures:
        raise RuntimeError(f"Aggregation failed for split(s): {failures}")
    logger.info("All requested splits aggregated")


if __name__ == "__main__":
    main()
