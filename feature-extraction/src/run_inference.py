import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import pandas as pd
import logging
from typing import Dict, List, Optional

import config
from utils.util import get_plate_chunks

from cellpainting_huggingface_loader.create_splits import CreateSplits


CPG_SPLITS = ("cpg-crispr", "cpg-tgt2", "cpg-compound", "cpg-moa")
SUPPORTED_SPLITS = CPG_SPLITS + ("bbbc036",)


def setup_gpu_logger(gpu_id, log_path):
    """Create and configure a logger for this GPU process."""
    logger = logging.getLogger(f"gpu_{gpu_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, f"inference_gpu_{gpu_id}.log")
        fh = logging.FileHandler(log_file)
        fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    return logger


def inference(gpu_id,
              model_name,
              split_generator,
              plates_chunk,
              cfg):
    """Run inference for a single model on a single GPU."""
    _, _, output_path, log_path, _ = config.get_paths(cfg)
    inference_cfg = cfg.get("inference", {})

    logger = setup_gpu_logger(gpu_id, log_path)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"{device} starting inference for model={model_name}")

    dataset = split_generator.generate_dataset_by_name(
        split_type="by_plate",
        split=plates_chunk[gpu_id]
    ).with_format("torch")

    custom_transform = config.get_transform(model_name, cfg)
    dataloader_settings = config.get_dataloader_settings(model_name, cfg)

    save_path = os.path.join(output_path, model_name)
    # Per-model mixed_precision override, falling back to global inference setting
    model_cfg = cfg["models"][model_name]
    use_mixed_precision = model_cfg.get("mixed_precision",
                                        inference_cfg.get("mixed_precision", True))

    # Adaptive save frequency: target ~50k samples per checkpoint.
    # Models with tiny batch sizes (CLOOME=15, SubCell=60) would otherwise
    # create hundreds of small pkl files when save_frequency is in batches.
    batch_size = dataloader_settings.get("batch_size", 64)
    target_samples_per_save = inference_cfg.get("target_samples_per_save", 50000)
    save_frequency = max(100, target_samples_per_save // max(batch_size, 1))
    logger.info(f"save_frequency={save_frequency} batches (batch_size={batch_size}, "
                f"target ~{target_samples_per_save} samples/checkpoint)")

    # Build model wrapper kwargs
    wrapper_kwargs = dict(
        dataset=dataset,
        preprocess_transform=custom_transform,
        dataloader_settings=dataloader_settings,
        save_path=save_path,
        save_frequency=save_frequency,
        device=device,
        logger=logger,
        use_mixed_precision=use_mixed_precision,
    )

    # Patching config
    patching_cfg = cfg.get("patching", {})
    wrapper_kwargs["patching_enabled"] = patching_cfg.get("enabled", False)
    wrapper_kwargs["num_patches"] = patching_cfg.get("num_patches", 4)

    # Model-specific extra kwargs
    model_cfg = cfg["models"][model_name]
    if model_name == "cloome":
        wrapper_kwargs["checkpoint_path"] = model_cfg["checkpoint"]
        wrapper_kwargs["model_config_path"] = model_cfg.get("model_config")
    elif model_name == "subcell":
        wrapper_kwargs["encoder_path"] = model_cfg["encoder_path"]

    wrapper_cls = config.get_model_wrapper(model_name)
    model = wrapper_cls(**wrapper_kwargs)
    model.run_inference()
    logger.info(f"{device} finished inference for model={model_name}")


def run_model(model_name, cfg, split_generator, plate_chunks, num_gpus):
    """Spawn multi-GPU inference for a single model.

    When a staging_dir is configured (recommended for blob storage outputs),
    intermediate pickles are written to fast local disk first, then copied
    to the final blob-mounted output_path after all GPUs finish.  This avoids
    filling the blobfuse2 local-disk cache with hundreds of GB of writes.
    """
    logger = logging.getLogger("main")
    logger.info(f"Starting inference for model: {model_name}")

    _, _, output_path, _, _ = config.get_paths(cfg)
    staging_dir = cfg.get("inference", {}).get("staging_dir")

    # If staging is enabled, temporarily redirect output to local fast storage
    if staging_dir:
        original_output = output_path
        staged_output = os.path.join(staging_dir, os.path.basename(output_path))
        cfg_copy = _deep_copy_cfg_with_output(cfg, staged_output)
        logger.info(f"Staging results to {staged_output}, will copy to {original_output} after inference")
    else:
        cfg_copy = cfg

    if num_gpus > 0:
        mp.spawn(
            inference,
            args=(model_name, split_generator, plate_chunks, cfg_copy),
            nprocs=num_gpus,
            join=True,
        )
    else:
        logger.error("No GPUs available")

    # Copy staged results to final blob destination
    if staging_dir:
        src = os.path.join(staged_output, model_name)
        dst = os.path.join(original_output, model_name)
        if os.path.isdir(src):
            logger.info(f"Copying {model_name} results: {src} → {dst}")
            import shutil
            os.makedirs(dst, exist_ok=True)
            # Copy each GPU result directory
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logger.info(f"Copy complete for {model_name}, cleaning up staging dir")
            shutil.rmtree(src, ignore_errors=True)
        else:
            logger.warning(f"No staged results found at {src}")

    logger.info(f"Completed inference for model: {model_name}")


def _deep_copy_cfg_with_output(cfg, new_output_path):
    """Return a shallow copy of cfg with the output path redirected."""
    import copy
    cfg_copy = copy.deepcopy(cfg)
    mode = cfg_copy.get("execution", {}).get("mode", "local")
    if mode == "amlt":
        cfg_copy["amlt"]["output_path"] = new_output_path
    else:
        cfg_copy["local"]["output_path"] = new_output_path
    return cfg_copy


def _select_plates_from_csv(plates_df, split, cfg=None):
    """Select plate names from CPG0016 plate.csv.gz based on the dataset split.

    Supported splits:
      cpg-crispr   — Metadata_PlateType == "CRISPR"
      cpg-tgt2     — Metadata_PlateType == "TARGET2"
      cpg-compound — Metadata_Source in {source_5, source_9, source_11}
      cpg-moa      — All plates that appear in labeled_moa_samples.csv
    """
    if split == "cpg-crispr":
        mask = plates_df["Metadata_PlateType"] == "CRISPR"
    elif split == "cpg-tgt2":
        mask = plates_df["Metadata_PlateType"] == "TARGET2"
    elif split == "cpg-compound":
        mask = plates_df["Metadata_Source"].isin(["source_5", "source_9", "source_11"])
    elif split == "cpg-moa":
        moa_csv = cfg["dataset"].get("moa_labels_csv") if cfg else None
        if not moa_csv:
            moa_csv = "feature-extraction/configs/labeled_moa_samples.csv"
        if not os.path.isabs(moa_csv):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            moa_csv = os.path.join(repo_root, moa_csv)
        moa_df = pd.read_csv(moa_csv)
        moa_plates = set(moa_df["Metadata_Plate"].unique())
        mask = plates_df["Metadata_Plate"].isin(moa_plates)
    else:
        raise ValueError(
            f"Unknown CPG0016 split '{split}'. Must be one of: cpg-crispr, cpg-tgt2, cpg-compound, cpg-moa"
        )
    selected = plates_df.loc[mask, "Metadata_Plate"].astype(str).tolist()
    if not selected:
        raise RuntimeError(f"Split '{split}' matched 0 plates.")
    return selected


def _select_plates_from_dir(data_path, exclude_plates=None):
    """List all plate directories from the data path (for datasets without plate.csv.gz).

    Used for BBBC036 and similar datasets where plates are simply
    the subdirectories of the data mount.
    """
    exclude = set(exclude_plates or [])
    plates = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d)) and d not in exclude
    ])
    if not plates:
        raise RuntimeError(f"No plate directories found in {data_path}")
    return plates


def get_supported_splits():
    """Return supported split names for the HuggingFace inference path."""
    return SUPPORTED_SPLITS


def resolve_plate_names(cfg, max_plates: Optional[int] = None, logger=None) -> List[str]:
    """Resolve plate names for the dataset/split described by cfg."""
    data_path, _, _, _, git_clone_dir = config.get_paths(cfg)
    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split")
    exclude_plates = cfg["dataset"].get("exclude_plates", [])

    if split and split.startswith("cpg-"):
        if not git_clone_dir:
            raise ValueError("CPG split inference requires git_clone_dir with metadata/plate.csv.gz")
        all_plates = pd.read_csv(
            os.path.join(git_clone_dir, "metadata/plate.csv.gz")
        )
        chosen_plate_names = _select_plates_from_csv(all_plates, split, cfg)
        existing = [p for p in chosen_plate_names if os.path.isdir(os.path.join(data_path, p))]
        if len(existing) < len(chosen_plate_names) and logger:
            logger.warning(
                f"Split '{split}': {len(chosen_plate_names) - len(existing)} plates not found "
                f"in {data_path}, skipping them."
            )
        chosen_plate_names = existing
    else:
        chosen_plate_names = _select_plates_from_dir(data_path, exclude_plates)

    test_plates = cfg["dataset"].get("test_plates")
    if test_plates:
        chosen_plate_names = [p for p in test_plates if p in chosen_plate_names or
                              os.path.isdir(os.path.join(data_path, str(p)))]
        if logger:
            logger.info(f"Using test_plates subset: {chosen_plate_names}")

    effective_max_plates = max_plates
    if effective_max_plates is None:
        effective_max_plates = cfg["dataset"].get("max_plates")
    if effective_max_plates is not None:
        chosen_plate_names = chosen_plate_names[:effective_max_plates]

    if not chosen_plate_names:
        split_label = split or dataset_name
        raise RuntimeError(f"Dataset '{split_label}' matched 0 existing plates in {data_path}")
    return chosen_plate_names


def build_split_generator(cfg):
    """Build the HuggingFace split generator for cfg."""
    data_path, metadata_path, _, _, _ = config.get_paths(cfg)
    return CreateSplits(
        metadata_path=metadata_path,
        blob_mount_point=data_path,
    )


def select_models(cfg, model_override: Optional[str] = None) -> List[str]:
    """Resolve models to run from CLI override or enabled model flags."""
    if model_override:
        if model_override not in cfg["models"]:
            raise ValueError(f"Unknown model '{model_override}'. Available: {sorted(cfg['models'])}")
        return [model_override]
    return config.get_enabled_models(cfg)


def run_inference_for_config(
    cfg,
    model_override: Optional[str] = None,
    max_plates: Optional[int] = None,
    dry_run: bool = False,
    logger=None,
) -> Dict[str, object]:
    """Run or describe inference for one fully-resolved config."""
    logger = logger or logging.getLogger("main")
    data_path, metadata_path, output_path, log_path, git_clone_dir = config.get_paths(cfg)
    dataset_name = cfg["dataset"]["name"]
    split = cfg["dataset"].get("split", dataset_name)

    chosen_plate_names = resolve_plate_names(cfg, max_plates=max_plates, logger=logger)
    models_to_run = select_models(cfg, model_override)

    summary = {
        "dataset": dataset_name,
        "split": split,
        "num_plates": len(chosen_plate_names),
        "plates": chosen_plate_names,
        "models": models_to_run,
        "data_path": data_path,
        "metadata_path": metadata_path,
        "output_path": output_path,
        "log_path": log_path,
        "git_clone_dir": git_clone_dir,
    }

    logger.info(
        f"Dataset '{dataset_name}' split '{split}': {len(chosen_plate_names)} plates, "
        f"models={models_to_run}, output={output_path}"
    )

    if dry_run:
        logger.info(f"Dry run complete for split '{split}'")
        return summary

    available_gpus = torch.cuda.device_count()
    num_gpus = min(available_gpus, len(chosen_plate_names))
    logger.info(f"Found {available_gpus} GPUs; using {num_gpus} for {len(chosen_plate_names)} plates")
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. Exiting.")

    plate_chunks = get_plate_chunks(num_gpus, chosen_plate_names)
    split_generator = build_split_generator(cfg)

    for model_name in models_to_run:
        run_model(model_name, cfg, split_generator, plate_chunks, num_gpus)

    logger.info(f"Inference complete for split '{split}'")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction inference")
    parser.add_argument(
        "--config", type=str,
        default="configs/inference_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override: run only this model (ignores 'enabled' flags in config)",
    )
    parser.add_argument(
        "--max-plates", type=int, default=None,
        help="Optional smoke-test limit on plates for this invocation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve plates/models/output paths without running GPU inference",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [main] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("main")

    # Load config
    cfg = config.load_config(args.config)
    logger.info(f"Config loaded: mode={cfg['execution']['mode']}, dataset={cfg['dataset']['name']}")
    try:
        run_inference_for_config(
            cfg,
            model_override=args.model,
            max_plates=args.max_plates,
            dry_run=args.dry_run,
            logger=logger,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    logger.info("All inference complete")


if __name__ == "__main__":
    main()
