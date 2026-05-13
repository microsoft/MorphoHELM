"""Direct inference script — loads images from disk, pre-filters to relevant compounds.

Unlike run_inference.py (HuggingFace dataset path), this script:
  1. Reads metadata.parquet with per-FOV image paths
  2. Filters to MOA compounds + negative/positive controls
  3. Loads images directly from raw PNGs (no HF dataset overhead)
  4. Splits filtered rows across GPUs (not by plates)

~4x faster than HF path for MOA inference (525K vs 2.2M FOVs).

Usage:
    python run_inference_direct.py --config configs/direct_moa_config.yaml
    python run_inference_direct.py --config configs/direct_moa_config.yaml --model open_phenom
"""

import os
import sys
import argparse
import copy
import gc
import logging
import shutil

import pandas as pd
import torch
import torch.multiprocessing as mp

import config
from custom_datasets.direct_dataset import DirectImageDataset

# ── Compound filtering constants ──────────────────────────────────────────────

CONTROL_JCP = "JCP2022_033924"  # Negative control (non-targeting)

POSCON_JCPS = [
    "JCP2022_025848",
    "JCP2022_046054",
    "JCP2022_050797",
    "JCP2022_064022",
    "JCP2022_085227",
    "JCP2022_037716",
    "JCP2022_012818",
    "JCP2022_035095",
]


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


def _find_metadata_file(metadata_path):
    """Locate metadata.parquet, trying standard paths."""
    candidates = [
        os.path.join(metadata_path, "plate_metadata", "metadata.parquet"),
        os.path.join(metadata_path, "metadata.parquet"),
    ]
    for c in candidates:
        print(f"[metadata] Checking: {c} ... ", end="", flush=True)
        if os.path.exists(c):
            print("FOUND", flush=True)
            return c
        print("not found", flush=True)
    raise FileNotFoundError(f"metadata.parquet not found. Tried: {candidates}")


def _resolve_moa_csv(cfg):
    """Resolve the MOA labels CSV path from config."""
    moa_csv = cfg["dataset"].get("moa_labels_csv")
    if not moa_csv:
        moa_csv = "feature-extraction/configs/labeled_moa_samples.csv"
    if not os.path.isabs(moa_csv):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        moa_csv = os.path.join(repo_root, moa_csv)
    return moa_csv


def load_and_filter_metadata(cfg, logger):
    """Load metadata.parquet and filter to MOA compounds + controls.

    Batch-scoped filtering: includes negcons from ALL plates in batches that
    contain at least one MOA-labeled well, ensuring sufficient negcon coverage
    for batchwise normalization.

    Returns:
        filtered_metadata (pd.DataFrame): Rows for relevant compounds only.
    """
    _, metadata_path, _, _, _ = config.get_paths(cfg)

    metadata_file = _find_metadata_file(metadata_path)
    print(f"[metadata] Loading {metadata_file} ...", flush=True)
    logger.info(f"Loading metadata from {metadata_file}")
    metadata_df = pd.read_parquet(metadata_file)
    print(f"[metadata] Loaded {len(metadata_df)} total rows", flush=True)
    logger.info(f"Total metadata rows: {len(metadata_df)}")

    moa_csv = _resolve_moa_csv(cfg)
    labeled_samples = pd.read_csv(moa_csv)
    moa_jcps = set(labeled_samples["Metadata_JCP2022"].unique().tolist())
    moa_plates = set(labeled_samples["Metadata_Plate"].astype(str).unique().tolist())
    logger.info(f"MOA compounds: {len(moa_jcps)} JCPs across {len(moa_plates)} plates")

    # Find batches that contain MOA plates (batch-scoped negcon coverage)
    moa_batches = set(
        metadata_df.loc[
            metadata_df["Metadata_Plate"].astype(str).isin(moa_plates), "Metadata_Batch"
        ].dropna().unique()
    )
    logger.info(f"Batches containing MOA plates: {len(moa_batches)}")

    plate_str = metadata_df["Metadata_Plate"].astype(str)

    # MOA compounds on their specific plates
    moa_mask = plate_str.isin(moa_plates) & metadata_df["Metadata_JCP2022"].isin(moa_jcps)
    # Negcons in ALL plates of MOA-relevant batches (for batchwise normalization)
    negcon_mask = (
        (metadata_df["Metadata_JCP2022"] == CONTROL_JCP)
        & metadata_df["Metadata_Batch"].isin(moa_batches)
    )
    # Poscons on MOA plates
    poscon_mask = plate_str.isin(moa_plates) & metadata_df["Metadata_JCP2022"].isin(set(POSCON_JCPS))

    filtered = metadata_df[moa_mask | negcon_mask | poscon_mask].copy()

    n_moa = moa_mask.sum()
    n_neg = negcon_mask.sum()
    n_pos = poscon_mask.sum()
    print(f"[metadata] Batch-scoped filter: {n_moa} MOA + {n_neg} negcon + {n_pos} poscon FOVs", flush=True)
    print(f"[metadata] Total: {len(filtered)} FOVs ({len(filtered)/len(metadata_df)*100:.1f}%)", flush=True)
    logger.info(f"Batch-scoped filter: {n_moa} MOA + {n_neg} negcon + {n_pos} poscon = {len(filtered)} FOVs")

    # Sanity check: every MOA batch should have negcons
    neg_batches = set(filtered.loc[filtered["Metadata_JCP2022"] == CONTROL_JCP, "Metadata_Batch"].dropna().unique())
    missing = moa_batches - neg_batches
    if missing:
        logger.warning(f"MOA batches without negcons after filtering: {missing}")
    else:
        logger.info(f"All {len(moa_batches)} MOA batches have negcon coverage")

    return filtered


def inference(gpu_id, model_name, metadata_chunks, image_root, cfg):
    """Run inference on a chunk of metadata assigned to this GPU."""
    _, _, output_path, log_path, _ = config.get_paths(cfg)
    inference_cfg = cfg.get("inference", {})

    logger = setup_gpu_logger(gpu_id, os.path.join(log_path, model_name))
    device = torch.device(f"cuda:{gpu_id}")

    chunk = metadata_chunks[gpu_id]
    logger.info(f"{device} starting direct inference: {len(chunk)} FOVs")
    print(f"[{model_name}] GPU {gpu_id}: starting — {len(chunk)} FOVs", flush=True)

    # Build dataset
    custom_transform = config.get_direct_transform(model_name, cfg)
    dataset = DirectImageDataset(
        metadata_df=chunk,
        image_root=image_root,
        transform=custom_transform,
        model_name=model_name,
    )
    print(f"[{model_name}] GPU {gpu_id}: dataset built — {len(dataset)} items", flush=True)
    logger.info(f"Dataset size: {len(dataset)} items")

    # Build dataloader settings (no custom collate)
    dataloader_settings = config.get_direct_dataloader_settings(model_name, cfg)

    save_path = os.path.join(output_path, model_name)

    # Mixed precision
    model_cfg = cfg["models"][model_name]
    use_mixed_precision = model_cfg.get(
        "mixed_precision", inference_cfg.get("mixed_precision", True)
    )

    # Adaptive save frequency
    batch_size = dataloader_settings.get("batch_size", 64)
    target_samples = inference_cfg.get("target_samples_per_save", 50000)
    save_frequency = max(100, target_samples // max(batch_size, 1))

    # Build model wrapper
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

    # Model-specific kwargs
    if model_name == "cloome":
        wrapper_kwargs["checkpoint_path"] = model_cfg["checkpoint"]
        wrapper_kwargs["model_config_path"] = model_cfg.get("model_config")
    elif model_name == "subcell":
        wrapper_kwargs["encoder_path"] = model_cfg["encoder_path"]

    wrapper_cls = config.get_model_wrapper(model_name)
    # For direct mode, transform is already applied in dataset — pass None
    wrapper_kwargs["preprocess_transform"] = None
    print(f"[{model_name}] GPU {gpu_id}: loading model...", flush=True)
    model = wrapper_cls(**wrapper_kwargs)
    print(f"[{model_name}] GPU {gpu_id}: model loaded, starting inference...", flush=True)
    model.run_inference()
    print(f"[{model_name}] GPU {gpu_id}: FINISHED", flush=True)
    logger.info(f"{device} finished inference")


def run_model(model_name, cfg, filtered_metadata, num_gpus):
    """Spawn multi-GPU inference for a single model with staging support."""
    logger = logging.getLogger("main")
    print(f"\n{'='*60}", flush=True)
    print(f"[{model_name}] Starting direct inference — {len(filtered_metadata)} FOVs, {num_gpus} GPUs", flush=True)
    print(f"{'='*60}", flush=True)
    logger.info(f"Starting direct inference for model: {model_name}")

    data_path, _, output_path, _, _ = config.get_paths(cfg)
    staging_dir = cfg.get("inference", {}).get("staging_dir")

    # Image root for raw PNGs
    image_root = cfg.get("direct", {}).get("image_root", data_path)

    # Split metadata across GPUs
    chunk_size = len(filtered_metadata) // num_gpus
    metadata_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = len(filtered_metadata) if i == num_gpus - 1 else (i + 1) * chunk_size
        metadata_chunks.append(filtered_metadata.iloc[start:end])

    logger.info(f"Split {len(filtered_metadata)} FOVs across {num_gpus} GPUs: "
                f"{[len(c) for c in metadata_chunks]}")

    # Staging support
    if staging_dir:
        original_output = output_path
        staged_output = os.path.join(staging_dir, os.path.basename(output_path))
        cfg_copy = _deep_copy_cfg_with_output(cfg, staged_output)
        logger.info(f"Staging to {staged_output}, will copy to {original_output}")
    else:
        cfg_copy = cfg

    mp.spawn(
        inference,
        args=(model_name, metadata_chunks, image_root, cfg_copy),
        nprocs=num_gpus,
        join=True,
    )

    # Copy staged results
    if staging_dir:
        src = os.path.join(staged_output, model_name)
        dst = os.path.join(original_output, model_name)
        if os.path.isdir(src):
            logger.info(f"Copying {model_name} results: {src} → {dst}")
            os.makedirs(dst, exist_ok=True)
            for item in os.listdir(src):
                s, d = os.path.join(src, item), os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logger.info(f"Copy complete, cleaning staging for {model_name}")
            shutil.rmtree(src, ignore_errors=True)
        else:
            logger.warning(f"No staged results at {src}")

    logger.info(f"Completed direct inference for model: {model_name}")


def _deep_copy_cfg_with_output(cfg, new_output_path):
    """Return a deep copy of cfg with output path redirected."""
    cfg_copy = copy.deepcopy(cfg)
    mode = cfg_copy.get("execution", {}).get("mode", "local")
    if mode == "amlt":
        cfg_copy["amlt"]["output_path"] = new_output_path
    else:
        cfg_copy["local"]["output_path"] = new_output_path
    return cfg_copy


def main():
    parser = argparse.ArgumentParser(description="Run direct-mode feature extraction inference")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (must include direct.image_root)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override: run only this model (ignores 'enabled' flags)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [main] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("main")

    # Load config
    cfg = config.load_config(args.config)
    mode = cfg['execution']['mode']
    print(f"[main] Config loaded: mode={mode}", flush=True)
    logger.info(f"Config loaded: mode={mode}")

    # Print paths for debugging
    _, metadata_path, output_path, _, _ = config.get_paths(cfg)
    image_root = cfg.get("direct", {}).get("image_root", "N/A")
    print(f"[main] metadata_path: {metadata_path}", flush=True)
    print(f"[main] output_path:   {output_path}", flush=True)
    print(f"[main] image_root:    {image_root}", flush=True)

    num_gpus = torch.cuda.device_count()
    print(f"[main] GPUs available: {num_gpus}", flush=True)
    logger.info(f"Found {num_gpus} GPUs")
    if num_gpus == 0:
        logger.error("No GPUs available. Exiting.")
        return

    # Load and filter metadata
    filtered_metadata = load_and_filter_metadata(cfg, logger)

    # Determine models to run
    if args.model:
        models_to_run = [args.model]
    else:
        models_to_run = config.get_enabled_models(cfg)

    print(f"[main] Models to run: {models_to_run}", flush=True)
    logger.info(f"Models to run: {models_to_run}")

    for model_name in models_to_run:
        torch.cuda.empty_cache()
        gc.collect()
        run_model(model_name, cfg, filtered_metadata, num_gpus)

    logger.info("All direct inference complete")


if __name__ == "__main__":
    main()
