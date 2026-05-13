"""Central configuration module driven by YAML config file.

Loads inference_config.yaml and builds transforms, model wrappers, and
dataloader settings dynamically based on the selected dataset and models.

Usage:
    import config
    cfg = config.load_config("configs/inference_config.yaml")
    # or for the active config (set after load_config is called):
    transform = config.get_transform("dino_v2")
    wrapper_cls = config.get_model_wrapper("dino_v2")
you are"""

import yaml
import torch
import torchvision.transforms as transforms
from functools import partial
from torch.utils.data.dataloader import default_collate

from models import (
    ResNetWrapper,
    ResNetUntrainedWrapper,
    VGG19Wrapper,
    DINOv2Wrapper,
    DINOv2HighRes448Wrapper,
    OpenPhenomWrapper,
    CloomeWrapper,
    SubCellWrapper,
)
from utils.util import (
    divide_255,
    divide_65535,
    multiply_255,
    repeat,
    minmax_normalize,
    get_plate_chunks,
    illumination_threshold_tensor,
)

# ── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Model wrapper class registry ─────────────────────────────────────────────
MODEL_WRAPPERS = {
    "dino_v2": DINOv2Wrapper,
    "dino_v2_448": DINOv2HighRes448Wrapper,
    "resnet": ResNetWrapper,
    "resnet_untrained": ResNetUntrainedWrapper,
    "vgg19": VGG19Wrapper,
    "open_phenom": OpenPhenomWrapper,
    "cloome": CloomeWrapper,
    "subcell": SubCellWrapper,
}

# ── Active config (populated by load_config) ──────────────────────────────────
_active_config = None


def load_config(config_path):
    """Load a YAML config file and set it as the active config."""
    global _active_config
    with open(config_path, "r") as f:
        _active_config = yaml.safe_load(f)
    return _active_config


def get_active_config():
    """Return the currently loaded config, or raise if none loaded."""
    if _active_config is None:
        raise RuntimeError("No config loaded. Call load_config() first.")
    return _active_config


# ── Path helpers ──────────────────────────────────────────────────────────────


def get_paths(cfg=None):
    """Return (data_path, metadata_path, output_path, log_path, git_clone_dir)
    based on execution mode (local vs amlt)."""
    cfg = cfg or get_active_config()
    mode = cfg["execution"]["mode"]
    section = cfg[mode]
    return (
        section["data_path"],
        section["metadata_path"],
        section["output_path"],
        section["log_path"],
        section["git_clone_dir"],
    )


# ── Transform builders ────────────────────────────────────────────────────────


def _build_classic_transform(bit_depth=8):
    """Build the standard transform for ImageNet-pretrained models.

    Pipeline: Resize(256) → CenterCrop(224) → scale to [0,1] → repeat 1→3 ch → ImageNet normalize.

    For 16-bit data: illumination thresholding first converts to 0-255 range
    (percentile-based contrast stretching), then divide_255 scales to [0,1].
    This avoids the dynamic range loss of naive /65535 scaling.
    """
    if bit_depth == 16:
        return transforms.Compose(
            [
                transforms.Lambda(illumination_threshold_tensor),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Lambda(divide_255),
                transforms.Lambda(repeat),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(divide_255),
            transforms.Lambda(repeat),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _build_dinov2_448_transform(bit_depth=8):
    """Build the high-resolution DINOv2 transform.

    Keeps the existing DINOv2 grayscale→RGB repeat and ImageNet normalization,
    but changes the final crop from 224x224 to 448x448. 448 is divisible by
    DINOv2 ViT-B/14's token patch size.
    """
    steps = []
    if bit_depth == 16:
        steps.append(transforms.Lambda(illumination_threshold_tensor))

    steps.extend(
        [
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.Lambda(divide_255),
            transforms.Lambda(repeat),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def _build_openphenom_transform(bit_depth=8):
    """Build transform for OpenPhenom.

    Both 8-bit and 16-bit: Resize to 256×256. Model pos_embed supports max
    11 channels × 16² patches = 2816 + 1 cls = 2817 tokens. At 256×256 with
    5 channels: 5×256 + 1 = 1281 ≤ 2817 → fits.
    16-bit: illumination thresholding converts to 0-255 range first (proper
    contrast stretching instead of naive /65535 scaling).
    """
    if bit_depth == 16:
        return transforms.Compose(
            [
                transforms.Lambda(illumination_threshold_tensor),
                transforms.Resize((256, 256)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
        ]
    )


def _to_float(x):
    """Convert tensor to float. Module-level for picklability with mp.spawn."""
    return x.float()


def _build_cloome_transform(bit_depth=8):
    """Build transform for CLOOME.

    NOTE: CLOOME normalization requires all 5 channels, so per-channel Normalize
    must be applied AFTER the collate function stacks channels. The per-sample
    transform only converts to float here. The full normalization is applied
    in the collate function via _get_cloome_post_collate_transform().

    For 16-bit data (BBBC036): converts each channel from 0-65535 to 0-255
    using illumination thresholding (cap top 0.01% brightest pixels, then
    scale to 8-bit range). This matches the original CLOOME preprocessing.
    """
    if bit_depth == 16:
        return transforms.Lambda(illumination_threshold_tensor)
    return transforms.Lambda(_to_float)


def _get_cloome_post_collate_transform(bit_depth=8, dataset_name=None):
    """Post-collate transform for CLOOME: CenterCrop + per-channel Normalize.
    Applied after the collate function has concatenated 5 channels.

    CPG0016: CenterCrop(996) — crop to usable area, no zero-padding.
    BBBC036: CenterCrop(520) — crop to smaller dim.

    For 16-bit data, per-channel 16→8-bit conversion already happened in
    _build_cloome_transform, so values are already in 0-255 float range.
    """
    from models.cloome_wrapper import CLOOME_MEAN, CLOOME_STD

    crop_size = 520 if dataset_name == "bbbc036" else 996
    steps = [
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=CLOOME_MEAN, std=CLOOME_STD),
    ]
    return transforms.Compose(steps)


def _build_subcell_transform(bit_depth=8):
    """Build transform for SubCell.

    Resize to 448×448 and convert to float.
    MinMax normalization is applied inside SubCellWrapper.infer() per-batch
    AFTER channel selection, matching the SubCellPortable inference pipeline.
    For 16-bit data, illumination thresholding converts to 0-255 range first
    (proper contrast stretching instead of naive /65535 scaling).
    """
    steps = []
    if bit_depth == 16:
        steps.append(transforms.Lambda(illumination_threshold_tensor))
    steps.extend(
        [
            transforms.Resize((448, 448)),
            transforms.Lambda(_to_float),
        ]
    )
    return transforms.Compose(steps)


# ── Transform dispatch ────────────────────────────────────────────────────────

_TRANSFORM_BUILDERS = {
    "dino_v2": _build_classic_transform,
    "dino_v2_448": _build_dinov2_448_transform,
    "resnet": _build_classic_transform,
    "resnet_untrained": _build_classic_transform,
    "vgg19": _build_classic_transform,
    "open_phenom": _build_openphenom_transform,
    "cloome": _build_cloome_transform,
    "subcell": _build_subcell_transform,
}


def get_transform(model_name, cfg=None):
    """Get the appropriate preprocessing transform for a given model."""
    cfg = cfg or get_active_config()
    bit_depth = cfg["dataset"]["bit_depth"]
    builder = _TRANSFORM_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown model: {model_name}")
    return builder(bit_depth)


# ── Model wrapper constructor ─────────────────────────────────────────────────


def get_model_wrapper(model_name):
    """Get the model wrapper class for a given model name."""
    cls = MODEL_WRAPPERS.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return cls


# ── Collate functions ─────────────────────────────────────────────────────────

# BBBC036 channel name → canonical index (same sorted order as CPG0016)
# Canonical order: [AGP(0), DNA(1), ER(2), Mito(3), RNA(4)]
BBBC036_CHANNEL_MAP = {
    "Ph_golgi": 0,   # AGP
    "Hoechst": 1,    # DNA
    "ERSyto": 2,     # ER
    "Mito": 3,       # Mito
    "ERSytoBleed": 4, # RNA
}


def _parse_filename(filename):
    """Parse well, FOV, and channel index from either CPG0016 or BBBC036 filenames.

    CPG0016:  A01_i1_ch0.png → well='A01', fov='1', ch_idx=0
    BBBC036:  cdp2bioactives_a01_s1_w1UUID_ch_Hoechst.png → well='a01', fov='1', ch_idx=1
    """
    # Detect BBBC036 by "_ch_" delimiter (handles channel names with underscores like Ph_golgi)
    if "_ch_" in filename:
        # BBBC036: cdp2bioactives_a01_s1_wUUID_ch_Hoechst.png
        parts = filename.split("_")
        well = parts[1]
        fov = parts[2].replace("s", "")
        ch_name = filename.split("_ch_")[1].replace(".png", "")
        ch_idx = BBBC036_CHANNEL_MAP.get(ch_name, -1)
        return well, fov, ch_idx
    else:
        # CPG0016: A01_i1_ch0.png
        parts = filename.split("_")
        well = parts[0]
        fov = parts[1].replace("i", "")
        ch_idx = int(parts[2].replace("ch", "").split(".")[0])
        return well, fov, ch_idx


def collate_cpjump(batch):
    """Collate function for OpenPhenom: concatenates 5 channels per FOV.

    HF dataset serves individual channel images in order: ch0(AGP), ch1(DNA),
    ch2(ER), ch3(Mito), ch4(RNA). Every 5 consecutive samples belong to the
    same well/FOV. This collate concatenates them into (5, H, W) per FOV.
    """
    n_fovs = len(batch) // 5
    images = []
    well_ids = []
    plate_ids = []
    for i in range(n_fovs):
        fov_channels = torch.cat(
            [batch[i * 5 + j]["image"] for j in range(5)], dim=0
        )  # (5, H, W)
        images.append(fov_channels)
        well_ids.append(batch[i * 5]["filename"])
        plate_ids.append(batch[i * 5]["plate_name"])
    return {
        "image": torch.stack(images),  # (n_fovs, 5, H, W)
        "filename": well_ids,
        "plate_name": plate_ids,
    }


def channel_concat_collate_fn(batch, post_transform=None, channel_reorder=None):
    """Collate function that concatenates channels from multiple samples into one tensor.

    Groups samples by (plate, well, FOV), sorts by channel index, concatenates.
    After concatenation the channel order is: [AGP(0), DNA(1), ER(2), Mito(3), RNA(4)].
    Handles both CPG0016 and BBBC036 filename formats via _parse_filename().

    Args:
        batch: List of sample dicts with 'image', 'filename', 'plate_name'.
        post_transform: Optional transform applied after concatenation (e.g., Normalize).
        channel_reorder: Optional list of indices to reorder channels before post_transform.
            E.g., [3, 2, 4, 0, 1] converts sorted order to CLOOME training order.
    """
    groups = {}
    for sample in batch:
        filename = sample["filename"]
        well, fov, ch_idx = _parse_filename(filename)
        plate = sample["plate_name"]
        key = f"{plate}_{well}_{fov}"
        if key not in groups:
            groups[key] = []
        groups[key].append((ch_idx, sample))

    new_batch = []
    for key, items in groups.items():
        if len(items) < 5:
            continue  # skip incomplete FOV groups (batch boundary split)
        items.sort(key=lambda x: x[0])
        concatenated_image = torch.cat([s["image"] for _, s in items], dim=0)
        if channel_reorder is not None:
            concatenated_image = concatenated_image[channel_reorder]
        if post_transform is not None:
            concatenated_image = post_transform(concatenated_image)
        new_sample = {
            "image": concatenated_image,
            "filename": key,
            "plate_name": items[0][1]["plate_name"],
        }
        new_batch.append(new_sample)

    if not new_batch:
        return None
    return default_collate(new_batch)


# ── DataLoader settings builder ───────────────────────────────────────────────


def get_dataloader_settings(model_name, cfg=None):
    """Build dataloader kwargs from config for a given model."""
    cfg = cfg or get_active_config()
    model_cfg = cfg["models"][model_name]
    inference_cfg = cfg.get("inference", {})

    settings = {
        "batch_size": model_cfg.get("batch_size", 64),
        "shuffle": False,
        "num_workers": model_cfg.get("num_workers", 4),
        "pin_memory": inference_cfg.get("pin_memory", True),
    }

    # prefetch_factor is only valid when num_workers > 0
    if settings["num_workers"] > 0:
        settings["prefetch_factor"] = inference_cfg.get("prefetch_factor", 2)

    # Model-specific collate functions
    if model_name == "open_phenom":
        settings["collate_fn"] = collate_cpjump
    elif model_name == "cloome":
        bit_depth = cfg["dataset"]["bit_depth"]
        dataset_name = cfg["dataset"].get("name")
        post_transform = _get_cloome_post_collate_transform(bit_depth, dataset_name)
        # Reorder from sorted [AGP,DNA,ER,Mito,RNA] to CLOOME training order [Mito,ER,RNA,AGP,DNA]
        settings["collate_fn"] = partial(
            channel_concat_collate_fn,
            post_transform=post_transform,
            channel_reorder=[3, 2, 4, 0, 1],
        )
    elif model_name == "subcell":
        settings["collate_fn"] = channel_concat_collate_fn

    return settings


# ── Direct-mode transforms (channels pre-stacked in dataset) ──────────────────


# Reorder indices: sorted [AGP,DNA,ER,Mito,RNA] → CLOOME training [Mito,ER,RNA,AGP,DNA]
CLOOME_CHANNEL_REORDER = [3, 2, 4, 0, 1]


def _cloome_reorder_channels(x):
    """Reorder from canonical [AGP,DNA,ER,Mito,RNA] to CLOOME training order."""
    return x[CLOOME_CHANNEL_REORDER]


def _sixteen_to_eight_multichannel(x):
    """Convert multi-channel 16-bit tensor (C, H, W) to 8-bit float range (0-255).
    Uses per-channel illumination thresholding (cap top 0.01% brightest pixels).
    """
    result = torch.empty_like(x, dtype=torch.float32)
    for c in range(x.shape[0]):
        result[c] = illumination_threshold_tensor(x[c].unsqueeze(0)).squeeze(0)
    return result


def _build_direct_cloome_transform(bit_depth=8):
    """Direct-mode transform for CLOOME: reorder channels → 16→8-bit (if needed) → CenterCrop → Normalize.

    CPG0016 (8-bit, 1080×1080): CenterCrop(1024) — matches original CLOOME training.
    BBBC036 (16-bit, 696×520): 16→8-bit conversion + CenterCrop(520).
    """
    from models.cloome_wrapper import CLOOME_MEAN, CLOOME_STD

    crop_size = 520 if bit_depth == 16 else 1024
    steps = [transforms.Lambda(_cloome_reorder_channels)]

    if bit_depth == 16:
        steps.append(transforms.Lambda(_sixteen_to_eight_multichannel))
    else:
        steps.append(transforms.Lambda(_to_float))

    steps.extend([
        transforms.CenterCrop(crop_size),
        transforms.Normalize(mean=CLOOME_MEAN, std=CLOOME_STD),
    ])
    return transforms.Compose(steps)


_DIRECT_TRANSFORM_BUILDERS = {
    "dino_v2": _build_classic_transform,
    "dino_v2_448": _build_dinov2_448_transform,
    "resnet": _build_classic_transform,
    "resnet_untrained": _build_classic_transform,
    "vgg19": _build_classic_transform,
    "open_phenom": _build_openphenom_transform,
    "cloome": _build_direct_cloome_transform,
    "subcell": _build_subcell_transform,
}


def get_direct_transform(model_name, cfg=None):
    """Get preprocessing transform for direct-mode (channels pre-stacked in dataset)."""
    cfg = cfg or get_active_config()
    bit_depth = cfg["dataset"]["bit_depth"]
    builder = _DIRECT_TRANSFORM_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Unknown model: {model_name}")
    return builder(bit_depth)


def get_direct_dataloader_settings(model_name, cfg=None):
    """Build dataloader kwargs for direct-mode (no custom collate needed).

    For multi-channel models, divides batch_size by 5 since the dataset
    now returns one 5-channel item per FOV instead of 5 single-channel items.
    """
    cfg = cfg or get_active_config()
    model_cfg = cfg["models"][model_name]
    inference_cfg = cfg.get("inference", {})

    batch_size = model_cfg.get("batch_size", 64)

    # Multi-channel models: dataset already stacks channels, so batch_size
    # represents FOVs directly (was previously inflated 5x for collate grouping)
    if model_name in ("open_phenom",):
        batch_size = max(1, batch_size // 5)

    settings = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": model_cfg.get("num_workers", 4),
        "pin_memory": inference_cfg.get("pin_memory", True),
    }

    if settings["num_workers"] > 0:
        settings["prefetch_factor"] = inference_cfg.get("prefetch_factor", 2)

    # No custom collate needed — channels pre-stacked in dataset
    return settings


# ── Enabled models helper ─────────────────────────────────────────────────────


def get_enabled_models(cfg=None):
    """Return list of model names that are enabled in the config."""
    cfg = cfg or get_active_config()
    return [name for name, mcfg in cfg["models"].items() if mcfg.get("enabled", False)]


# ── Legacy helpers (preserved for backward compat) ────────────────────────────


def get_well_fovs(filenames):
    return ["_".join(filename.split("_")[:2]) for filename in filenames]


def iter_border_patches(width, height, patch_size):
    for x in range(0, width - patch_size + 1, patch_size):
        for y in range(0, height - patch_size + 1, patch_size):
            yield x, y


def patch_image(image_array, patch_size=256):
    _, width, height = image_array.shape
    output_patches = []
    for x, y in iter_border_patches(width, height, patch_size):
        patch = image_array[:, y : y + patch_size, x : x + patch_size].clone()
        output_patches.append(patch)
    return torch.stack(output_patches)
