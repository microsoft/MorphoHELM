"""Direct image loading dataset for Cell Painting feature extraction.

Loads images directly from PNGs on disk/blob storage, bypassing HuggingFace datasets.
Pre-filters to only relevant compounds via metadata.parquet for ~4x efficiency gain.
"""

import os
import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Canonical channel order matching HF dataset (sorted by ch#)
CHANNEL_COLS = ["agp_path", "dna_path", "er_path", "mito_path", "rna_path"]

# Multi-channel models that receive all 5 channels stacked per FOV
MULTI_CHANNEL_MODELS = {"open_phenom", "cloome", "subcell"}


class DirectImageDataset(Dataset):
    """PyTorch Dataset that loads Cell Painting images directly from disk.

    Supports two modes:
      - Multi-channel (open_phenom, cloome, subcell): stacks 5 channels → (5, H, W)
        per FOV. One dataset item per metadata row.
      - Single-channel (dino_v2, resnet, etc.): returns (1, H, W) per channel.
        Five dataset items per metadata row.

    Channel order for stacked output: [AGP, DNA, ER, Mito, RNA] — same as HF dataset.
    Any model-specific reordering (e.g., CLOOME) should be done in transforms.

    Args:
        metadata_df: DataFrame filtered to relevant compounds. Must have columns:
            Metadata_Plate, Metadata_Well, FOV, agp_path, dna_path, er_path,
            mito_path, rna_path.
        image_root: Root directory for image paths (paths in metadata are relative).
        transform: Optional torchvision transform applied to each image tensor.
        model_name: Determines multi-channel vs single-channel mode.
    """

    def __init__(self, metadata_df, image_root, transform=None, model_name="dino_v2"):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.model_name = model_name
        self.stack_channels = model_name in MULTI_CHANNEL_MODELS

        if self.stack_channels:
            self._len = len(self.metadata_df)
        else:
            self._len = len(self.metadata_df) * len(CHANNEL_COLS)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.stack_channels:
            return self._get_stacked_item(idx)
        else:
            return self._get_single_channel_item(idx)

    def _load_image(self, img_path):
        """Load a PNG image and return as uint8 tensor (1, H, W)."""
        full_path = os.path.join(self.image_root, img_path)
        img = Image.open(full_path)
        arr = np.array(img)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W), uint8

    def _get_stacked_item(self, idx):
        """Load and stack all 5 channels for multi-channel models."""
        row = self.metadata_df.iloc[idx]

        channels = []
        for col in CHANNEL_COLS:
            ch_tensor = self._load_image(row[col])
            channels.append(ch_tensor)

        # (5, H, W) — canonical order [AGP, DNA, ER, Mito, RNA]
        image = torch.cat(channels, dim=0)

        if self.transform:
            image = self.transform(image)

        # Filename: plate_well_fov format for aggregation compatibility
        plate = str(row["Metadata_Plate"])
        well = row["Metadata_Well"]
        fov = str(int(row["FOV"])) if not isinstance(row["FOV"], str) else row["FOV"]

        return {
            "image": image,
            "filename": f"{plate}_{well}_{fov}",
            "plate_name": plate,
        }

    def _get_single_channel_item(self, idx):
        """Load a single channel for single-channel models."""
        row_idx = idx // len(CHANNEL_COLS)
        channel_idx = idx % len(CHANNEL_COLS)

        row = self.metadata_df.iloc[row_idx]
        channel_col = CHANNEL_COLS[channel_idx]
        img_path = row[channel_col]

        image = self._load_image(img_path)  # (1, H, W), uint8

        if self.transform:
            image = self.transform(image)

        # Use basename for aggregation compatibility: "A01_i1_ch0.png"
        filename = os.path.basename(img_path)

        return {
            "image": image,
            "filename": filename,
            "plate_name": str(row["Metadata_Plate"]),
        }
