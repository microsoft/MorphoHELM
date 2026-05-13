import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Union

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from .base_model_wrapper import ModelWrapper


# ── GatedAttentionPooler (from SubCellPortable) ─────────────────────────────
class GatedAttentionPooler(nn.Module):
    """Gated attention pooling for ViT outputs."""
    def __init__(self, dim: int, int_dim: int = 512, num_heads: int = 1, out_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.attention_v = nn.Sequential(nn.Linear(dim, int_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, int_dim), nn.GELU())
        self.attention = nn.Linear(int_dim, num_heads)
        self.softmax = nn.Softmax(dim=-1)
        if out_dim is None:
            self.out_dim = dim * num_heads
            self.out_proj = nn.Identity()
        else:
            self.out_dim = out_dim
            self.out_proj = nn.Linear(dim * num_heads, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.attention_v(x)
        u = self.attention_u(x)
        attn = self.attention(v * u).permute(0, 2, 1)
        attn = self.softmax(attn)
        x = torch.bmm(attn, x)
        x = x.view(x.shape[0], -1)
        x = self.out_proj(x)
        return x, attn


# ── SubCell ViT with attention pooling ───────────────────────────────────────
class SubCellViT(nn.Module):
    """SubCell ViT encoder + GatedAttentionPooler for feature extraction."""
    def __init__(self, vit_config: dict, pool_config: dict):
        super().__init__()
        config = ViTConfig(**vit_config, attn_implementation='eager')
        self.encoder = ViTModel(config, add_pooling_layer=False)
        self.pool_model = GatedAttentionPooler(**pool_config)
        self.out_dim = self.pool_model.out_dim

    def load_subcell_weights(self, encoder_path: str, device='cpu'):
        """Load encoder + pool_model weights from SubCellPortable checkpoint."""
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
        encoder_ckpt = {
            k[len("encoder."):]: v for k, v in checkpoint.items() if k.startswith("encoder.")
        }
        status = self.encoder.load_state_dict(encoder_ckpt, strict=False)
        print(f"SubCell encoder status: {status}")

        pool_ckpt = {
            k.replace("pool_model.", ""): v
            for k, v in checkpoint.items() if "pool_model." in k
        }
        # SubCellPortable applies key remapping for pool weights
        pool_ckpt = {k.replace("1.", "0."): v for k, v in pool_ckpt.items()}
        if pool_ckpt:
            status = self.pool_model.load_state_dict(pool_ckpt, strict=False)
            print(f"SubCell pool_model status: {status}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, interpolate_pos_encoding=True)
        pool_op, _ = self.pool_model(outputs.last_hidden_state)
        return pool_op


class SubCellWrapper(ModelWrapper):
    """Wrapper for the SubCell model.

    SubCell is a ViT-based model trained on Human Protein Atlas images.
    For Cell Painting data, we use the ER-DNA-Protein (ybg) variant:
      - ER and DNA serve as reference channels
      - Each remaining channel (RNA, AGP, Mito) is profiled as the "Protein" channel
      - The model is run 3 times and embeddings are concatenated

    This approach follows the methodology in the SubCell paper
    (https://www.biorxiv.org/content/10.1101/2024.12.06.627299v1).

    Architecture: ViT-B/16 with GatedAttentionPooler (2 attention heads).

    Preprocessing:
      - Resize to 448×448 (done by transform in config.py)
      - MinMax normalize each 3ch input to [0,1] per sample (done here)

    Input per pass: 3×448×448 [ER(yellow), DNA(blue), Protein(green)]
    Output per pass: 1536-d (768 hidden × 2 attention heads)
    Total output: 3 passes × 1536-d = 4608-d concatenated embeddings

    Required config fields:
      - encoder_path: path to SubCell encoder.pth file
    """
    PATCH_ALIGNMENT = 16  # ViT-B/16 token patch size

    # Cell Painting channel mapping to SubCell ybg order
    # After channel_concat_collate_fn sorts by ch#: 0=AGP, 1=DNA, 2=ER, 3=Mito, 4=RNA
    # ybg model: [ER(y), DNA(b), Protein(g)]
    ER_CHANNEL = 2   # yellow (y) — ch2 in sorted order
    DNA_CHANNEL = 1  # blue (b) — ch1 in sorted order
    PROFILING_CHANNELS = [0, 3, 4]  # AGP, Mito, RNA → each serves as Protein (g)

    # ybg model architecture config
    VIT_CONFIG = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "image_size": 448,
        "patch_size": 16,
        "num_channels": 3,
        "qkv_bias": True,
    }
    POOL_CONFIG = {
        "dim": 768,
        "int_dim": 512,
        "num_heads": 2,
    }

    def __init__(self, encoder_path, **kwargs):
        # Remove classifier_path and portable_repo_path if passed (not needed for feature extraction)
        kwargs.pop("classifier_path", None)
        kwargs.pop("portable_repo_path", None)
        self.encoder_path = encoder_path
        super().__init__(**kwargs)

    def _load_model(self):
        print("Loading SubCell model...")
        assert os.path.exists(self.encoder_path), \
            f"SubCell encoder not found: {self.encoder_path}"

        model = SubCellViT(self.VIT_CONFIG, self.POOL_CONFIG)
        model.load_subcell_weights(self.encoder_path)
        return model

    @staticmethod
    def min_max_standardize(im):
        """Per-sample MinMax normalization to [0,1]."""
        min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
        max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)
        return (im - min_val) / (max_val - min_val + 1e-6)

    def infer(self, data_input):
        """Run 3-pass inference: once per profiling channel.

        data_input: (B, C, H, W) tensor with all 5 channels stacked.
        Cell Painting channels (sorted by ch#): 0=AGP, 1=DNA, 2=ER, 3=Mito, 4=RNA.
        """
        embeddings_list = []

        for prof_ch in self.PROFILING_CHANNELS:
            # Build 3-channel input in ybg order: [ER, DNA, Protein]
            three_ch_input = torch.stack([
                data_input[:, self.ER_CHANNEL],   # ER (yellow)
                data_input[:, self.DNA_CHANNEL],   # DNA (blue)
                data_input[:, prof_ch],            # profiling channel (green)
            ], dim=1)  # (B, 3, H, W)

            three_ch_input = self.min_max_standardize(three_ch_input)

            pool_op = self.model(three_ch_input)
            # pool_op: (B, 1536) from GatedAttentionPooler with num_heads=2
            embeddings_list.append(pool_op)

        # Concatenate: (B, 3×1536) = (B, 4608)
        return torch.cat(embeddings_list, dim=1)

    def postprocess(self, data_input, model_output):
        return {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "features": model_output.cpu().numpy(),
        }
