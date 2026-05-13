"""Unit tests for the patching logic in ModelWrapper.

Tests cover:
  - Patch geometry (correct number, size, alignment)
  - Channel preservation
  - Averaging over patches (tensor and dict outputs)
  - Round-trip: patches tile back to the (center-cropped) original
  - Config validation (non-perfect-square rejected)
"""

import math
import pytest
import torch
import sys
import os

# Add src to path so we can import base_model_wrapper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base_model_wrapper import ModelWrapper


# ── Minimal concrete subclass for testing ────────────────────────────────────

class DummyWrapper(ModelWrapper):
    """Minimal ModelWrapper subclass for testing patching methods."""
    PATCH_ALIGNMENT = 1

    def _load_model(self):
        return torch.nn.Identity()

    def infer(self, data_input):
        return data_input.mean(dim=(2, 3))  # (B, C) global avg pool

    def postprocess(self, data_input, model_output):
        return {"features": model_output}


class DummyViTWrapper(ModelWrapper):
    """ModelWrapper with ViT-like alignment (patch token size 14)."""
    PATCH_ALIGNMENT = 14

    def _load_model(self):
        return torch.nn.Identity()

    def infer(self, data_input):
        return data_input.mean(dim=(2, 3))

    def postprocess(self, data_input, model_output):
        return {"features": model_output}


class DummyViT16Wrapper(ModelWrapper):
    """ModelWrapper with ViT-B/16 alignment."""
    PATCH_ALIGNMENT = 16

    def _load_model(self):
        return torch.nn.Identity()

    def infer(self, data_input):
        return data_input.mean(dim=(2, 3))

    def postprocess(self, data_input, model_output):
        return {"features": model_output}


def _make_wrapper(cls, num_patches=4, patching_enabled=True):
    """Helper to create a wrapper without a real dataset/dataloader."""
    import logging
    logger = logging.getLogger("test")

    class _Wrap(cls):
        def __init__(self, **kwargs):
            # Skip dataset/dataloader setup — just init patching fields
            self.device = "cpu"
            self.model = self._load_model()
            self.patching_enabled = kwargs.get("patching_enabled", True)
            self.num_patches = kwargs.get("num_patches", 4)
            self.logger = kwargs.get("logger", logger)
            self.use_mixed_precision = False
            # Run the same validation as ModelWrapper.__init__
            if self.patching_enabled:
                grid_dim = int(math.sqrt(self.num_patches))
                assert grid_dim * grid_dim == self.num_patches, \
                    f"num_patches must be a perfect square, got {self.num_patches}"

    return _Wrap(patching_enabled=patching_enabled, num_patches=num_patches, logger=logger)


# ── Tests: Patch geometry ─────────────────────────────────────────────────────

class TestPatchGeometry:
    """Verify _patch_batch produces correct shapes and counts."""

    def test_4_patches_no_alignment(self):
        w = _make_wrapper(DummyWrapper, num_patches=4)
        images = torch.randn(2, 3, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape == (2 * 4, 3, 540, 540)

    def test_9_patches_no_alignment(self):
        w = _make_wrapper(DummyWrapper, num_patches=9)
        images = torch.randn(2, 3, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape == (2 * 9, 3, 360, 360)

    def test_16_patches_no_alignment(self):
        w = _make_wrapper(DummyWrapper, num_patches=16)
        images = torch.randn(1, 3, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape == (16, 3, 270, 270)

    def test_4_patches_vit14_alignment(self):
        """DINOv2: alignment=14, image 1080 → patches 532×532 (38×14)."""
        w = _make_wrapper(DummyViTWrapper, num_patches=4)
        images = torch.randn(2, 3, 1080, 1080)
        patches = w._patch_batch(images)
        # 1080 // 2 = 540, 540 // 14 = 38, 38 * 14 = 532
        assert patches.shape == (8, 3, 532, 532)
        assert patches.shape[2] % 14 == 0
        assert patches.shape[3] % 14 == 0

    def test_4_patches_vit16_alignment(self):
        """OpenPhenom/SubCell: alignment=16, image 1080 → patches 528×528 (33×16)."""
        w = _make_wrapper(DummyViT16Wrapper, num_patches=4)
        images = torch.randn(2, 5, 1080, 1080)
        patches = w._patch_batch(images)
        # 1080 // 2 = 540, 540 // 16 = 33, 33 * 16 = 528
        assert patches.shape == (8, 5, 528, 528)
        assert patches.shape[2] % 16 == 0

    def test_4_patches_5_channels(self):
        """Multi-channel (e.g. CLOOME/SubCell) preserves channel count."""
        w = _make_wrapper(DummyWrapper, num_patches=4)
        images = torch.randn(3, 5, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape == (12, 5, 540, 540)

    def test_single_channel(self):
        """Single-channel images (before repeat) preserve channel count."""
        w = _make_wrapper(DummyWrapper, num_patches=4)
        images = torch.randn(2, 1, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape == (8, 1, 540, 540)


# ── Tests: Patch content (round-trip) ─────────────────────────────────────────

class TestPatchContent:
    """Verify patches tile back to the original (center-cropped) image."""

    def test_round_trip_no_alignment(self):
        w = _make_wrapper(DummyWrapper, num_patches=4)
        images = torch.randn(1, 3, 1080, 1080)
        patches = w._patch_batch(images)

        # Reconstruct: (4, 3, 540, 540) → (1, 3, 1080, 1080)
        grid_dim = 2
        pH, pW = patches.shape[2], patches.shape[3]
        recon = patches.reshape(1, grid_dim, grid_dim, 3, pH, pW)
        recon = recon.permute(0, 3, 1, 4, 2, 5).reshape(1, 3, grid_dim * pH, grid_dim * pW)
        assert torch.allclose(recon, images)

    def test_round_trip_with_alignment(self):
        """Patches tile back to the center-cropped (aligned) image."""
        w = _make_wrapper(DummyViTWrapper, num_patches=4)
        images = torch.randn(1, 3, 1080, 1080)
        patches = w._patch_batch(images)

        grid_dim = 2
        pH, pW = patches.shape[2], patches.shape[3]
        recon = patches.reshape(1, grid_dim, grid_dim, 3, pH, pW)
        recon = recon.permute(0, 3, 1, 4, 2, 5).reshape(1, 3, grid_dim * pH, grid_dim * pW)

        # The reconstructed image should match the center-cropped original
        target_h, target_w = pH * grid_dim, pW * grid_dim
        start_h = (1080 - target_h) // 2
        start_w = (1080 - target_w) // 2
        cropped = images[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
        assert torch.allclose(recon, cropped)

    def test_patches_are_non_overlapping(self):
        """Each patch covers a unique spatial region — no pixel appears in two patches."""
        w = _make_wrapper(DummyWrapper, num_patches=4)
        # Use a tensor where every pixel has a unique value
        images = torch.arange(1080 * 1080, dtype=torch.float32).reshape(1, 1, 1080, 1080)
        patches = w._patch_batch(images)

        all_values = patches.reshape(-1)
        unique_values = torch.unique(all_values)
        assert len(unique_values) == len(all_values), "Patches overlap — duplicate pixel values found"


# ── Tests: Averaging ──────────────────────────────────────────────────────────

class TestAveragePatchOutputs:
    """Verify _average_patch_outputs handles tensor and dict outputs."""

    def test_average_tensor(self):
        w = _make_wrapper(DummyWrapper, num_patches=4)
        # Simulate 2 images × 4 patches × 768-d features
        output = torch.randn(8, 768)
        averaged = w._average_patch_outputs(output, batch_size=2)
        assert averaged.shape == (2, 768)
        # Check that averaging is correct
        expected = output.view(2, 4, 768).mean(dim=1)
        assert torch.allclose(averaged, expected)

    def test_average_dict_with_tensors(self):
        """DINOv2 returns dict with cls_token and patch_tokens."""
        w = _make_wrapper(DummyWrapper, num_patches=4)
        output = {
            "x_norm_clstoken": torch.randn(8, 768),
            "x_norm_patchtokens": torch.randn(8, 256, 768),
        }
        averaged = w._average_patch_outputs(output, batch_size=2)
        assert averaged["x_norm_clstoken"].shape == (2, 768)
        assert averaged["x_norm_patchtokens"].shape == (2, 256, 768)

    def test_average_dict_preserves_non_tensors(self):
        w = _make_wrapper(DummyWrapper, num_patches=4)
        output = {
            "features": torch.randn(8, 512),
            "metadata": "some_string",
        }
        averaged = w._average_patch_outputs(output, batch_size=2)
        assert averaged["features"].shape == (2, 512)
        assert averaged["metadata"] == "some_string"


# ── Tests: Config validation ──────────────────────────────────────────────────

class TestConfigValidation:
    """Verify that invalid configurations raise errors."""

    def test_non_perfect_square_raises(self):
        with pytest.raises(AssertionError, match="perfect square"):
            _make_wrapper(DummyWrapper, num_patches=5)

    def test_non_perfect_square_6_raises(self):
        with pytest.raises(AssertionError, match="perfect square"):
            _make_wrapper(DummyWrapper, num_patches=6)

    def test_perfect_squares_pass(self):
        for n in [1, 4, 9, 16, 25]:
            w = _make_wrapper(DummyWrapper, num_patches=n)
            assert w.num_patches == n


# ── Tests: Channel consistency for multi-channel models ───────────────────────

class TestChannelConsistency:
    """Verify that patching preserves the correct number of channels."""

    @pytest.mark.parametrize("num_channels", [1, 3, 5, 6])
    def test_channel_count_preserved(self, num_channels):
        w = _make_wrapper(DummyWrapper, num_patches=4)
        images = torch.randn(2, num_channels, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape[1] == num_channels

    def test_5ch_with_vit16_alignment(self):
        """5-channel images with ViT-B/16 alignment (OpenPhenom, SubCell)."""
        w = _make_wrapper(DummyViT16Wrapper, num_patches=4)
        images = torch.randn(2, 5, 1080, 1080)
        patches = w._patch_batch(images)
        assert patches.shape[1] == 5
        assert patches.shape[2] % 16 == 0
        assert patches.shape[3] % 16 == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
