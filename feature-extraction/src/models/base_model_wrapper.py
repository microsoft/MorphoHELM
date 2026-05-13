import math
import sys
import threading

import torch
from torchvision import transforms
from tqdm import tqdm

import pickle
import os
import time
from abc import ABC, abstractmethod

from utils.huggingface_loader import DatasetWrapper

class ModelWrapper(ABC):
    """
    An abstract base class that defines the standard interface for all model wrappers.
    Each subclass is self-contained and implements its own model loading, inference, and postprocessing.
    """

    # Minimum spatial unit the model requires patches to be divisible by.
    # ViT models set this to their token patch size (e.g. 14, 16).
    # CNN models leave it at 1 (no constraint).
    PATCH_ALIGNMENT = 1

    def __init__(self, dataset, preprocess_transform, dataloader_settings, save_path, logger,
                 save_frequency=100, device='cpu', use_mixed_precision=True,
                 patching_enabled=False, num_patches=4):
        self.device = device
        self.model = self._load_model().to(self.device).eval()
        self.dataset = dataset
        self.preprocess_transform = preprocess_transform
        self.save_path = save_path
        self.logger = logger
        self.save_frequency = save_frequency
        self.dataloader_settings = dataloader_settings
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()

        self.patching_enabled = patching_enabled
        self.num_patches = num_patches
        if patching_enabled:
            grid_dim = int(math.sqrt(num_patches))
            assert grid_dim * grid_dim == num_patches, \
                f"num_patches must be a perfect square, got {num_patches}"
            self.logger.info(
                f"Patching enabled: {num_patches} patches ({grid_dim}×{grid_dim} grid), "
                f"alignment={self.PATCH_ALIGNMENT}"
            )

    @abstractmethod
    def _load_model(self):
        """Loads the pretrained model. Must be implemented by subclasses."""
        pass

    def preprocess(self):
        """Preprocess raw data inputs."""
        preprocessed_dataset = DatasetWrapper(
            original_dataset=self.dataset,
            transform=self.preprocess_transform
        )
        return preprocessed_dataset

    @abstractmethod
    def postprocess(self, data_input, model_output):
        """Extracts and formats the final prediction from the model's raw output."""
        pass
    
    @abstractmethod
    def infer(self, preprocessed_input):
        """Runs inference on the input data and returns the processed output."""
        pass

    def _patch_batch(self, images):
        """Split a batch of images into non-overlapping patches.

        Images are center-cropped to the nearest size divisible by
        (grid_dim × PATCH_ALIGNMENT), then reshaped into a grid of patches.

        Args:
            images: (B, C, H, W) tensor
        Returns:
            patches: (B * num_patches, C, pH, pW) tensor
        """
        B, C, H, W = images.shape
        grid_dim = int(math.sqrt(self.num_patches))
        a = self.PATCH_ALIGNMENT

        # Compute aligned patch dimensions
        raw_ph, raw_pw = H // grid_dim, W // grid_dim
        pH = (raw_ph // a) * a
        pW = (raw_pw // a) * a

        # Center-crop to aligned grid
        target_h, target_w = pH * grid_dim, pW * grid_dim
        start_h = (H - target_h) // 2
        start_w = (W - target_w) // 2
        images = images[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

        # Reshape into grid of patches
        patches = images.reshape(B, C, grid_dim, pH, grid_dim, pW)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.reshape(B * self.num_patches, C, pH, pW)
        return patches

    def _average_patch_outputs(self, output, batch_size):
        """Average model outputs over patches.

        Reshapes (B*P, ...) → (B, P, ...) and averages over the patch
        dimension for all tensor values. Non-tensor values are passed through.
        """
        P = self.num_patches
        if isinstance(output, dict):
            averaged = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    averaged[k] = v.view(batch_size, P, *v.shape[1:]).mean(dim=1)
                else:
                    averaged[k] = v
            return averaged
        return output.view(batch_size, P, *output.shape[1:]).mean(dim=1)

    @torch.inference_mode()
    def predict(self, data_input):
        """The main unified inference method with optional mixed precision and patching."""
        images = data_input['image'].to(self.device, non_blocking=True)
        B = images.shape[0]

        if self.patching_enabled:
            images = self._patch_batch(images)

        if self.use_mixed_precision:
            with torch.amp.autocast("cuda"):
                model_output = self.infer(images)
        else:
            model_output = self.infer(images)

        if self.patching_enabled:
            model_output = self._average_patch_outputs(model_output, B)

        return self.postprocess(data_input, model_output)

    def get_dataloader(self):
        preprocessed_dataset = self.preprocess()
        dataloader = torch.utils.data.DataLoader(
            preprocessed_dataset,
            **self.dataloader_settings
        )
        return dataloader

    def save_results(self, i, results):
        save_dir = os.path.join(self.save_path, f"gpu_{self.device}_results")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"results_part_{i}.pkl")
        with open(save_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_results_background(self, i, results):
        """Save results in a background thread to avoid blocking inference."""
        if hasattr(self, '_save_thread') and self._save_thread is not None:
            self._save_thread.join()
        self._save_thread = threading.Thread(
            target=self.save_results, args=(i, results)
        )
        self._save_thread.start()

    def run_inference(self):
        dataloader = self.get_dataloader()
        total_batches = len(dataloader)
        all_results = []
        start_time = time.time()
        self._save_thread = None

        pbar = tqdm(
            enumerate(dataloader),
            total=total_batches,
            desc=f"GPU{self.device}",
            file=sys.stdout,
            mininterval=30,
            dynamic_ncols=False,
            ncols=100,
        )
        for i, batch in pbar:
            if batch is None:
                continue  # skip empty batches (e.g. all unparseable filenames)
            result = self.predict(batch)
            all_results.append(result)

            if (i + 1) % self.save_frequency == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total_batches - (i + 1)) / rate if rate > 0 else 0
                self.logger.info(
                    f"Batch {i + 1}/{total_batches} | "
                    f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s"
                )
                self._save_results_background(i, all_results)
                all_results = []

        pbar.close()
        # Wait for any pending background save
        if self._save_thread is not None:
            self._save_thread.join()
        # Save any remaining results
        if all_results:
            self.save_results(total_batches, all_results)
            self.logger.info(f"Saved final {len(all_results)} batches")

        total_time = time.time() - start_time
        self.logger.info(f"Inference complete: {total_batches} batches in {total_time:.0f}s")
        return all_results