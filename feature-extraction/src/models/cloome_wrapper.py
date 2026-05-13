import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from .base_model_wrapper import ModelWrapper
from .cloome.model import CLIPGeneral


# CLOOME per-channel normalization stats (from Cell Painting training data)
CLOOME_MEAN = [47.1314, 40.8138, 53.7692, 46.2656, 28.7243]
CLOOME_STD = [47.1314, 40.8138, 53.7692, 46.2656, 28.7243]

# Path to bundled model config
_CLOOME_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_CONFIG = os.path.join(_CLOOME_DIR, "cloome", "RN50.json")


class CloomeWrapper(ModelWrapper):
    """Wrapper for the CLOOME model (Contrastive Learning of Optical microscopy and Omics Expression).

    CLOOME encodes Cell Painting images into a shared embedding space with molecular structures.
    It uses a modified ResNet-50 backbone (by default) that accepts 5-channel microscopy images.

    Preprocessing:
      - 16-bit images are converted to 8-bit using illumination thresholding
        (removes top 0.01% brightest pixels to set display_max).
      - 5 channels are stacked and passed through: CenterCrop(1024) → per-channel Normalize.
      - Channel order: Mito, ERSyto, ERSytoBleed, Ph_golgi, Hoechst
        (i.e., Mito, ER, RNA, AGP, DNA — reordered from sorted ch0-ch4 by collate).

    Input: 5×1024×1024 normalized tensor (preprocessing applied externally via config transform).
    Output: 512-d L2-normalized embedding vectors.

    Self-contained: CLOOME model code and config are vendored under models/cloome/.
    Only the checkpoint (.pt file) must be provided externally.

    Required config fields:
      - checkpoint: path to CLOOME .pt checkpoint file
      - model_config: (optional) path to model config JSON; defaults to bundled RN50.json
    """

    def __init__(self, checkpoint_path, model_config_path=None, **kwargs):
        self.checkpoint_path = checkpoint_path
        self.model_config_path = model_config_path or DEFAULT_MODEL_CONFIG
        # Drop legacy repo_path if passed from config
        kwargs.pop("repo_path", None)
        super().__init__(**kwargs)

    def _load_model(self):
        print("Loading CLOOME model...")

        assert os.path.exists(self.model_config_path), \
            f"CLOOME config not found: {self.model_config_path}"
        with open(self.model_config_path, 'r') as f:
            model_info = json.load(f)

        model = CLIPGeneral(**model_info)

        assert os.path.exists(self.checkpoint_path), \
            f"CLOOME checkpoint not found: {self.checkpoint_path}"
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint["state_dict"]
        # Remove 'module.' prefix from DataParallel state dict keys
        new_state_dict = {k[len('module.'):] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        return model

    def infer(self, data_input):
        return self.model.encode_image(data_input)

    def postprocess(self, data_input, model_output):
        # L2 normalize embeddings (F.normalize handles zero-norm gracefully)
        embedding = F.normalize(model_output, dim=-1)
        return {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "features": embedding.cpu().numpy(),
        }
