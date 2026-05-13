import warnings

# Ignore the specific xFormers UserWarning
warnings.filterwarnings("ignore", message="xFormers is available")

import torch
import torchvision.transforms as transforms
from utils.util import *

from .base_model_wrapper import ModelWrapper

class DINOv2Wrapper(ModelWrapper):
    """Wrapper for a standard DINOv2 model."""
    PATCH_ALIGNMENT = 14  # ViT-B/14 token patch size
    def _load_model(self):
        print("Loading DINOv2...")
        model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
        return model            

    def infer(self, data_input):
        return self.model.forward_features(data_input)
    
    def postprocess(self, data_input, model_output):
        batch_results = {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "cls_token_features": model_output["x_norm_clstoken"].cpu().numpy(),
            "patch_token_features": torch.mean(model_output["x_norm_patchtokens"], dim=1).cpu().numpy()
        }
        return batch_results


class DINOv2HighRes448Wrapper(DINOv2Wrapper):
    """Distinct DINOv2 model identity for 448x448 inputs.

    The weights and forward pass are inherited from DINOv2Wrapper; the separate
    class keeps config, output folders, aggregation, and benchmark labels
    distinct from the existing 224x224 DINOv2 path.
    """

    INPUT_SIZE = 448
