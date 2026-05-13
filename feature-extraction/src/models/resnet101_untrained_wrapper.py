import torch.nn as nn
from torchvision import models

from .base_model_wrapper import ModelWrapper


class ResNetUntrainedWrapper(ModelWrapper):
    """Wrapper for a randomly-initialized (untrained) ResNet-101 model.

    Architecture is identical to the pretrained ResNet-101 wrapper, but weights
    are random. Used as a baseline to compare learned vs random features.

    Input: 3×224×224 ImageNet-normalized RGB (single-channel repeated to 3).
    Output: 2048-d feature vectors from the penultimate layer.
    """
    def _load_model(self):
        print("Loading ResNet-101 (untrained)...")
        model = models.resnet101(weights=None)
        model.fc = nn.Identity()
        return model

    def infer(self, data_input):
        return self.model(data_input)

    def postprocess(self, data_input, model_output):
        return {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "features": model_output.cpu().numpy(),
        }
