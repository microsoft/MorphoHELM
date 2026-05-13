import torch.nn as nn
from torchvision import models

from .base_model_wrapper import ModelWrapper

class VGG19Wrapper(ModelWrapper):
    """Wrapper for a pretrained VGG19 model.

    Input: 3×224×224 ImageNet-normalized RGB (single-channel repeated to 3).
    Output: 4096-d feature vectors from the penultimate classifier layer.
    """
    def _load_model(self):
        print("Loading VGG19...")
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        model.classifier[6] = nn.Identity()
        return model    
    
    def infer(self, data_input):
        return self.model(data_input)
    
    def postprocess(self, data_input, model_output):
        return {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "features": model_output.cpu().numpy(),
        }