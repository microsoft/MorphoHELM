from .base_model_wrapper import ModelWrapper
from .openphenom.huggingface_mae import MAEModel


class OpenPhenomWrapper(ModelWrapper):
    """Wrapper for the OpenPhenom MAE model from Recursion Pharma.

    Input: 5×256×256 images in 0-255 range (model applies internal ÷255 + InstanceNorm).
    For BBBC036 (16-bit): preprocess with ÷65535 × 255 to get 0-255 range.
    Output: 384-d feature vectors (average of all patch tokens).

    Self-contained: all OpenPhenom source code is vendored under models/openphenom/.
    Weights are automatically downloaded from HuggingFace Hub (recursionpharma/OpenPhenom).
    """
    PATCH_ALIGNMENT = 16  # MAE ViT patch size
    def __init__(self, *args, **kwargs):
        # Drop legacy repo_path if passed from config
        kwargs.pop("repo_path", None)
        super().__init__(*args, **kwargs)

    def _load_model(self):
        print("Loading OpenPhenom model...")
        model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
        return model

    def infer(self, data_input):
        return self.model.predict(data_input)

    def postprocess(self, data_input, model_output):
        return {
            "filename": data_input["filename"],
            "plate_name": data_input["plate_name"],
            "features": model_output.cpu().numpy(),
        }