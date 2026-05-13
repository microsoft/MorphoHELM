"""Convenience imports for model wrappers.

Allows: from models import ResNetWrapper, DINOv2Wrapper, OpenPhenomWrapper, ...
"""

from .base_model_wrapper import ModelWrapper
from .resnet101_wrapper import ResNetWrapper
from .resnet101_untrained_wrapper import ResNetUntrainedWrapper
from .vgg19_wrapper import VGG19Wrapper
from .dinov2_wrapper import DINOv2HighRes448Wrapper, DINOv2Wrapper
from .OpenPhenom_wrapper import OpenPhenomWrapper
from .cloome_wrapper import CloomeWrapper
from .subcell_wrapper import SubCellWrapper

__all__ = [
    "ModelWrapper",
    "ResNetWrapper",
    "ResNetUntrainedWrapper",
    "VGG19Wrapper",
    "DINOv2Wrapper",
    "DINOv2HighRes448Wrapper",
    "OpenPhenomWrapper",
    "CloomeWrapper",
    "SubCellWrapper",
]
