"""Illumination-correction helpers shared by dataset downloaders."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def image_bytes_to_array(payload: bytes) -> np.ndarray:
    """Decode image bytes into a NumPy array."""
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image)


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert an image-like array to uint8 without changing existing uint8 data."""
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scaled = array.astype(np.float32) / float(info.max)
        return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)

    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if finite.size == 0:
        return finite.astype(np.uint8)
    if finite.min() >= 0.0 and finite.max() <= 1.0:
        finite = finite * 255.0
    else:
        min_value = float(finite.min())
        max_value = float(finite.max())
        if max_value > min_value:
            finite = (finite - min_value) / (max_value - min_value) * 255.0
    return np.clip(np.rint(finite), 0, 255).astype(np.uint8)


def _divide_by_illumination(image: np.ndarray, illumination: np.ndarray) -> np.ndarray:
    illum = illumination.astype(np.float32)
    illum = np.where(illum == 0, 1.0, illum)
    return image.astype(np.float32) / illum


def correct_cpg0016_to_uint8(image: np.ndarray, illumination: np.ndarray) -> np.ndarray:
    """Apply the legacy CPG0016 correction and return a uint8 image."""
    corrected = _divide_by_illumination(image, illumination)
    vmin, vmax = np.percentile(corrected, (0.05, 99.95))
    if vmax > vmin:
        corrected = (corrected - vmin) / (vmax - vmin)
    corrected = np.clip(corrected, 0.0, 1.0)
    return to_uint8(corrected)


def correct_bbbc036_to_uint8(image: np.ndarray, illumination: np.ndarray) -> np.ndarray:
    """Apply the legacy BBBC036 correction and return a uint8 image."""
    corrected = _divide_by_illumination(image, illumination)
    min_value = float(np.min(corrected))
    max_value = float(np.max(corrected))
    if max_value > min_value:
        corrected = (corrected - min_value) / (max_value - min_value)
    corrected = np.clip(corrected, 0.0, 1.0)
    return to_uint8(corrected)


def save_uint8_png(path, image: np.ndarray) -> None:
    """Save a uint8 grayscale image as PNG."""
    Image.fromarray(to_uint8(image)).save(path, format="PNG")

