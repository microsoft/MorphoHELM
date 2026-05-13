import numpy as np
import torch


def divide_255(x):
    """Convert 0-255 range to 0-1."""
    return x.float() / 255.0


def divide_65535(x):
    """Convert 0-65535 (16-bit) range to 0-1."""
    return x.float() / 65535.0


def multiply_255(x):
    """Scale 0-1 range to 0-255."""
    return x * 255.0


def normalize(x):
    """Z-score normalization (mean=0, std=1)."""
    return (x - x.mean()) / (x.std() + 1e-7)


def repeat(x):
    """Repeat single-channel grayscale to 3-channel RGB."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def to_float(x):
    """Convert tensor to float32."""
    return x.float()


def minmax_normalize(x):
    """MinMax normalize to [0, 1] range across spatial dims."""
    x = x.float()
    vmin = x.amin(dim=(-2, -1), keepdim=True)
    vmax = x.amax(dim=(-2, -1), keepdim=True)
    return (x - vmin) / (vmax - vmin + 1e-6)


def illumination_threshold(arr, perc=0.01):
    """Return threshold value to exclude the top `perc`% of brightest pixels.
    Used for CLOOME 16-bit to 8-bit conversion.
    """
    perc = perc / 100
    total_pixels = arr.shape[0] * arr.shape[1]
    n_pixels = max(1, int(np.around(total_pixels * perc)))
    flat_inds = np.argpartition(arr.ravel(), -n_pixels)[-n_pixels:]
    return arr.ravel()[flat_inds].min()


def sixteen_to_eight_bit(arr, display_max, display_min=0):
    """Convert 16-bit image array to 8-bit using illumination thresholding."""
    threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))
    scaled_image = threshold_image * (255 / (display_max - display_min + 1e-7))
    scaled_image[scaled_image > 255] = 255
    return scaled_image.astype(np.uint8)


def illumination_threshold_tensor(x, perc=0.01):
    """Convert a single-channel 16-bit tensor (1, H, W) to 8-bit float range (0-255).

    Matches original CLOOME preprocessing: cap the top `perc`% brightest pixels
    to set display_max, then linearly scale to 0-255.

    Args:
        x: Tensor of shape (1, H, W), 16-bit values (0-65535).
        perc: Percentage of brightest pixels to exclude (default 0.01%).

    Returns:
        Float tensor (1, H, W) in 0-255 range.
    """
    x = x.float()
    flat = x.reshape(-1)
    n_pixels = max(1, int(round(flat.numel() * perc / 100)))
    # kth_value gives the smallest of the top n_pixels values
    display_max = flat.kthvalue(flat.numel() - n_pixels + 1).values.item()
    display_max = max(display_max, 1.0)  # avoid division by zero
    scaled = x * (255.0 / display_max)
    return scaled.clamp(0, 255)


def get_plate_chunks(num_gpus, chosen_plates):
    """Split plates into roughly equal parts among GPUs."""
    n = len(chosen_plates)
    base_chunk_size = n // num_gpus
    remainder = n % num_gpus

    plate_chunks = []
    start = 0
    for i in range(num_gpus):
        chunk_size = base_chunk_size + (1 if i < remainder else 0)
        end = start + chunk_size
        plate_chunks.append(chosen_plates[start:end])
        start = end

    for i, chunk in enumerate(plate_chunks):
        print(f"Chunk {i}: {len(chunk)} plates")
    return plate_chunks