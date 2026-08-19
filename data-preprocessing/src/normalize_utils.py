"""
Core normalization utilities for cell painting feature preprocessing.

Pipeline (configurable):
  - pre_pca mode:  CenterScale(plate, all wells) → PCA(128) → MAD(plate, controls) → Spherize(plate, controls)
  - post_pca mode: PCA(128) → CenterScale(plate, controls) → MAD(plate, controls) → Spherize(plate, controls)

Optimized: replaces pycytominer with direct numpy for ~10x speedup.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precompute plate/control indices (avoids repeated boolean masking)
# ---------------------------------------------------------------------------

def _build_plate_indices(plate_labels: np.ndarray, control_mask: np.ndarray):
    """Build per-plate row indices and control indices once.

    Returns list of (plate_id, all_idx, ctrl_idx) tuples.
    """
    plates = np.unique(plate_labels)
    result = []
    for p in plates:
        all_idx = np.where(plate_labels == p)[0]
        ctrl_idx = np.where((plate_labels == p) & control_mask)[0]
        result.append((p, all_idx, ctrl_idx))
    return result


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def embed_by_pca(
    features: np.ndarray,
    n_components: int = 128,
) -> np.ndarray:
    """PCA on the full dataset. Returns (N, n_components) float32 array.

    ``random_state`` and ``svd_solver`` are pinned so the transform is fully
    reproducible run-to-run (the default 'auto' solver picks the stochastic
    'randomized' SVD for these shapes).
    """
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    return pca.fit_transform(features).astype(np.float32)


# ---------------------------------------------------------------------------
# CenterScale (fit on controls, transform all)
# ---------------------------------------------------------------------------

def centerscale_by_plate(
    features: np.ndarray,
    plate_indices: list,
    fit_on: str = "controls",
    min_controls: int = 5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Center and scale features per plate.

    Parameters
    ----------
    features : (N, D) float32 array
    plate_indices : list of (plate_id, all_idx, ctrl_idx) from _build_plate_indices
    fit_on : "all" or "controls"
        "all"      — compute mean/std from ALL wells on the plate.
                     Removes plate-level batch effects without referencing controls.
                     Typically used before PCA so components capture biology, not batch.
        "controls" — compute mean/std from negative control wells only.
                     Z-scores each well relative to the untreated baseline, so
                     feature values represent deviations from no-treatment.
    min_controls : when fit_on="controls", skip plates with fewer controls than this
    eps : floor for std to avoid division by zero
    """
    out = features.copy()
    for plate_id, all_idx, ctrl_idx in plate_indices:
        if fit_on == "controls":
            if len(ctrl_idx) < min_controls:
                logger.warning("Plate %s: %d controls < %d — skipping CenterScale",
                               plate_id, len(ctrl_idx), min_controls)
                continue
            fit_data = features[ctrl_idx]
        else:  # fit_on == "all"
            fit_data = features[all_idx]

        mu = fit_data.mean(axis=0)
        std = fit_data.std(axis=0)
        std = np.maximum(std, eps)
        out[all_idx] = (features[all_idx] - mu) / std
    return out


# ---------------------------------------------------------------------------
# MAD robustize (fit on controls, transform all)
# ---------------------------------------------------------------------------

def _mad_robustize_plate(
    features: np.ndarray,
    all_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    eps: float = 1e-18,
) -> np.ndarray:
    """MAD-robustize one plate: (x - median_ctrl) / (1.4826 * MAD_ctrl + eps).

    Matches pycytominer convention: scale=1/1.4826 → multiply MAD by 1.4826.
    """
    ctrl_data = features[ctrl_idx]
    median_ctrl = np.median(ctrl_data, axis=0)
    mad_ctrl = np.median(np.abs(ctrl_data - median_ctrl), axis=0) * np.float32(1.4826)
    mad_ctrl = np.maximum(mad_ctrl, eps)
    plate_data = features[all_idx]
    return (plate_data - median_ctrl) / mad_ctrl


# ---------------------------------------------------------------------------
# Spherize (ZCA-cor, fit on controls, transform all)
# ---------------------------------------------------------------------------

def _spherize_plate_zca_cor(
    features: np.ndarray,
    all_idx: np.ndarray,
    ctrl_idx: np.ndarray,
    epsilon: float = 1e-3,
) -> Optional[np.ndarray]:
    """ZCA-cor spherize one plate, matching pycytominer's implementation.

    ZCA-cor = standardize → SVD → whiten → rotate back.
    Fit on controls, transform all wells.
    Returns None if computation fails.
    """
    ctrl_data = features[ctrl_idx].astype(np.float64)
    n, d = ctrl_data.shape

    # Standardize controls (ZCA-cor operates on correlation structure)
    mu = ctrl_data.mean(axis=0)
    std = ctrl_data.std(axis=0)
    zero_var = std == 0
    if zero_var.any():
        return None
    ctrl_std = (ctrl_data - mu) / std

    # SVD of standardized control data
    try:
        _, Sigma, Vt = np.linalg.svd(ctrl_std, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    # Pad Sigma if n <= d
    if n <= d:
        r = min(n, len(Sigma))
        Sigma = np.concatenate([Sigma[:r], np.full(d - r, Sigma[r - 1] if r > 0 else epsilon)])

    Sigma = Sigma + epsilon

    # W_pca = (Vt / Sigma[:, None])^T * sqrt(n-1)
    W = (Vt / Sigma[:, None]).T * np.sqrt(n - 1)
    # ZCA rotation: W_zca = W @ Vt
    W = W @ Vt

    # Transform ALL wells: standardize then apply W
    plate_data = features[all_idx].astype(np.float64)
    plate_std = (plate_data - mu) / std
    result = plate_std @ W

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Combined MAD + Spherize plate-wise
# ---------------------------------------------------------------------------

def mad_and_spherize_platewise(
    features: np.ndarray,
    mad_indices: list,
    spherize_indices: Optional[list] = None,
    spherize_epsilon: float = 1e-3,
    min_controls_for_mad: int = 5,
    min_controls_for_spherize: int = 20,
    do_spherize: bool = True,
) -> np.ndarray:
    """MAD-robustize by one grouping, optionally ZCA-cor spherize by another.

    Parameters
    ----------
    features : (N, D) float32 array (post CenterScale)
    mad_indices : list of (group_id, all_idx, ctrl_idx) for MAD robustization
    spherize_indices : list of (group_id, all_idx, ctrl_idx) for sphering. If
        omitted, uses ``mad_indices`` for backwards-compatible plate-wise
        behavior.
    """
    n_samples, n_feat = features.shape
    out = np.zeros_like(features)
    spherize_indices = mad_indices if spherize_indices is None else spherize_indices

    n_mad_only = 0
    n_spherized = 0
    n_skipped = 0

    for group_id, all_idx, ctrl_idx in mad_indices:
        n_ctrl = len(ctrl_idx)

        if n_ctrl < min_controls_for_mad:
            logger.warning("MAD group %s: %d controls — skipping normalization", group_id, n_ctrl)
            out[all_idx] = features[all_idx]
            n_skipped += 1
            continue

        mad_result = _mad_robustize_plate(features, all_idx, ctrl_idx)
        out[all_idx] = mad_result

    if not do_spherize:
        logger.info("MAD groups: %d, skipped: %d; spherize disabled", len(mad_indices), n_skipped)
        return out

    spherize_input = out.copy()
    for group_id, all_idx, ctrl_idx in spherize_indices:
        n_ctrl = len(ctrl_idx)
        if n_ctrl < min_controls_for_spherize or n_ctrl < n_feat:
            n_mad_only += 1
            continue

        sph_result = _spherize_plate_zca_cor(spherize_input, all_idx, ctrl_idx, spherize_epsilon)
        if sph_result is not None:
            out[all_idx] = sph_result
            n_spherized += 1
        else:
            # Keep MAD-only result (already in out)
            n_mad_only += 1

    logger.info("Spherize groups: %d spherized, %d MAD-only; MAD groups skipped: %d",
                n_spherized, n_mad_only, n_skipped)
    return out


# ---------------------------------------------------------------------------
# Zero-variance feature removal
# ---------------------------------------------------------------------------

def find_zero_variance_cols(features: np.ndarray, ctrl_idx: np.ndarray) -> np.ndarray:
    """Return boolean mask of features with >1 unique value in controls."""
    ctrl_data = features[ctrl_idx]
    # A feature has zero variance if min == max across all controls
    valid = (ctrl_data.max(axis=0) - ctrl_data.min(axis=0)) > 0
    return valid


# ---------------------------------------------------------------------------
# Full pipeline for one model
# ---------------------------------------------------------------------------

def normalize_single_model(
    features: np.ndarray,
    plate_labels: np.ndarray,
    control_mask: np.ndarray,
    n_pca_components: int = 128,
    spherize_epsilon: float = 1e-3,
    min_controls_for_mad: int = 5,
    min_controls_for_spherize: int = 20,
    centerscale_position: str = "pre_pca",
    centerscale_fit_on: str = "all",
    centerscale_enabled: bool = True,
    centerscale_labels: Optional[np.ndarray] = None,
    mad_labels: Optional[np.ndarray] = None,
    spherize_labels: Optional[np.ndarray] = None,
    do_spherize: bool = True,
) -> np.ndarray:
    """Full normalization pipeline with configurable CenterScale placement.

    Parameters
    ----------
    features : (N, D) float32 array of raw features
    plate_labels : (N,) plate identifiers
    control_mask : (N,) boolean, True for negative control wells
    n_pca_components : PCA dimensionality reduction target. Set to 0 to skip PCA.
    spherize_epsilon : regularization for spherize
    centerscale_position : "pre_pca", "post_pca", or "none"
        "pre_pca"  — CenterScale → PCA → MAD → Spherize
        "post_pca" — PCA → CenterScale → MAD → Spherize
        "none"     — PCA → MAD → Spherize
        When n_pca_components=0: CenterScale → MAD → Spherize (position ignored)
    centerscale_fit_on : "all" or "controls"
        "all"      — fit mean/std on all wells per plate
        "controls" — fit mean/std on negative controls only

    Returns
    -------
    (N, D') float32 normalized features (D'=n_pca_components if PCA, else D)
    """
    import time
    n_samples, n_feat = features.shape
    logger.info("Input: %d samples × %d features", n_samples, n_feat)
    if not centerscale_enabled:
        centerscale_position = "none"
    logger.info(
        "CenterScale: enabled=%s, position=%s, fit_on=%s",
        centerscale_enabled,
        centerscale_position,
        centerscale_fit_on,
    )

    # Handle NaN/Inf
    bad = np.isnan(features) | np.isinf(features)
    n_bad = bad.sum()
    if n_bad > 0:
        logger.warning("Replacing %d NaN/Inf values with 0", n_bad)
        features = features.copy()
        features[bad] = 0.0

    # Precompute plate indices once
    centerscale_indices = _build_plate_indices(
        plate_labels if centerscale_labels is None else centerscale_labels,
        control_mask,
    )
    mad_indices = _build_plate_indices(
        plate_labels if mad_labels is None else mad_labels,
        control_mask,
    )
    spherize_indices = _build_plate_indices(
        plate_labels if spherize_labels is None else spherize_labels,
        control_mask,
    )
    all_ctrl_idx = np.where(control_mask)[0]
    logger.info(
        "%d CenterScale groups, %d MAD groups, %d Spherize groups, %d total controls",
        len(centerscale_indices),
        len(mad_indices),
        len(spherize_indices),
        len(all_ctrl_idx),
    )

    # --- No PCA: CenterScale → MAD → Spherize ---
    if n_pca_components == 0:
        if centerscale_position != "none":
            t0 = time.time()
            features = centerscale_by_plate(features, centerscale_indices, fit_on=centerscale_fit_on)
            logger.info("CenterScale (no PCA, fit_on=%s) in %.1fs", centerscale_fit_on, time.time() - t0)

    # --- pre_pca: CenterScale first, then PCA ---
    elif centerscale_position == "pre_pca":
        t0 = time.time()
        features = centerscale_by_plate(features, centerscale_indices, fit_on=centerscale_fit_on)
        logger.info("CenterScale (pre-PCA, fit_on=%s) in %.1fs", centerscale_fit_on, time.time() - t0)

        t0 = time.time()
        features = embed_by_pca(features, n_pca_components)
        logger.info("PCA(%d) → %s in %.1fs", n_pca_components, features.shape, time.time() - t0)

    # --- none: PCA first, no CenterScale ---
    elif centerscale_position == "none":
        t0 = time.time()
        features = embed_by_pca(features, n_pca_components)
        logger.info("PCA(%d, no CenterScale) → %s in %.1fs", n_pca_components, features.shape, time.time() - t0)

    # --- post_pca: PCA first, then CenterScale ---
    else:
        t0 = time.time()
        features = embed_by_pca(features, n_pca_components)
        logger.info("PCA(%d) → %s in %.1fs", n_pca_components, features.shape, time.time() - t0)

        # Remove zero-variance features in controls (post-PCA)
        valid_mask = find_zero_variance_cols(features, all_ctrl_idx)
        n_removed = (~valid_mask).sum()
        if n_removed > 0:
            logger.info("Removing %d zero-variance features post-PCA", n_removed)
            features = features[:, valid_mask]

        t0 = time.time()
        features = centerscale_by_plate(features, centerscale_indices, fit_on=centerscale_fit_on)
        logger.info("CenterScale (post-PCA, fit_on=%s) in %.1fs", centerscale_fit_on, time.time() - t0)

    # Remove zero-variance features post-CenterScale
    valid_mask = find_zero_variance_cols(features, all_ctrl_idx)
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        logger.info("Removing %d zero-variance features", n_removed)
        features = features[:, valid_mask]

    # MAD + Spherize (always fit on controls)
    t0 = time.time()
    features = mad_and_spherize_platewise(
        features,
        mad_indices,
        spherize_indices=spherize_indices,
        spherize_epsilon=spherize_epsilon,
        min_controls_for_mad=min_controls_for_mad,
        min_controls_for_spherize=min_controls_for_spherize,
        do_spherize=do_spherize,
    )
    logger.info("MAD+Spherize in %.1fs", time.time() - t0)

    return features
