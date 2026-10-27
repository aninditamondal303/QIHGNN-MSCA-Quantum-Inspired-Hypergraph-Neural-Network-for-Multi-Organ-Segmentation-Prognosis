"""Modality-aware intensity normalisation, resampling and label remapping (report, 4.3)."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from ..config import LABELS

# CT windows (centre, width) in Hounsfield units. A single-phase CT volume is expanded into four
# windowed channels so that CT and four-sequence MRI share the 4-channel input of the network.
CT_WINDOWS: Dict[str, Tuple[float, float]] = {
    "abdomen": (40.0, 400.0),
    "liver": (60.0, 160.0),
    "bone": (400.0, 1800.0),
    "wide": (0.0, 2000.0),
}
MRI_SEQUENCES: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


def ct_window(hu: np.ndarray, centre: float, width: float) -> np.ndarray:
    lo, hi = centre - width / 2, centre + width / 2
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def ct_to_channels(hu: np.ndarray, windows: Mapping[str, Tuple[float, float]] = CT_WINDOWS) -> np.ndarray:
    """(H, W, D) HU -> (4, H, W, D) in [0, 1]."""
    return np.stack([ct_window(hu, c, w) for c, w in windows.values()])


def mri_zscore(vol: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """z-score over the non-zero (head) region, then clip outliers. Background stays 0."""
    vol = vol.astype(np.float32)
    mask = vol != 0
    if mask.sum() < 2:
        return np.zeros_like(vol)
    mu, sd = vol[mask].mean(), vol[mask].std()
    out = np.zeros_like(vol)
    out[mask] = np.clip((vol[mask] - mu) / (sd + 1e-8), -clip, clip)
    return out


def rescale_display_mask(mask: np.ndarray) -> np.ndarray:
    """Masks exported as display images store {0, 255}; map them back to {0, 1}.

    Without this, every foreground voxel would get class index 255, far outside the label space.
    """
    m = np.asarray(mask)
    values = np.unique(m)
    if values.size <= 2 and values.max(initial=0) == 255:
        return (m > 127).astype(np.uint8)
    return m.astype(np.uint8)


def remap_labels(mask: np.ndarray, mapping: Mapping[int, str]) -> np.ndarray:
    """Map a corpus's own label values to unified channel indices. Unlisted values become background."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    for value, name in mapping.items():
        out[mask == value] = LABELS.index(name)
    return out


def supervision_vector(channels: Iterable[str]) -> np.ndarray:
    v = np.zeros(len(LABELS), dtype=bool)
    v[0] = True
    for name in channels:
        v[LABELS.index(name)] = True
    return v


def resample(vol: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    """Resample the last three axes to `shape`. order=1 for images, order=0 for labels."""
    from scipy import ndimage

    spatial = vol.shape[-3:]
    if tuple(spatial) == tuple(shape):
        return vol
    factors = [t / s for t, s in zip(shape, spatial)]
    if vol.ndim == 4:
        return np.stack([ndimage.zoom(c, factors, order=order) for c in vol])
    return ndimage.zoom(vol, factors, order=order)


def crop_or_pad(vol: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Centre-crop or zero-pad the last three axes to exactly `shape` (fixes zoom rounding)."""
    out = vol
    for axis_offset, target in enumerate(shape):
        axis = vol.ndim - 3 + axis_offset
        size = out.shape[axis]
        if size > target:
            start = (size - target) // 2
            out = np.take(out, np.arange(start, start + target), axis=axis)
        elif size < target:
            pad = [(0, 0)] * out.ndim
            before = (target - size) // 2
            pad[axis] = (before, target - size - before)
            out = np.pad(out, pad)
    return out


def to_grid(vol: np.ndarray, shape: Sequence[int], order: int) -> np.ndarray:
    return crop_or_pad(resample(vol, shape, order), shape)
