"""NumPy reference implementations of the composite objective and the metrics.

These are deliberately independent of PyTorch: the numerical verification
(`verify.py`) runs on them, and `tests/test_losses_torch.py` checks that the
PyTorch losses used in training agree with them to 1e-5.

Shapes: probabilities `p` are (C, V) for one sample, C = K + 1 channels with
channel 0 = background, V = number of voxels. Labels `y` are (V,) integers.
`supervised` is a boolean vector of length C; background is always supervised.
"""
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

EPS_DICE = 1e-5
EPS_LOG = 1e-8


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def merge_unsupervised(p: np.ndarray, supervised: np.ndarray) -> np.ndarray:
    """Eqs. 4.8-4.9: fold every unsupervised channel's mass into background."""
    sup = np.asarray(supervised, dtype=bool).copy()
    sup[0] = True
    merged = p * sup[:, None]
    merged[0] = merged[0] + p[~sup].sum(axis=0)
    return merged


def one_hot(y: np.ndarray, c: int) -> np.ndarray:
    g = np.zeros((c, y.size), dtype=np.float64)
    g[y.ravel(), np.arange(y.size)] = 1.0
    return g


def soft_dice_terms(p: np.ndarray, g: np.ndarray, channels: Iterable[int]) -> np.ndarray:
    ch = list(channels)
    inter = (p[ch] * g[ch]).sum(axis=1)
    return (2 * inter + EPS_DICE) / (p[ch].sum(axis=1) + g[ch].sum(axis=1) + EPS_DICE)


def marginal_loss(p: np.ndarray, y: np.ndarray, supervised: np.ndarray) -> float:
    """Eq. 4.10: Dice over S u {0} plus cross-entropy, both on the merged distribution."""
    sup = np.asarray(supervised, dtype=bool).copy()
    sup[0] = True
    y = np.where(sup[y], y, 0)  # a label in an unsupervised channel cannot exist; treat as background
    pm = merge_unsupervised(p, sup)
    g = one_hot(y, p.shape[0])
    channels = np.flatnonzero(sup)
    dice = soft_dice_terms(pm, g, channels).mean()
    ce = -np.log(np.clip(pm[y, np.arange(y.size)], EPS_LOG, None)).mean()
    return float(1.0 - dice + ce)


def naive_loss(p: np.ndarray, y: np.ndarray) -> float:
    """The same Dice + CE evaluated on the unmerged distribution over all channels."""
    g = one_hot(y, p.shape[0])
    dice = soft_dice_terms(p, g, range(p.shape[0])).mean()
    ce = -np.log(np.clip(p[y, np.arange(y.size)], EPS_LOG, None)).mean()
    return float(1.0 - dice + ce)


def exclusion_loss(p: np.ndarray) -> float:
    """Eq. 4.11: penalise foreground mass above 1 at any voxel."""
    return float(np.maximum(0.0, p[1:].sum(axis=0) - 1.0).mean())


def cox_loss(log_risk: Sequence[float], time: Sequence[float], event: Sequence[int]) -> float:
    """Eq. 4.12: negative Cox partial log-likelihood, Breslow ties, within-batch risk set."""
    eta = np.asarray(log_risk, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    d = np.asarray(event, dtype=bool)
    if not d.any():
        return 0.0
    at_risk = t[None, :] >= t[:, None]  # row i: j with t_j >= t_i
    masked = np.where(at_risk, eta[None, :], -np.inf)
    m = masked.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(masked - m).sum(axis=1)))
    return float(-(eta[d] - lse[d]).mean())


def horizon_labels(time, event, horizons) -> tuple[np.ndarray, np.ndarray]:
    """Label 1 = alive at horizon. Undetermined pairs (censored before tau) are masked out."""
    t = np.asarray(time, dtype=np.float64)[:, None]
    d = np.asarray(event, dtype=bool)[:, None]
    tau = np.asarray(horizons, dtype=np.float64)[None, :]
    alive = t > tau
    died_before = d & (t <= tau)
    determined = alive | died_before
    return alive.astype(np.float64), determined


def horizon_loss(logits, time, event, horizons) -> float:
    """Eq. 4.13: BCE over determined (case, horizon) pairs only."""
    z = np.asarray(logits, dtype=np.float64)
    y, mask = horizon_labels(time, event, horizons)
    if not mask.any():
        return 0.0
    bce = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(bce[mask].mean())


def uncertainty_weighted(losses: Sequence[float], log_vars: Sequence[float]) -> float:
    """Eq. 4.14: sum_t exp(-s_t) L_t + s_t."""
    L = np.asarray(losses, dtype=np.float64)
    s = np.asarray(log_vars, dtype=np.float64)
    return float((np.exp(-s) * L + s).sum())


# ----------------------------------------------------------------------------- metrics
def dice_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    """Eq. 4.15 on binary masks. Two empty masks agree perfectly (returns 1.0)."""
    a = np.asarray(pred, dtype=bool)
    b = np.asarray(truth, dtype=bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / s)


def hd95(pred: np.ndarray, truth: np.ndarray, spacing: Sequence[float] | None = None) -> float:
    """95th-percentile symmetric Hausdorff distance between mask surfaces (in spacing units).

    Returns nan when either mask is empty, because the distance is undefined there.
    """
    from scipy import ndimage

    a = np.asarray(pred, dtype=bool)
    b = np.asarray(truth, dtype=bool)
    if not a.any() or not b.any():
        return float("nan")
    spacing = tuple(spacing) if spacing is not None else (1.0,) * a.ndim
    surf_a = a & ~ndimage.binary_erosion(a)
    surf_b = b & ~ndimage.binary_erosion(b)
    dt_b = ndimage.distance_transform_edt(~surf_b, sampling=spacing)
    dt_a = ndimage.distance_transform_edt(~surf_a, sampling=spacing)
    d = np.concatenate([dt_b[surf_a], dt_a[surf_b]])
    return float(np.percentile(d, 95))


def concordance_index(log_risk, time, event) -> float:
    """Harrell's C (Eq. 4.16). Ties in risk count one half. 0.5 = chance."""
    eta = np.asarray(log_risk, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    d = np.asarray(event, dtype=bool)
    num = 0.0
    den = 0.0
    for i in range(len(t)):
        if not d[i]:
            continue
        later = t > t[i]
        den += later.sum()
        num += (eta[i] > eta[later]).sum() + 0.5 * (eta[i] == eta[later]).sum()
    return float(num / den) if den else float("nan")


def time_dependent_auc(log_risk, time, event, horizon: float) -> float:
    """Cumulative/dynamic AUC at `horizon`: cases died by tau, controls still alive after tau."""
    eta = np.asarray(log_risk, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    d = np.asarray(event, dtype=bool)
    cases = eta[d & (t <= horizon)]
    controls = eta[t > horizon]
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    diff = cases[:, None] - controls[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def summarize(values: Dict[str, float]) -> Dict[str, float]:
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in values.items()}
