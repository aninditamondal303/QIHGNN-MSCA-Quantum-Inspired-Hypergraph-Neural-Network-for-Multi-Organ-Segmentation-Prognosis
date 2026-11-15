"""Shared pieces for the screening models: image I/O, near-duplicate grouping, leakage-free
splits and the metrics reported on the website. NumPy only, so it is unit-tested without torch."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CLASSES: Tuple[str, ...] = ("no_tumour", "glioma", "meningioma", "pituitary", "not_brain_mri")
TUMOUR = (1, 2, 3)
NOT_BRAIN = 4
CLS_SIZE = 224   # classifier input (letterboxed)
SEG_SIZE = 192   # segmenter input (letterboxed)


# ----------------------------------------------------------------------------- images
def load_gray(path: Path) -> np.ndarray:
    """Any image file -> uint8 (H, W). Colour images are converted with ITU-R 601 luma, like the browser."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def to_uint8(slice2d: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Scale an MRI slice to 0..255 using percentiles of its non-zero pixels (what a viewer export does)."""
    s = np.asarray(slice2d, dtype=np.float32)
    nz = s[s > 0]
    if nz.size < 16:
        return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(nz, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((s - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8) * (s > 0)


def letterbox(img: np.ndarray, size: int, order: int = 1) -> Tuple[np.ndarray, Dict[str, float]]:
    """Resize so the longer side is `size`, pad the rest with zeros, centred. Mirrors web/src/lib/screen.ts."""
    from PIL import Image

    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resample = Image.BILINEAR if order else Image.NEAREST
    small = np.asarray(Image.fromarray(img).resize((nw, nh), resample))
    out = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = small
    return out, {"scale": scale, "top": top, "left": left, "h": nh, "w": nw}


def seg_normalise(img_u8: np.ndarray) -> np.ndarray:
    """Segmenter input: percentile scaling over non-zero pixels to [0, 1]. Mirrors the browser."""
    s = img_u8.astype(np.float32)
    nz = s[s > 0]
    if nz.size < 16:
        return np.zeros_like(s)
    lo, hi = np.percentile(nz, [1.0, 99.0])
    return np.clip((s - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- duplicates and splits
def dhash(img_u8: np.ndarray, size: int = 8) -> int:
    """Difference hash: robust to re-encoding and resizing, so copies of one scan collide."""
    from PIL import Image

    small = np.asarray(Image.fromarray(img_u8).resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash_canonical(img_u8: np.ndarray) -> int:
    """Smallest dHash over the 8 flips and 90-degree rotations, so mirrored or rotated copies collide too."""
    variants = []
    for k in range(4):
        r = np.rot90(img_u8, k)
        variants += [r, r[:, ::-1]]
    return min(dhash(np.ascontiguousarray(v)) for v in variants)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(hashes: Sequence[int], max_bits: int = 2) -> List[int]:
    """Group indices whose 64-bit dHashes differ in at most `max_bits` bits. Returns a group id per item.

    Near matches are found exactly with the pigeonhole trick: two hashes within `max_bits` bits agree on at
    least one of (max_bits + 1) disjoint bit bands, so only items sharing a band value are compared.
    """
    n = len(hashes)
    uf = UnionFind(n)
    bands = max_bits + 1
    width = math.ceil(64 / bands)
    for b in range(bands):
        shift, mask = b * width, (1 << width) - 1
        buckets: Dict[int, List[int]] = {}
        for i, h in enumerate(hashes):
            buckets.setdefault((h >> shift) & mask, []).append(i)
        for members in buckets.values():
            if len(members) < 2 or len(members) > 2000:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    i, j = members[x], members[y]
                    if bin(hashes[i] ^ hashes[j]).count("1") <= max_bits:
                        uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def group_split(key: str, fractions=(0.70, 0.15, 0.15), salt: str = "qihgnn-screen") -> str:
    """Deterministic split of a group (patient or duplicate cluster), so copies never straddle splits."""
    u = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    if u < fractions[0]:
        return "train"
    return "val" if u < fractions[0] + fractions[1] else "test"


# ----------------------------------------------------------------------------- metrics
def confusion(y: Sequence[int], p: Sequence[int], k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC by the rank-sum (Mann-Whitney) formula, ties counted half."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binary_report(tumour_prob: Sequence[float], is_tumour: Sequence[int], threshold: float = 0.5) -> Dict:
    s, y = np.asarray(tumour_prob, float), np.asarray(is_tumour, int)
    pred = (s >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    return {
        "n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp / max(1, tp + fn), "sensitivity_ci": wilson(tp, tp + fn),
        "specificity": tn / max(1, tn + fp), "specificity_ci": wilson(tn, tn + fp),
        "accuracy": (tp + tn) / max(1, len(y)), "accuracy_ci": wilson(tp + tn, len(y)),
        "auc": binary_auc(s, y),
    }


def ece(probs: np.ndarray, labels: Sequence[int], bins: int = 15) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == np.asarray(labels)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    p, t = pred.astype(bool), truth.astype(bool)
    s = p.sum() + t.sum()
    return 1.0 if s == 0 else float(2 * (p & t).sum() / s)


def write_json(path: Path, obj) -> None:
    from ..verify import to_json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(obj, indent=1))


def summarise(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x == x], float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "median": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
