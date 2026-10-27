"""Per-corpus readers that turn raw downloads into unified cases.

Each adapter yields `RawCase` objects: a (4, H, W, D) float32 image already
normalised for its modality, a (H, W, D) uint8 label in the unified label
space, the supervision vector, and survival data when the corpus has it.
Layouts differ between Kaggle mirrors, so every adapter supports the layouts
the inspection step can encounter and fails loudly on anything else.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from .preprocess import MRI_SEQUENCES, ct_to_channels, mri_zscore, remap_labels, rescale_display_mask, supervision_vector
from .survival import SurvivalRecord, read_survival_csv


@dataclass
class RawCase:
    patient_id: str
    source: str
    image: np.ndarray
    label: np.ndarray
    supervised: np.ndarray
    modality: str
    time_days: Optional[float] = None
    event: Optional[int] = None
    covariates: List[float] = field(default_factory=lambda: [0.0] * 5)


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def _find_survival(root: Path) -> Dict[str, SurvivalRecord]:
    for p in root.rglob("*.csv"):
        if "surviv" in p.name.lower():
            try:
                return read_survival_csv(p)
            except ValueError:
                continue
    return {}


# ----------------------------------------------------------------------------- BraTS
BRATS_ID = re.compile(r"(BraTS20_Training_\d+)", re.I)
H5_SLICE = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$", re.I)


def brats_cases(root: Path) -> Iterator[RawCase]:
    survival = _find_survival(root)
    seg_files = sorted(p for p in root.rglob("*") if p.name.lower().endswith(("_seg.nii", "_seg.nii.gz")))
    if seg_files:
        yield from _brats_nifti(seg_files, survival)
        return
    h5 = sorted(p for p in root.rglob("*.h5") if H5_SLICE.search(p.name))
    if h5:
        yield from _brats_h5(h5, survival)
        return
    raise FileNotFoundError(f"No BraTS NIfTI segmentations or volume_*_slice_*.h5 files under {root}")


def _attach_survival(case: RawCase, survival: Dict[str, SurvivalRecord]) -> RawCase:
    rec = survival.get(case.patient_id)
    if rec:
        case.time_days, case.event, case.covariates = rec.time_days, rec.event, rec.covariates
    return case


def _brats_nifti(seg_files: List[Path], survival) -> Iterator[RawCase]:
    for seg in seg_files:
        base = seg.name.lower().replace("_seg.nii.gz", "").replace("_seg.nii", "")
        folder = seg.parent
        chans = []
        for s in MRI_SEQUENCES:
            cand = [p for p in folder.iterdir() if p.name.lower().startswith(f"{base}_{s}.nii")]
            if not cand:
                raise FileNotFoundError(f"missing {s} for {base}")
            chans.append(mri_zscore(_load_nifti(cand[0])))
        lab = _load_nifti(seg).astype(np.int16)
        label = remap_labels(lab, {1: "brain", 2: "brain", 4: "brain"})
        m = BRATS_ID.search(seg.as_posix())
        pid = m.group(1) if m else base
        case = RawCase(pid, "brats2020", np.stack(chans), label, supervision_vector(["brain"]), "MRI")
        yield _attach_survival(case, survival)


def _brats_h5(files: List[Path], survival) -> Iterator[RawCase]:
    import h5py

    volumes: Dict[int, List[tuple[int, Path]]] = defaultdict(list)
    for p in files:
        v, s = H5_SLICE.search(p.name).groups()
        volumes[int(v)].append((int(s), p))
    for v, items in sorted(volumes.items()):
        items.sort()
        imgs, masks = [], []
        for _, p in items:
            with h5py.File(p, "r") as f:
                imgs.append(np.asarray(f["image"], dtype=np.float32))  # (240, 240, 4)
                masks.append(np.asarray(f["mask"]))  # (240, 240, 3) one-hot NCR/ED/ET
        image = np.stack(imgs, axis=-1).transpose(2, 0, 1, 3)  # (4, H, W, D)
        mask = np.stack(masks, axis=-1).any(axis=2)  # (H, W, D) whole tumour
        image = np.stack([mri_zscore(c) for c in image])
        label = np.where(mask, 1, 0).astype(np.uint8)
        pid = f"BraTS20_Training_{v + 1:03d}"  # h5 volumes are 0-indexed in this mirror
        case = RawCase(pid, "brats2020", image, label, supervision_vector(["brain"]), "MRI")
        yield _attach_survival(case, survival)


# ----------------------------------------------------------------------------- LiTS
LITS_VOL = re.compile(r"volume-(\d+)\.nii", re.I)
LITS_SEG = re.compile(r"segmentation-(\d+)\.nii", re.I)
SLICE_PAIR = re.compile(r"(\d+)\D+(\d+)\.(png|jpg|jpeg)$", re.I)


def lits_cases(root: Path) -> Iterator[RawCase]:
    vols = {LITS_VOL.search(p.name).group(1): p for p in root.rglob("*.nii*") if LITS_VOL.search(p.name)}
    segs = {LITS_SEG.search(p.name).group(1): p for p in root.rglob("*.nii*") if LITS_SEG.search(p.name)}
    if vols and segs:
        for k in sorted(set(vols) & set(segs), key=int):
            hu = _load_nifti(vols[k])
            lab = remap_labels(_load_nifti(segs[k]).astype(np.int16), {1: "liver", 2: "liver"})
            yield RawCase(f"LiTS_{int(k):03d}", "lits", ct_to_channels(hu), lab, supervision_vector(["liver"]), "CT")
        return
    yield from _slice_stacks(root, "lits", "liver")


def _slice_stacks(root: Path, source: str, channel: str) -> Iterator[RawCase]:
    """Rebuild volumes from 2D slice images named <patient><sep><slice>.png, with masks in mask/label folders.

    Slices exported as display images have lost Hounsfield units, so they are replicated
    across the four input channels instead of being re-windowed.
    """
    from PIL import Image

    def is_mask(p: Path) -> bool:
        return bool(re.search(r"(mask|seg|label)", p.relative_to(root).as_posix(), re.I))

    images, masks = defaultdict(dict), defaultdict(dict)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = SLICE_PAIR.search(p.name)
        if not m:
            continue
        pid, sl = m.group(1), int(m.group(2))
        (masks if is_mask(p) else images)[pid][sl] = p
    if not images:
        raise FileNotFoundError(f"no slice images found under {root}")
    for pid in sorted(images, key=lambda s: int(s)):
        common = sorted(set(images[pid]) & set(masks.get(pid, {})))
        if len(common) < 8:
            continue
        img = np.stack([np.asarray(Image.open(images[pid][s]).convert("L"), np.float32) / 255.0 for s in common], -1)
        msk = np.stack([rescale_display_mask(np.asarray(Image.open(masks[pid][s]).convert("L"))) for s in common], -1)
        label = remap_labels((msk > 0).astype(np.int16), {1: channel})
        yield RawCase(f"{source}_{pid}", source, np.stack([img] * 4), label, supervision_vector([channel]), "CT")


ADAPTERS = {"brats2020": brats_cases, "lits": lits_cases}


def cases_for(key: str, root: Path) -> Iterator[RawCase]:
    if key == "cervical":
        # Only reached if inspection admitted it (volumetric + masks); layouts vary, so slice stacks are tried.
        yield from _slice_stacks(root, "cervical", "cervix")
        return
    yield from ADAPTERS[key](root)
