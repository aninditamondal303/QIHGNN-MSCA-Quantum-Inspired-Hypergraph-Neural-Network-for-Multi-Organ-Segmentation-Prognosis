"""Builds the screening datasets from the raw Kaggle downloads.

Every split is decided per group (a patient, a CT volume, or a cluster of near-duplicate images), so
the same scan never sits on both sides of the train/test line. That is the reason these numbers are
lower than the ~99% often quoted for the brain-tumour MRI dataset, whose official Testing folder
shares images with its Training folder (`audit` below counts them).
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.adapters import H5_SLICE, _read_h5, brats_h5_patient_id, h5_id_offset
from ..data.listing import list_files
from ..data.survival import parse_days
from .common import SEG_SIZE, dhash_canonical, duplicate_groups, group_split, letterbox, load_gray, to_uint8

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SEQUENCES = ("T1", "T1ce", "T2", "FLAIR")   # channel order inside the BraTS .h5 slices


@dataclass
class Item:
    """One classifier image: a file on disk or an array already in memory."""
    source: str
    label: int
    group: str
    path: Optional[Path] = None
    array: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def image(self) -> np.ndarray:
        return self.array if self.array is not None else load_gray(self.path)


# ----------------------------------------------------------------------------- dataset discovery
def _cls_from_folder(name: str) -> Optional[int]:
    n = name.lower().replace("_", "").replace(" ", "").replace("-", "")
    if n.startswith("no") and ("tum" in n or n == "no"):
        return 0
    if "glioma" in n:
        return 1
    if "mening" in n:
        return 2
    if "pituitar" in n:
        return 3
    return None


def masoud_items(root: Path, limit: int = 0) -> List[Item]:
    """Brain Tumor MRI Dataset: Training/ and Testing/ folders with glioma, meningioma, notumor, pituitary."""
    out = []
    for p in list_files(root, IMG_EXT):
        c = _cls_from_folder(p.parent.name)
        if c is None:
            continue
        official = "test" if "test" in p.parent.parent.name.lower() else "train"
        out.append(Item("brain-mri-4class", c, "", path=p, meta={"official_split": official}))
    if limit:
        rng = np.random.default_rng(0)
        out = [out[i] for i in sorted(rng.choice(len(out), min(limit, len(out)), replace=False))]
    return out


def external_items(root: Path) -> List[Item]:
    """Brain MRI Images for Brain Tumor Detection: yes/ and no/ folders. Used only as an external test set."""
    out = []
    for p in list_files(root, IMG_EXT):
        n = p.parent.name.lower()
        if n in ("yes", "y"):
            out.append(Item("external-yes-no", 1, f"ext:{p.name}", path=p))
        elif n in ("no", "n"):
            out.append(Item("external-yes-no", 0, f"ext:{p.name}", path=p))
    return out


def figshare_pairs(root: Path, limit: int = 0) -> List[Tuple[Path, Path]]:
    """2D tumour segmentation set (T1 contrast-enhanced, skull present): images/ and masks/ with matching names."""
    files = list_files(root, IMG_EXT)
    imgs, masks = {}, {}
    for p in files:
        rel = p.relative_to(root).as_posix().lower()
        key = p.stem.lower().replace("_mask", "").replace("-mask", "")
        (masks if re.search(r"mask|label|seg", rel) else imgs)[key] = p
    pairs = [(imgs[k], masks[k]) for k in sorted(imgs) if k in masks]
    return pairs[:limit] if limit else pairs


SLICE = re.compile(r"(\d+)\D+(\d+)\.(png|jpg|jpeg)$", re.I)


def ct_items(root: Path, per_volume: int = 10, max_volumes: int = 0) -> List[Item]:
    """Abdominal CT slices (LiTS mirror) as 'not a brain MRI' examples, grouped by CT volume."""
    vols: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for p in list_files(root, IMG_EXT):
        if re.search(r"mask|seg|label", p.relative_to(root).as_posix(), re.I):
            continue
        m = SLICE.search(p.name)
        if m:
            vols[m.group(1)].append((int(m.group(2)), p))
    keys = sorted(vols, key=int)[: max_volumes or None]
    out = []
    for k in keys:
        sl = sorted(vols[k])
        for j in np.linspace(0, len(sl) - 1, min(per_volume, len(sl))).round().astype(int):
            out.append(Item("abdominal-ct", 4, f"ct:{k}", path=sl[j][1]))
    return out


def photo_items(root: Path, limit: int = 0) -> List[Item]:
    """Colposcopy photographs (cervical dataset) as 'not a brain MRI' examples."""
    ps = [p for p in list_files(root, (".png", ".jpg", ".jpeg")) if not re.search(r"mask|label", p.as_posix(), re.I)]
    ps = ps[:limit] if limit else ps
    return [Item("colposcopy-photo", 4, f"photo:{p.stem}", path=p) for p in ps]


def synthetic_items(n: int, seed: int = 7) -> List[Item]:
    """Non-medical images (noise, gradients, shapes, text, blank) so the model can say 'not a brain MRI'."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = i % 6
        s = int(rng.integers(160, 400))
        if kind == 0:
            a = rng.integers(0, 256, (s, s)).astype(np.uint8)
        elif kind == 1:
            a = np.tile(np.linspace(rng.integers(0, 120), rng.integers(130, 256), s), (s, 1)).astype(np.uint8)
            a = np.rot90(a, int(rng.integers(0, 4))).copy()
        elif kind == 2:
            a = np.full((s, s), int(rng.integers(0, 256)), np.uint8)
        else:
            im = Image.new("L", (s, s), int(rng.integers(0, 256)))
            d = ImageDraw.Draw(im)
            for _ in range(int(rng.integers(3, 15))):
                x0, y0 = rng.integers(0, s, 2)
                x1, y1 = x0 + rng.integers(10, s // 2), y0 + rng.integers(10, s // 2)
                fill = int(rng.integers(0, 256))
                if kind == 3:
                    d.ellipse([int(x0), int(y0), int(x1), int(y1)], fill=fill)
                elif kind == 4:
                    d.rectangle([int(x0), int(y0), int(x1), int(y1)], fill=fill)
                else:
                    d.text((int(x0), int(y0)), "".join(chr(int(c)) for c in rng.integers(65, 91, 8)), fill=fill)
            a = np.asarray(im)
        out.append(Item("synthetic", 4, f"syn:{i}", array=a))
    return out


# ----------------------------------------------------------------------------- BraTS: one pass, everything
@dataclass
class BratsVolume:
    patient_id: str
    split: str
    wt_cc: float
    tc_cc: float
    et_cc: float
    seg_x: np.ndarray            # (n, 4, S, S) uint8, the four sequences of the sampled slices
    seg_y: np.ndarray            # (n, S, S) uint8 whole-tumour mask
    seg_tumour: np.ndarray       # (n,) bool, slice contains tumour
    cls: List[Tuple[np.ndarray, int, str]]   # (uint8 image, label 0/1, sequence)
    full: Optional[Dict] = None  # test patients: whole FLAIR and T2 volumes + mask for 3D evaluation


def brats_volumes(root: Path, limit: int = 0, workers: int = 8, seed: int = 2026) -> Tuple[List[BratsVolume], Dict]:
    files = list_files(root, (".h5",))
    vols: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
    for p in files:
        m = H5_SLICE.search(p.name)
        if m:
            vols[int(m.group(1))].append((int(m.group(2)), p))
    off = h5_id_offset(vols)
    keys = sorted(vols)[: limit or None]
    info = {"volumes_found": len(vols), "first_volume_number": min(vols) if vols else None, "id_offset": off}

    def one(v: int) -> BratsVolume:
        rng = np.random.default_rng(seed + v)
        items = sorted(vols[v])
        with ThreadPoolExecutor(16) as pool:
            pairs = list(pool.map(_read_h5, [p for _, p in items]))
        img = np.stack([a for a, _ in pairs], 0)          # (D, 240, 240, 4) float
        msk = np.stack([b for _, b in pairs], 0) > 0       # (D, 240, 240, 3) NCR/NET, ED, ET
        wt, tc, et = msk.any(-1), msk[..., 0] | msk[..., 2], msk[..., 2]
        pid = brats_h5_patient_id(v, off)
        split = group_split(f"brats:{pid}")
        brain = (np.abs(img).sum(-1) > 0)
        brain_frac = brain.reshape(len(img), -1).mean(1)
        wt_area = wt.reshape(len(img), -1).sum(1)
        tumour_idx = np.flatnonzero(wt_area > 0)
        # Segmenter slices: every second tumour slice plus tumour-free brain slices (30% as many).
        seg_pos = tumour_idx[::2]
        free = np.flatnonzero((wt_area == 0) & (brain_frac > 0.05))
        seg_neg = rng.choice(free, min(len(free), max(2, int(0.3 * len(seg_pos)))), replace=False) if len(free) else np.array([], int)
        pick = np.sort(np.concatenate([seg_pos, seg_neg])).astype(int)
        seg_x = np.zeros((len(pick), 4, SEG_SIZE, SEG_SIZE), np.uint8)
        seg_y = np.zeros((len(pick), SEG_SIZE, SEG_SIZE), np.uint8)
        for i, d in enumerate(pick):
            for c in range(4):
                seg_x[i, c] = letterbox(to_uint8(img[d, :, :, c]), SEG_SIZE)[0]
            seg_y[i] = letterbox(wt[d].astype(np.uint8), SEG_SIZE, order=0)[0]
        # Classifier slices: clear tumour (>=1% of the brain) labelled glioma; slices >=8 away from any tumour labelled no tumour.
        cls = []
        big = [d for d in tumour_idx if wt_area[d] >= 0.01 * max(1, brain[d].sum()) and brain_frac[d] > 0.1]
        for d in rng.choice(big, min(3, len(big)), replace=False) if big else []:
            c = int(rng.integers(0, 4))
            cls.append((to_uint8(img[d, :, :, c]), 1, SEQUENCES[c]))
        if len(tumour_idx):
            dist = np.min(np.abs(np.arange(len(img))[:, None] - tumour_idx[None, :]), 1)
            far = np.flatnonzero((dist >= 8) & (brain_frac > 0.15))
            for d in rng.choice(far, min(3, len(far)), replace=False) if len(far) else []:
                c = int(rng.integers(0, 4))
                cls.append((to_uint8(img[d, :, :, c]), 0, SEQUENCES[c]))
        full = None
        if split == "test":
            full = {s: np.stack([letterbox(to_uint8(img[d, :, :, SEQUENCES.index(s)]), SEG_SIZE)[0] for d in range(len(img))])
                    for s in ("FLAIR", "T2")}
            full["mask"] = np.stack([letterbox(wt[d].astype(np.uint8), SEG_SIZE, order=0)[0] for d in range(len(img))])
            full["scale_mm"] = 240 / SEG_SIZE  # in-plane mm per pixel after letterboxing (BraTS is 1 mm isotropic)
        return BratsVolume(pid, split, wt.sum() / 1000.0, tc.sum() / 1000.0, et.sum() / 1000.0,
                           seg_x, seg_y, np.isin(pick, tumour_idx), cls, full)

    out: List[BratsVolume] = []
    with ThreadPoolExecutor(workers) as pool:
        for i, bv in enumerate(pool.map(one, keys)):
            out.append(bv)
            if (i + 1) % 25 == 0 or i + 1 == len(keys):
                print(f"  brats {i + 1}/{len(keys)} volumes", flush=True)
    return out, info


def brats_tables(root: Path) -> Dict[str, Dict]:
    """survival_info.csv -> {patient_id: {age, resection, time, event}}; also echoes name_mapping for the record."""
    surv: Dict[str, Dict] = {}
    for p in list_files(root, (".csv",)):
        if "surviv" not in p.name.lower():
            continue
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f)
            keys = {k.lower(): k for k in r.fieldnames or []}
            idk = next((keys[k] for k in keys if "id" in k), None)
            dk = next((keys[k] for k in keys if "surviv" in k), None)
            ak = next((keys[k] for k in keys if k.startswith("age")), None)
            rk = next((keys[k] for k in keys if "resection" in k), None)
            for row in r:
                t, e = parse_days(row.get(dk, ""))
                try:
                    age = float(row[ak])
                except (TypeError, ValueError, KeyError):
                    age = None
                if t is None or age is None or not idk:
                    continue
                surv[row[idk].strip()] = {"age": age, "resection": (row.get(rk) or "").strip() if rk else "", "time": t, "event": e}
        break
    return surv


def print_head(root: Path, names=("name_mapping.csv", "meta_data.csv"), n: int = 3) -> None:
    for p in list_files(root, (".csv",)):
        if p.name in names:
            with p.open(encoding="utf-8", errors="replace") as f:
                print(f"--- {p.name}:", *[next(f, "").rstrip() for _ in range(n)], sep="\n  ")


# ----------------------------------------------------------------------------- grouping and audit
def assign_groups(items: List[Item], hashes: Dict[int, int]) -> Dict:
    """Items of the 4-class dataset get their group from near-duplicate clustering; others keep theirs."""
    idx = [i for i, it in enumerate(items) if it.source == "brain-mri-4class"]
    gid = duplicate_groups([hashes[i] for i in idx])
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i, g in zip(idx, gid):
        clusters[g].append(i)
    leaked = conflicting = 0
    for members in clusters.values():
        rep = min(members)
        for i in members:
            items[i].group = f"mri4:{hashes[rep]:016x}"
        splits = {items[i].meta.get("official_split") for i in members}
        if len(splits) > 1:
            leaked += sum(items[i].meta.get("official_split") == "test" for i in members)
        if len({items[i].label for i in members}) > 1:
            conflicting += 1
    official_test = sum(items[i].meta.get("official_split") == "test" for i in idx)
    return {"images": len(idx), "duplicate_clusters": sum(len(m) > 1 for m in clusters.values()),
            "images_in_duplicate_clusters": sum(len(m) for m in clusters.values() if len(m) > 1),
            "official_test_images": official_test, "official_test_images_with_a_copy_in_training": leaked,
            "clusters_with_conflicting_labels": conflicting}


def hash_items(items: List[Item], workers: int = 16) -> Dict[int, int]:
    def h(i):
        try:
            return i, dhash_canonical(items[i].image())
        except Exception:  # unreadable file: give it a unique hash, it will be dropped later
            return i, -1
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(h, range(len(items))))


def split_items(items: List[Item]) -> Dict[str, List[Item]]:
    out: Dict[str, List[Item]] = {"train": [], "val": [], "test": []}
    for it in items:
        out[group_split(it.group)].append(it)
    return out


def load_images(items: List[Item], workers: int = 16) -> List[np.ndarray]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda it: it.image(), items))


Loader = Callable[[], np.ndarray]
