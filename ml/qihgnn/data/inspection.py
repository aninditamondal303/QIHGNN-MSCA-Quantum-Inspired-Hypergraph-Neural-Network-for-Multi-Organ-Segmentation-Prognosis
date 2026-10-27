"""Inspect downloaded corpora before anything assumes a format.

Walks data/raw/<corpus>/, records file types, samples shapes / dtypes / label
values, finds survival tables, and decides which corpora are admitted and which
channels they may supervise. Writes data/inspection_report.json.

Run: `python -m qihgnn.data.inspection` (from ml/).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..verify import ROOT, to_json

MASK_HINT = re.compile(r"(mask|seg|label|annot|gt|ground)", re.I)
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VOLUME_EXT = {".nii", ".nii.gz", ".h5", ".hdf5", ".mha", ".nrrd"}


def ext_of(p: Path) -> str:
    name = p.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else p.suffix.lower()


def sample_file(p: Path) -> Dict[str, Any]:
    e = ext_of(p)
    info: Dict[str, Any] = {"path": str(p), "ext": e}
    try:
        if e in {".nii", ".nii.gz"}:
            import nibabel as nib

            img = nib.load(str(p))
            arr = np.asanyarray(img.dataobj)
            info.update(shape=list(arr.shape), dtype=str(arr.dtype), spacing=[float(v) for v in img.header.get_zooms()])
        elif e in {".h5", ".hdf5"}:
            import h5py

            with h5py.File(p, "r") as f:
                info["keys"] = {k: {"shape": list(f[k].shape), "dtype": str(f[k].dtype)} for k in f.keys()}
                arr = np.asarray(f[next(k for k in f.keys() if MASK_HINT.search(k))]) if any(MASK_HINT.search(k) for k in f.keys()) else None
        elif e in IMAGE_EXT:
            from PIL import Image

            with Image.open(p) as im:
                info.update(size=list(im.size), mode=im.mode)
                arr = np.asarray(im)
        elif e == ".csv":
            info["header"] = p.open(encoding="utf-8", errors="replace").readline().strip().split(",")
            arr = None
        else:
            arr = None
        if arr is not None and arr.size and MASK_HINT.search(p.as_posix()):
            vals = np.unique(arr)
            info["unique_values"] = vals[:32].tolist()
    except ImportError as exc:
        info["error"] = f"install {exc.name} to inspect this file type"
    except Exception as exc:  # inspection must never crash on one bad file
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def inspect_dir(root: Path, per_type: int = 3) -> Dict[str, Any]:
    if not root.exists():
        return {"present": False}
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext = Counter(ext_of(p) for p in files)
    masks = [p for p in files if MASK_HINT.search(p.relative_to(root).as_posix())]
    samples: List[Dict[str, Any]] = []
    for e in by_ext:
        group = [p for p in files if ext_of(p) == e]
        mask_group = [p for p in group if p in set(masks)]
        picks = group[:per_type] + mask_group[:per_type]
        samples += [sample_file(p) for p in dict.fromkeys(picks)]
    tables = [str(p.relative_to(root)) for p in files if p.suffix.lower() == ".csv"]
    return {
        "present": True,
        "files": len(files),
        "by_extension": dict(by_ext),
        "mask_like_files": len(masks),
        "tables": tables,
        "samples": samples,
    }


def _is_volumetric(summary: Dict[str, Any]) -> bool:
    exts = set(summary.get("by_extension", {}))
    return bool(exts & VOLUME_EXT)


def decide(key: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Admission rule: pixel-level masks AND volumetric CT/MRI (or slice stacks that rebuild a volume)."""
    if not summary.get("present"):
        return {"status": "pending inspection", "reason": f"data/raw/{key} not found; run the download script first"}
    has_masks = summary["mask_like_files"] > 0 or any(
        any(MASK_HINT.search(k) for k in s.get("keys", {})) for s in summary["samples"]
    )
    volumetric = _is_volumetric(summary)
    exts = summary["by_extension"]
    if key == "brats2020":
        if has_masks:
            fmt = "per-slice .h5" if (".h5" in exts or ".hdf5" in exts) else "NIfTI volumes"
            return {"status": "admitted", "supervises": ["brain"], "format": fmt, "modality": "MRI (T1, T1ce, T2, FLAIR)",
                    "reason": "Segmentation masks found; whole tumour (labels 1, 2, 4) maps to the brain channel."}
        return {"status": "rejected", "reason": "No segmentation masks found."}
    if key == "lits":
        if has_masks:
            fmt = "NIfTI volumes" if volumetric else "2D slices (rebuilt into volumes per patient)"
            return {"status": "admitted", "supervises": ["liver"], "format": fmt, "modality": "CT",
                    "reason": "Liver masks found; liver and tumour merge into the liver channel."}
        return {"status": "rejected", "reason": "No liver masks found."}
    if key == "cervical":
        if not has_masks:
            return {"status": "rejected", "reason": "No pixel-level masks: classification labels cannot supervise a segmentation channel (same reason SipakMed was rejected)."}
        if not volumetric:
            return {"status": "rejected", "format": "2D images with masks",
                    "reason": "Masks exist but the images are 2D (not volumetric CT/MRI). Feeding them to the 3D network would supervise the cervix channel with a modality it never sees at inference. Kept for a separate 2D study; cervix stays unsupervised."}
        return {"status": "admitted", "supervises": ["cervix"], "format": "volumetric with masks",
                "reason": "Volumetric images with cervix masks found."}
    return {"status": "rejected", "reason": "Unknown corpus key."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "inspection_report.json"))
    args = ap.parse_args()
    raw = Path(args.raw)
    datasets, decisions = {}, {}
    for key in ("brats2020", "lits", "cervical"):
        summary = inspect_dir(raw / key)
        datasets[key] = summary
        decisions[key] = decide(key, summary)
        print(f"{key:<10} {decisions[key]['status']:<20} {decisions[key]['reason']}")
    report = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "raw_root": str(raw),
              "datasets": datasets, "decisions": decisions}
    Path(args.out).write_text(to_json(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
