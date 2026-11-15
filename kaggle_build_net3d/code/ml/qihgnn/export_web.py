"""Export everything the website reads into web/public/data.

Volumes are written as lossless 8-bit PNG mosaics: the D axial slices of an
(H, W, D) volume are tiled into a grid, so a 128 x 128 x 64 volume becomes one
1024 x 1024 PNG per channel. Browsers decode PNG natively, which keeps the site
free of custom binary formats.

Run: `python -m qihgnn.export_web` (from ml/). Add `--cases-from-cache N` after
preprocessing real data to also publish N held-out real cases.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image

from .config import LABEL_DESCRIPTIONS, LABELS
from .data.registry import corpus_summary
from .phantoms import Phantom, all_phantoms
from .verify import ROOT, build_report, to_json

WEB_DATA = ROOT / "web" / "public" / "data"


def mosaic(vol: np.ndarray) -> tuple[np.ndarray, int]:
    """(H, W, D) uint8 -> (rows*H, cols*W) with slice k at row k // cols, col k % cols."""
    h, w, d = vol.shape
    cols = int(math.ceil(math.sqrt(d)))
    rows = int(math.ceil(d / cols))
    out = np.zeros((rows * h, cols * w), np.uint8)
    for k in range(d):
        r, c = divmod(k, cols)
        # Store slices with rows = first image axis (H), columns = second axis (W).
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = vol[:, :, k]
    return out, cols


def to_uint8(channel: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(channel, [0.5, 99.5])
    if hi - lo < 1e-6:
        return np.zeros(channel.shape, np.uint8)
    return (np.clip((channel - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def export_case(case: Phantom, out_dir: Path, synthetic: bool = True) -> Dict[str, Any]:
    d = out_dir / case.case_id
    d.mkdir(parents=True, exist_ok=True)
    channel_files = []
    cols = 0
    for i, name in enumerate(case.channels):
        img, cols = mosaic(to_uint8(case.image[i]))
        fname = f"ch{i}.png"
        Image.fromarray(img, mode="L").save(d / fname, optimize=True)
        channel_files.append({"name": name, "file": f"{case.case_id}/{fname}"})
    lab, _ = mosaic(case.label.astype(np.uint8))
    Image.fromarray(lab, mode="L").save(d / "label.png", optimize=True)
    present = sorted({int(v) for v in np.unique(case.label)} - {0})
    voxels = {LABELS[i]: int((case.label == i).sum()) for i in present}
    return {
        "id": case.case_id,
        "title": case.title,
        "modality": case.modality,
        "analogOf": case.analog_of,
        "synthetic": synthetic,
        "shape": list(case.label.shape),
        "mosaicColumns": cols,
        "channels": channel_files,
        "label": f"{case.case_id}/label.png",
        "spacing": [1, 1, round(case.label.shape[0] / case.label.shape[2], 3)],
        "supervised": [LABELS[i] for i in np.flatnonzero(case.supervised)],
        "present": [LABELS[i] for i in present],
        "voxels": voxels,
        "note": case.note,
    }


def export_cases(cases: Iterable[Phantom], synthetic: bool = True) -> list:
    out_dir = WEB_DATA / "cases"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    return [export_case(c, out_dir, synthetic) for c in cases]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-from-cache", type=int, default=0, help="also export N real test cases from data/cache")
    args = ap.parse_args()
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    cases = list(all_phantoms())
    index = export_cases(cases, synthetic=True)
    if args.cases_from_cache:
        from .data.cache import load_real_cases_for_web

        real = load_real_cases_for_web(args.cases_from_cache)
        out_dir = WEB_DATA / "cases"
        index += [export_case(c, out_dir, synthetic=False) for c in real]
    (WEB_DATA / "cases.json").write_text(json.dumps({"labels": list(LABELS), "labelDescriptions": LABEL_DESCRIPTIONS, "cases": index}, indent=1))

    verification = ROOT / "results" / "verification.json"
    report = json.loads(verification.read_text()) if verification.exists() else build_report()
    (WEB_DATA / "verification.json").write_text(to_json(report))
    (WEB_DATA / "corpus.json").write_text(json.dumps(corpus_summary(ROOT), indent=1))
    metrics = ROOT / "results" / "metrics.json"
    if metrics.exists():
        shutil.copy(metrics, WEB_DATA / "metrics.json")
    elif (WEB_DATA / "metrics.json").exists():
        (WEB_DATA / "metrics.json").unlink()
    history = ROOT / "results" / "training_history.json"
    if history.exists():
        shutil.copy(history, WEB_DATA / "training_history.json")
    files = sorted(p.name for p in WEB_DATA.glob("*.json") if p.name != "manifest.json")
    (WEB_DATA / "manifest.json").write_text(json.dumps({"files": files}, indent=1))
    print(f"exported {len(index)} cases and JSON to {WEB_DATA}")


if __name__ == "__main__":
    main()
