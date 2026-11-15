"""Preprocess admitted corpora once into data/cache/*.npz plus an index with patient-level splits.

Run: `python -m qihgnn.data.cache` (from ml/). Needs data/inspection_report.json first.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..config import LABELS, ModelConfig
from ..verify import ROOT
from .adapters import RawCase, tasks_for
from .preprocess import to_grid
from .registry import resolved_corpora
from .splits import assert_disjoint, split_of

CACHE = ROOT / "data" / "cache"


def save_case(case: RawCase, shape, out: Path) -> Dict:
    image = to_grid(case.image, shape, order=1).astype(np.float16)
    label = to_grid(case.label, shape, order=0).astype(np.uint8)
    name = f"{case.source}__{case.patient_id}.npz"
    np.savez_compressed(out / name, image=image, label=label, supervised=case.supervised)
    return {
        "file": name, "patient_id": case.patient_id, "source": case.source, "modality": case.modality,
        "split": split_of(case.patient_id),
        "supervised": [LABELS[i] for i in np.flatnonzero(case.supervised)],
        "has_survival": case.time_days is not None,
        "time_days": case.time_days, "event": case.event, "covariates": case.covariates,
        "voxels": {LABELS[i]: int((label == i).sum()) for i in range(1, len(LABELS))},
    }


def build(limit: int = 0, workers: int = 0) -> Dict:
    workers = workers or min(6, os.cpu_count() or 2)
    shape = ModelConfig().input_shape
    CACHE.mkdir(parents=True, exist_ok=True)
    entries: List[Dict] = []
    for corpus in resolved_corpora(ROOT):
        if corpus.status != "admitted":
            print(f"skip {corpus.key}: {corpus.status}")
            continue
        t0 = time.time()
        tasks = tasks_for(corpus.key, ROOT / "data" / "raw" / corpus.key)
        if limit:
            tasks = tasks[:limit]
        print(f"{corpus.key}: {len(tasks)} cases located in {time.time() - t0:.0f}s", flush=True)

        def work(task, key=corpus.key):
            entry = save_case(task.load(), shape, CACHE)
            print(f"  {key} {entry['patient_id']}", flush=True)
            return entry

        # Several cases in flight at once: reading is I/O-bound and numpy/scipy release the GIL.
        with ThreadPoolExecutor(workers) as pool:
            entries += list(pool.map(work, tasks))
        print(f"{corpus.key}: cached {len(tasks)} cases in {(time.time() - t0) / 60:.1f} min", flush=True)
    splits = {s: sorted({e["patient_id"] for e in entries if e["split"] == s}) for s in ("train", "val", "test")}
    assert_disjoint(splits)
    index = {"shape": list(shape), "labels": list(LABELS), "cases": entries,
             "counts": {s: len(v) for s, v in splits.items()}}
    (CACHE / "index.json").write_text(json.dumps(index, indent=1))
    print(f"cached {len(entries)} cases -> {CACHE}")
    return index


def load_real_cases_for_web(n: int):
    """Pick n held-out test cases from the cache for publication on the website."""
    from ..phantoms import Phantom

    index = json.loads((CACHE / "index.json").read_text())
    picks = [e for e in index["cases"] if e["split"] == "test"][:n]
    out = []
    for e in picks:
        z = np.load(CACHE / e["file"])
        chans = ["T1", "T1ce", "T2", "FLAIR"] if e["modality"] == "MRI" else ["Abdomen window", "Liver window", "Bone window", "Wide window"]
        out.append(Phantom(
            case_id=f"{e['source'].upper()}-{e['patient_id']}"[:40], title=f"{e['source']} test case {e['patient_id']}",
            modality=e["modality"], analog_of=e["source"], channels=chans,
            image=z["image"].astype(np.float32), label=z["label"], supervised=z["supervised"],
            note="Held-out test case from the public corpus.",
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max cases per corpus (0 = all)")
    ap.add_argument("--workers", type=int, default=0, help="cases preprocessed concurrently (0 = auto)")
    args = ap.parse_args()
    build(args.limit, args.workers)


if __name__ == "__main__":
    main()
