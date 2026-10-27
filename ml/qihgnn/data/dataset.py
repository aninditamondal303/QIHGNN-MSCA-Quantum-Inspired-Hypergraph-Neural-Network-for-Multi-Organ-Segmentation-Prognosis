"""PyTorch datasets and the event-aware batch sampler."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..phantoms import random_training_phantom
from .preprocess import to_grid


def _item(image, label, supervised, time_days, event, covariates, has_survival, pid) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.as_tensor(np.asarray(image, dtype=np.float32)),
        "label": torch.as_tensor(np.asarray(label, dtype=np.int64)),
        "supervised": torch.as_tensor(np.asarray(supervised, dtype=bool)),
        "time": torch.tensor(float(time_days or 0.0)),
        "event": torch.tensor(int(event or 0)),
        "covariates": torch.as_tensor(np.asarray(covariates, dtype=np.float32)),
        "has_survival": torch.tensor(bool(has_survival)),
        "patient_id": pid,
    }


class CacheDataset(Dataset):
    """Reads data/cache/*.npz written by `qihgnn.data.cache` for one split."""

    def __init__(self, cache_dir: Path, split: str, augment: bool = False) -> None:
        self.dir = Path(cache_dir)
        index = json.loads((self.dir / "index.json").read_text())
        self.entries = [e for e in index["cases"] if e["split"] == split]
        if not self.entries:
            raise RuntimeError(f"no cases in split '{split}' under {self.dir}; run python -m qihgnn.data.cache")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        z = np.load(self.dir / e["file"])
        image, label = z["image"].astype(np.float32), z["label"]
        if self.augment and random.random() < 0.5:
            image, label = image[:, :, ::-1].copy(), label[:, ::-1].copy()  # left-right flip
        return _item(image, label, z["supervised"], e["time_days"], e["event"], e["covariates"], e["has_survival"], e["patient_id"])


class PhantomDataset(Dataset):
    """Synthetic phantoms for the smoke run and CI. Never used to report performance."""

    def __init__(self, n: int, shape: Sequence[int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.items = []
        for k in range(n):
            ph = random_training_phantom(rng)
            image = to_grid(ph.image, shape, order=1)
            label = to_grid(ph.label, shape, order=0)
            t = float(rng.integers(60, 2400))
            ev = int(rng.random() < 0.7)
            self.items.append(_item(image, label, ph.supervised, t, ev, ph.covariates, ph.modality == "MRI", ph.case_id + f"-{k}"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(batch: List[Dict]) -> Dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "patient_id"}
    out["patient_id"] = [b["patient_id"] for b in batch]
    return out


class EventAwareBatchSampler(Sampler[List[int]]):
    """Batches of `batch_size`. Survival batches are formed only from survival cases and contain
    at least `min_events` uncensored cases whenever the split has that many, so the
    within-batch Cox risk set is never empty for the cases that can supervise it (report, 4.9).
    """

    def __init__(self, has_survival: Sequence[bool], events: Sequence[int], batch_size: int = 2,
                 min_events: int = 2, seed: int = 0) -> None:
        self.surv = [i for i, h in enumerate(has_survival) if h]
        self.surv_events = [i for i in self.surv if events[i]]
        self.other = [i for i, h in enumerate(has_survival) if not h]
        self.bs, self.min_events = batch_size, min_events
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[List[int]]:
        batches: List[List[int]] = []
        surv = self.surv[:]
        self.rng.shuffle(surv)
        events = [i for i in surv if i in set(self.surv_events)]
        rest = [i for i in surv if i not in set(events)]
        k = min(self.min_events, self.bs)
        while len(events) >= k:
            b = [events.pop() for _ in range(k)]
            while len(b) < self.bs and (rest or events):
                b.append((rest or events).pop())
            batches.append(b)
        leftovers = events + rest + self.other[:]
        self.rng.shuffle(leftovers)
        batches += [leftovers[i:i + self.bs] for i in range(0, len(leftovers), self.bs) if len(leftovers[i:i + self.bs]) == self.bs]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.surv) + len(self.other)) // self.bs
