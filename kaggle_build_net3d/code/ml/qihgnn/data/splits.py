"""Patient-level splits. Slices of one patient never appear in two splits."""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List


def split_of(patient_id: str, fractions=(0.70, 0.15, 0.15), salt: str = "qihgnn-2026") -> str:
    h = int(hashlib.sha1(f"{salt}:{patient_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fractions[0]:
        return "train"
    if h < fractions[0] + fractions[1]:
        return "val"
    return "test"


def assign(patient_ids: Iterable[str], **kw) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for pid in sorted(set(patient_ids)):
        out[split_of(pid, **kw)].append(pid)
    return out


def assert_disjoint(splits: Dict[str, List[str]]) -> None:
    seen: Dict[str, str] = {}
    for name, ids in splits.items():
        for pid in ids:
            if pid in seen and seen[pid] != name:
                raise AssertionError(f"patient {pid} appears in both {seen[pid]} and {name}")
            seen[pid] = name
