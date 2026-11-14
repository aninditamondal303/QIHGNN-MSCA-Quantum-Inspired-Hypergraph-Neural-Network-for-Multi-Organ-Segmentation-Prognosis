"""Fast recursive file listing.

Kaggle mounts datasets on a network filesystem where every stat() is a round trip.
os.walk reads entry types from the directory listing itself, so listing 117k slice
files costs one call per folder instead of one per file (Path.rglob + is_file did both).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


def list_files(root: Path, suffixes: Tuple[str, ...] = ()) -> List[Path]:
    out: List[Path] = []
    for cur, dirs, files in os.walk(root, followlinks=True):
        dirs.sort()
        for f in sorted(files):
            if not suffixes or f.lower().endswith(suffixes):
                out.append(Path(cur) / f)
    return out
