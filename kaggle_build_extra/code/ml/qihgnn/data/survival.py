"""Parse BraTS-style survival tables into (time, event, covariates).

BraTS 2020 ships `survival_info.csv` with Brats20ID, Age, Survival_days,
Extent_of_Resection. Survival_days is usually a number (death observed) but can
read e.g. "ALIVE (361 days later)", which is a right-censored observation.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ALIVE = re.compile(r"alive\D*(\d+(?:\.\d+)?)", re.I)

# Covariate vector (5): age z-score, resection GTR, resection STR, resection unknown, has-clinical-record.
COVARIATE_NAMES = ("age_z", "resection_gtr", "resection_str", "resection_unknown", "has_clinical")
AGE_MEAN, AGE_SD = 61.0, 12.0  # population-level constants so test data never leaks into normalisation


@dataclass
class SurvivalRecord:
    patient_id: str
    time_days: float
    event: int
    covariates: List[float]


def parse_days(value: str) -> tuple[Optional[float], int]:
    v = (value or "").strip()
    if not v:
        return None, 0
    m = ALIVE.search(v)
    if m:
        return float(m.group(1)), 0
    try:
        return float(v), 1
    except ValueError:
        return None, 0


def covariates(age: Optional[float], resection: str) -> List[float]:
    r = (resection or "").strip().upper()
    return [
        0.0 if age is None else (age - AGE_MEAN) / AGE_SD,
        1.0 if r == "GTR" else 0.0,
        1.0 if r == "STR" else 0.0,
        1.0 if r not in {"GTR", "STR"} else 0.0,
        1.0,
    ]


def read_survival_csv(path: Path) -> Dict[str, SurvivalRecord]:
    out: Dict[str, SurvivalRecord] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = {k.lower(): k for k in reader.fieldnames or []}
        id_key = next((fields[k] for k in fields if "id" in k), None)
        days_key = next((fields[k] for k in fields if "surviv" in k), None)
        age_key = next((fields[k] for k in fields if "age" in k), None)
        res_key = next((fields[k] for k in fields if "resection" in k), None)
        if not id_key or not days_key:
            raise ValueError(f"{path} lacks an ID or survival column: {reader.fieldnames}")
        for row in reader:
            t, e = parse_days(row.get(days_key, ""))
            if t is None:
                continue
            try:
                age = float(row[age_key]) if age_key and row.get(age_key) else None
            except ValueError:
                age = None
            pid = row[id_key].strip()
            out[pid] = SurvivalRecord(pid, t, e, covariates(age, row.get(res_key, "") if res_key else ""))
    return out
