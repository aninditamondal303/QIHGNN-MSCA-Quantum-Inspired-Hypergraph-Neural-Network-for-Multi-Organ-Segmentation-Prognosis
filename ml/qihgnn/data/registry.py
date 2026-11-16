"""Corpus registry: what each dataset is, where it comes from and what it may supervise.

The supervision a corpus contributes is decided here and by the inspection
report, never by the training loop. A corpus is admitted for a channel only
if it provides pixel-level masks for that organ in volumetric CT or MRI.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..config import LABELS


@dataclass
class Corpus:
    key: str
    name: str
    kaggle: Optional[str]
    modality: str
    claimed_channels: List[str]
    expected_format: str
    report_cases: Optional[int]
    notes: str
    status: str = "pending inspection"  # admitted | rejected | pending inspection | not registered
    supervises: List[str] = field(default_factory=list)
    reason: str = ""


CORPORA: List[Corpus] = [
    Corpus(
        key="brats2020",
        name="BraTS 2020",
        kaggle="awsaf49/brats2020-training-data",
        modality="MRI (T1, T1ce, T2, FLAIR)",
        claimed_channels=["brain"],
        expected_format="NIfTI volumes or per-slice .h5 files (image 240x240x4, mask 240x240x3)",
        report_cases=368,
        notes="Survival data (age, survival days, resection) may be included; if so the prognosis head trains on it.",
    ),
    Corpus(
        key="lits",
        name="LiTS",
        kaggle="harshwardhanbhangale/lits-dataset",
        modality="CT (portal venous)",
        claimed_channels=["liver"],
        expected_format="NIfTI volumes (volume-*.nii, segmentation-*.nii) or PNG slices",
        report_cases=131,
        notes="Liver and tumour labels are merged into the liver channel.",
    ),
    Corpus(
        key="cervical",
        name="Cervical cancer segmentation",
        kaggle="balajikartheek/cervical-cancer-segmentation",
        modality="Modality and format are determined by inspection",
        claimed_channels=["cervix"],
        expected_format="",
        report_cases=None,
        notes="Admitted for the cervix channel only if it holds pixel-level cervix masks in volumetric CT/MRI. "
              "2D colposcopy or cytology images are rejected for the 3D network, as SipakMed was in the report.",
    ),
    Corpus(
        key="kits19",
        name="KiTS19",
        kaggle=None,
        modality="CT",
        claimed_channels=["kidney"],
        expected_format="NIfTI volumes",
        report_cases=210,
        notes="Used in the report; not part of the current dataset list. Kidney stays unsupervised until it is added.",
        status="not registered",
    ),
]


def load_inspection(root: Path) -> Optional[Dict]:
    path = root / "data" / "inspection_report.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def resolved_corpora(root: Path) -> List[Corpus]:
    """Apply the inspection report's decisions to the registry."""
    inspection = load_inspection(root) or {}
    decisions = inspection.get("decisions", {})
    out = []
    for c in CORPORA:
        c = Corpus(**asdict(c))
        if c.key in decisions:
            dec = decisions[c.key]
            c.status = dec["status"]
            c.supervises = dec.get("supervises", [])
            c.reason = dec.get("reason", "")
            if dec.get("modality"):
                c.modality = dec["modality"]
            if dec.get("format"):
                c.expected_format = dec["format"]
        out.append(c)
    return out


def supervised_channels(root: Path) -> List[str]:
    sup = {"background"}
    for c in resolved_corpora(root):
        if c.status == "admitted":
            sup.update(c.supervises)
    return [l for l in LABELS if l in sup]


def corpus_summary(root: Path) -> Dict:
    corpora = resolved_corpora(root)
    inspection = load_inspection(root)
    sup = supervised_channels(root)
    return {
        "inspected": inspection is not None,
        "inspectedAt": (inspection or {}).get("generated_at"),
        "labels": list(LABELS),
        # the website shows only the dataset's own name, never the uploader's account name
        "corpora": [asdict(c) | {"kaggle": c.kaggle.split("/")[-1] if c.kaggle else None,
                                 "cases": ((inspection or {}).get("datasets", {}).get(c.key, {}) or {}).get("cases")} for c in corpora],
        "supervisedChannels": sup,
        "unsupervisedChannels": [l for l in LABELS if l not in sup],
        "reportCorpus": {
            "cases": 709,
            "rows": [
                {"name": "BraTS 2020", "modality": "MRI, 4 sequences", "format": "3D NIfTI", "cases": 368, "supervises": "brain"},
                {"name": "LiTS", "modality": "CT", "format": "2D PNG slices", "cases": 131, "supervises": "liver"},
                {"name": "KiTS19", "modality": "CT", "format": "2D PNG slices", "cases": 210, "supervises": "kidney"},
                {"name": "SipakMed", "modality": "Cytology", "format": "2D cell crops", "cases": None, "supervises": "rejected"},
            ],
        },
    }
