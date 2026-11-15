"""Evaluation protocol (report, 4.10).

  python -m qihgnn.evaluate --checkpoint runs/<id>/best.pt [--label "full model"]
  python -m qihgnn.evaluate --checkpoint a.pt b.pt c.pt --label full baseline no-mpo   # comparison table

Dice and HD95 are reported per organ, only over cases whose corpus supervises
that organ, and never averaged across channels. Prognosis: Harrell's C and
time-dependent AUROC at 1/3/5 years over cases with survival data.
Writes results/metrics.json (read by the website and the API).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import reference as R
from .config import LABELS
from .verify import ROOT, to_json

DAYS = 365.25


def evaluate_checkpoint(checkpoint: Path, split: str = "test", limit: int = 0) -> Dict:
    from .data.dataset import CacheDataset, PhantomDataset
    from .serving import Predictor

    pred = Predictor(checkpoint)
    if pred.meta.get("dataset") == "phantom":
        ds = PhantomDataset(4, pred.cfg.input_shape, seed=999)
        source = "synthetic phantoms (smoke check only, not a performance result)"
    else:
        ds = CacheDataset(ROOT / "data" / "cache", split)
        source = f"{split} split of the preprocessed corpora"
    per_organ: Dict[str, Dict[str, List[float]]] = {}
    risks, times, events = [], [], []
    n = len(ds) if not limit else min(limit, len(ds))
    for i in range(n):
        item = ds[i]
        out = pred.predict(item["image"].numpy(), item["covariates"].tolist())
        truth = item["label"].numpy()
        sup = item["supervised"].numpy()
        for c in np.flatnonzero(sup):
            if c == 0 or LABELS[c] in pred.unsupervised:
                continue
            name = LABELS[c]
            d = per_organ.setdefault(name, {"dice": [], "hd95": []})
            d["dice"].append(R.dice_coefficient(out["label"] == c, truth == c))
            d["hd95"].append(R.hd95(out["label"] == c, truth == c))
        if bool(item["has_survival"]):
            risks.append(out["log_risk"])
            times.append(float(item["time"]))
            events.append(int(item["event"]))
    seg = {}
    for name, d in per_organ.items():
        hd = [h for h in d["hd95"] if not np.isnan(h)]
        seg[name] = {"cases": len(d["dice"]), "dice_mean": float(np.mean(d["dice"])), "dice_sd": float(np.std(d["dice"])),
                     "hd95_mean": float(np.mean(hd)) if hd else None}
    prog = None
    if len(risks) >= 2:
        prog = {"cases": len(risks), "events": int(sum(events)),
                "c_index": R.concordance_index(risks, times, events),
                "td_auc": {f"{y}y": R.time_dependent_auc(risks, times, events, y * DAYS) for y in (1, 3, 5)}}
    return {"checkpoint": str(checkpoint), "run_id": pred.meta.get("run_id"), "epoch": pred.meta.get("epoch"),
            "evaluated_on": source, "cases": n, "suppressed_channels": pred.unsupervised,
            "segmentation": seg, "prognosis": R.summarize(prog) if prog else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "metrics.json"))
    args = ap.parse_args()
    labels = args.label or [Path(c).parent.name for c in args.checkpoint]
    runs = []
    for ck, lab in zip(args.checkpoint, labels):
        r = evaluate_checkpoint(Path(ck), args.split, args.limit)
        r["label"] = lab
        runs.append(r)
        print(json.dumps({"label": lab, "segmentation": r["segmentation"], "prognosis": r["prognosis"]}, indent=1, default=str))
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(to_json(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
