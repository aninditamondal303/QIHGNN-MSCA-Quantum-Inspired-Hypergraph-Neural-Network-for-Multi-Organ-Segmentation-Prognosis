"""Numerical verification of every property the report claims (Sections 4.11, 5.3-5.7).

Runs on NumPy only. Writes `results/verification.json`, which the website and the
API read. Run: `python -m qihgnn.verify` (from the `ml/` folder).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

import numpy as np

from . import reference as R
from .config import LABELS, REPORT_PARAMETER_BUDGET, REPORT_TOTAL_PARAMETERS, ModelConfig
from .param_count import attention_cost, mpo_compression_table, parameter_budget

ROOT = Path(__file__).resolve().parents[2]


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def synthetic_case(seed: int = 7, shape=(16, 16, 16)):
    """One liver-only sample: background + liver supervised, four channels unsupervised."""
    rng = np.random.default_rng(seed)
    c = len(LABELS)
    v = int(np.prod(shape))
    zz, yy, xx = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    y = np.zeros(shape, dtype=np.int64)
    y[(xx / 0.6) ** 2 + (yy / 0.45) ** 2 + (zz / 0.5) ** 2 < 1] = LABELS.index("liver")
    logits = rng.normal(0, 2.0, size=(c, v))
    logits[LABELS.index("liver"), y.ravel() == LABELS.index("liver")] += 1.5
    p = R.softmax(logits, axis=0)
    supervised = np.zeros(c, dtype=bool)
    supervised[[0, LABELS.index("liver")]] = True
    return p, y.ravel(), supervised


def verify_marginal(seed: int = 7) -> Dict[str, Any]:
    p, y, sup = synthetic_case(seed)
    base_m = R.marginal_loss(p, y, sup)
    base_n = R.naive_loss(p, y)
    rows = []
    unsup = np.flatnonzero(~sup)
    for c in unsup:
        q = p.copy()
        q[c] = p[unsup].sum(axis=0)  # concentrate ALL unsupervised mass into channel c
        for other in unsup:
            if other != c:
                q[other] = 0.0
        m = R.marginal_loss(q, y, sup)
        n = R.naive_loss(q, y)
        rows.append({
            "channel": LABELS[c],
            "marginal_original": base_m, "marginal_perturbed": m, "marginal_change": abs(m - base_m),
            "naive_original": base_n, "naive_perturbed": n, "naive_change": abs(n - base_n),
        })
    return {
        "seed": seed,
        "voxels": int(y.size),
        "supervised_channels": [LABELS[i] for i in np.flatnonzero(sup)],
        "rows": rows,
        "max_marginal_change": max(r["marginal_change"] for r in rows),
        "min_naive_change": min(r["naive_change"] for r in rows),
        "passed": max(r["marginal_change"] for r in rows) < 1e-12 and min(r["naive_change"] for r in rows) > 1e-3,
    }


def shift_to_channel(p: np.ndarray, y: np.ndarray, target: int, t: float) -> np.ndarray:
    """On voxels labelled background, hand a fraction t of the background probability to `target`.

    This is the model correctly recognising an organ the sample's dataset never annotated.
    """
    q = p.copy()
    bg = y == 0
    delta = t * q[0, bg]
    q[0, bg] -= delta
    q[target, bg] += delta
    return q


def verify_unlabelled_prediction(seed: int = 7, t: float = 0.5) -> Dict[str, Any]:
    """The failure the marginal loss exists to prevent: a naive loss punishes correct predictions
    of unannotated organs, the marginal loss does not."""
    p, y, sup = synthetic_case(seed)
    rows = []
    for c in np.flatnonzero(~sup):
        q = shift_to_channel(p, y, int(c), t)
        rows.append({
            "channel": LABELS[c],
            "marginal_change": abs(R.marginal_loss(q, y, sup) - R.marginal_loss(p, y, sup)),
            "naive_change": R.naive_loss(q, y) - R.naive_loss(p, y),
        })
    return {
        "fraction_moved": t, "rows": rows,
        "max_marginal_change": max(r["marginal_change"] for r in rows),
        "min_naive_increase": min(r["naive_change"] for r in rows),
        "passed": max(r["marginal_change"] for r in rows) < 1e-12 and min(r["naive_change"] for r in rows) > 0.05,
    }


def verify_cox(seed: int = 11, n: int = 16) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    eta = rng.normal(0, 1, n)
    t = rng.integers(30, 2000, n).astype(float)
    t[3] = t[5]  # include a tie so the Breslow path is exercised
    e = rng.random(n) < 0.7
    e[0] = True
    base = R.cox_loss(eta, t, e)
    shifted = R.cox_loss(eta + 37.5, t, e)
    censored = R.cox_loss(eta, t, np.zeros(n, dtype=bool))
    return {
        "seed": seed, "cases": n, "events": int(e.sum()),
        "loss_original": base, "loss_shifted_37_5": shifted, "shift_change": abs(shifted - base),
        "loss_all_censored": censored,
        "passed": abs(shifted - base) < 1e-9 and censored == 0.0,
    }


def verify_uncertainty(losses=(1.3957, 0.0412, 1.1058, 0.6931), steps: int = 4000, lr: float = 0.05) -> Dict[str, Any]:
    L = np.asarray(losses, dtype=np.float64)
    s = np.zeros_like(L)
    for _ in range(steps):
        grad = -np.exp(-s) * L + 1.0  # d/ds [exp(-s) L + s]
        s -= lr * grad
    target = np.log(L)
    return {
        "task_losses": L.tolist(), "converged_log_vars": s.tolist(), "expected_log_L": target.tolist(),
        "effective_weights": np.exp(-s).tolist(), "max_abs_error": float(np.abs(s - target).max()),
        "passed": bool(np.abs(s - target).max() < 1e-6),
    }


def verify_dice() -> Dict[str, Any]:
    a = np.zeros((10, 10, 10), bool)
    a[:5, :4, :5] = True  # 100 voxels
    b = np.zeros_like(a)
    b[:5, 2:6, :5] = True  # 100 voxels, 50 shared
    disjoint = np.zeros_like(a)
    disjoint[6:, 6:, 6:] = True
    cases = {
        "perfect_overlap": (R.dice_coefficient(a, a), 1.0),
        "complete_disjunction": (R.dice_coefficient(a, disjoint), 0.0),
        "half_overlap": (R.dice_coefficient(a, b), 0.5),
    }
    return {
        "cases": {k: {"observed": v[0], "expected": v[1]} for k, v in cases.items()},
        "passed": all(abs(v[0] - v[1]) < 1e-12 for v in cases.values()),
    }


def verify_horizons() -> Dict[str, Any]:
    days = 365.25
    y, mask = R.horizon_labels([2 * days], [0], [1 * days, 3 * days, 5 * days])
    return {
        "case": "censored at 2 years",
        "label_alive": y[0].tolist(), "determined": mask[0].tolist(),
        "passed": mask[0].tolist() == [True, False, False] and y[0][0] == 1.0,
    }


def verify_exclusion(seed: int = 3) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(len(LABELS), 512))
    softmax_value = R.exclusion_loss(R.softmax(logits, axis=0))
    sig = 1 / (1 + np.exp(-logits))
    sigmoid_value = R.exclusion_loss(sig)
    return {
        "softmax_head": softmax_value, "sigmoid_head": sigmoid_value,
        "finding": "Under a softmax head the exclusion term is identically zero; it is only active for sigmoid heads.",
        "passed": softmax_value == 0.0 and sigmoid_value > 0.0,
    }


def verify_concordance() -> Dict[str, Any]:
    t = np.array([100, 200, 300, 400, 500.0])
    e = np.ones(5, bool)
    perfect = R.concordance_index(-t, t, e)
    reversed_ = R.concordance_index(t, t, e)
    return {"perfect_ordering": perfect, "reversed_ordering": reversed_, "passed": perfect == 1.0 and reversed_ == 0.0}


def build_report() -> Dict[str, Any]:
    cfg = ModelConfig()
    budget = parameter_budget(cfg)
    total = sum(budget.values())
    resolutions = [(64, 64, 32), (128, 128, 64), (256, 256, 128)]
    report: Dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": _git_hash(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "parameters": {
            "stages": budget, "total": total, "fp32_megabytes": round(total * 4 / 2 ** 20, 2),
            "report_stages": REPORT_PARAMETER_BUDGET, "report_total": REPORT_TOTAL_PARAMETERS,
            "matches_report": budget == REPORT_PARAMETER_BUDGET and total == REPORT_TOTAL_PARAMETERS,
        },
        "mpo_compression": mpo_compression_table(),
        "attention": {
            "token_budgets": [256, 32, 4, 1],
            "at_resolution": [{"shape": list(s), **attention_cost(shape=s)} for s in resolutions],
        },
        "marginal_loss": verify_marginal(),
        "unlabelled_prediction": verify_unlabelled_prediction(),
        "cox": verify_cox(),
        "uncertainty_weighting": verify_uncertainty(),
        "dice": verify_dice(),
        "horizon_labels": verify_horizons(),
        "exclusion": verify_exclusion(),
        "concordance": verify_concordance(),
    }
    checks = {k: v["passed"] for k, v in report.items() if isinstance(v, dict) and "passed" in v}
    checks["parameter_budget"] = report["parameters"]["matches_report"]
    report["checks"] = checks
    report["all_passed"] = all(checks.values())
    return report


def web_fixtures() -> Dict[str, Any]:
    """Small cases the TypeScript ports in web/src/lib must reproduce to 1e-9."""
    rng = np.random.default_rng(42)
    c, v = len(LABELS), 48
    logits = rng.normal(0, 1.5, size=(c, v))
    p = R.softmax(logits, axis=0)
    y = rng.choice([0, LABELS.index("liver")], size=v, p=[0.6, 0.4])
    sup = np.zeros(c, bool)
    sup[[0, LABELS.index("liver")]] = True
    return {
        "marginal": {
            "probs": p.tolist(), "labels": y.tolist(), "supervised": sup.tolist(),
            "marginal_loss": R.marginal_loss(p, y, sup), "naive_loss": R.naive_loss(p, y),
        },
        "attention": [{"shape": list(s), **attention_cost(shape=s)} for s in [(64, 64, 32), (128, 128, 64), (200, 180, 90)]],
        "mpo": mpo_compression_table(),
    }


def to_json(obj: Any, **kw) -> str:
    def clean(o):
        # Plain JSON has no NaN/Infinity: undefined metrics are written as null.
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.generic):
            o = o.item()
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        return o

    return json.dumps(clean(obj), allow_nan=False, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "verification.json"))
    ap.add_argument("--fixtures", default=str(ROOT / "web" / "src" / "lib" / "fixtures.json"))
    args = ap.parse_args()
    report = build_report()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(to_json(report, indent=2))
    Path(args.fixtures).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fixtures).write_text(to_json(web_fixtures()))
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"wrote {args.out}")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
