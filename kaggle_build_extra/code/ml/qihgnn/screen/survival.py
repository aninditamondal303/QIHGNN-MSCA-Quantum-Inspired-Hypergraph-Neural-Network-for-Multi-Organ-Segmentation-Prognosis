"""Prognosis: a ridge-penalised Cox proportional-hazards model on BraTS 2020 survival data.

Covariates (all available to a user of the website): age, whole-tumour volume (measured by the
segmenter on a 3D scan, or typed in), and extent of resection (GTR / STR / unknown).

Everything reported is out-of-sample: repeated 5-fold cross-validated concordance against an
age-only and a volume-only model, Kaplan-Meier curves of out-of-fold risk tertiles with a log-rank
test, and the BraTS challenge's short/mid/long survivor accuracy. The exported model carries 200
bootstrap refits so the browser can show an uncertainty band rather than a single curve.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

FEATURES = ("age_years", "log_wt_volume_cc", "resection_gtr", "resection_str")
SHORT_DAYS, LONG_DAYS = 300, 450   # BraTS survival classes: <10 months, 10-15 months, >15 months


def design(age: Sequence[float], wt_cc: Sequence[float], resection: Sequence[str]) -> np.ndarray:
    r = [str(x or "").strip().upper() for x in resection]
    return np.column_stack([
        np.asarray(age, float),
        np.log1p(np.maximum(np.asarray(wt_cc, float), 0.0)),
        [1.0 if x == "GTR" else 0.0 for x in r],
        [1.0 if x == "STR" else 0.0 for x in r],
    ])


class Cox:
    """Breslow partial likelihood, Newton-Raphson, L2 penalty on standardised coefficients."""

    def __init__(self, l2: float = 1.0, cols: Sequence[int] | None = None):
        self.l2, self.cols = l2, cols

    def _x(self, X: np.ndarray) -> np.ndarray:
        X = X if self.cols is None else X[:, list(self.cols)]
        return (X - self.mu) / self.sd

    def fit(self, X: np.ndarray, t: np.ndarray, e: np.ndarray) -> "Cox":
        Xs = X if self.cols is None else X[:, list(self.cols)]
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-9
        Z = self._x(X)
        order = np.argsort(-t, kind="mergesort")       # descending time: risk sets are prefixes
        Z, t_, e_ = Z[order], t[order], e[order].astype(float)
        beta = np.zeros(Z.shape[1])
        for _ in range(50):
            eta = Z @ beta
            w = np.exp(eta - eta.max())
            cw = np.cumsum(w)
            cwz = np.cumsum(w[:, None] * Z, 0)
            cwzz = np.cumsum(w[:, None, None] * Z[:, :, None] * Z[:, None, :], 0)
            # Breslow ties: every tied event uses the risk set that includes all subjects with time >= t.
            last = np.searchsorted(-t_, -t_, side="right") - 1
            S0, S1, S2 = cw[last], cwz[last], cwzz[last]
            mean = S1 / S0[:, None]
            grad = (e_[:, None] * (Z - mean)).sum(0) - self.l2 * beta
            hess = -(e_[:, None, None] * (S2 / S0[:, None, None] - mean[:, :, None] * mean[:, None, :])).sum(0)
            hess -= self.l2 * np.eye(len(beta))
            step = np.linalg.solve(hess, grad)
            beta = beta - step
            if np.abs(step).max() < 1e-8:
                break
        self.beta = beta
        # Breslow baseline cumulative hazard at the distinct event times (for the centred covariates).
        risk = np.exp(self._x(X) @ beta)
        times = np.unique(t[e == 1])
        dH = np.array([e[t == s].sum() / risk[t >= s].sum() for s in times])
        self.event_times, self.H0 = times, np.cumsum(dH)
        return self

    def lp(self, X: np.ndarray) -> np.ndarray:
        return self._x(X) @ self.beta

    def H0_at(self, grid: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.event_times, grid, side="right") - 1
        return np.where(idx >= 0, self.H0[np.clip(idx, 0, None)], 0.0)

    def median_survival(self, X: np.ndarray, grid: np.ndarray) -> np.ndarray:
        S = np.exp(-np.outer(np.exp(self.lp(X)), self.H0_at(grid)))
        below = S <= 0.5
        return np.where(below.any(1), grid[np.argmax(below, 1)], np.nan)


def concordance(risk: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """Harrell's C: among comparable pairs (earlier time had the event), share where the earlier has higher risk."""
    num = den = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        later = t > t[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(num / den) if den else float("nan")


def kaplan_meier(t: np.ndarray, e: np.ndarray) -> Tuple[List[float], List[float]]:
    times, surv, s = [0.0], [1.0], 1.0
    for u in np.unique(t[e == 1]):
        at_risk, d = (t >= u).sum(), ((t == u) & (e == 1)).sum()
        s *= 1 - d / at_risk
        times.append(float(u))
        surv.append(float(s))
    return times, surv


def logrank_p(t: np.ndarray, e: np.ndarray, group: np.ndarray) -> float:
    """k-group log-rank test; p-value from the chi-square distribution with k-1 dof."""
    gs = np.unique(group)
    O, E = np.zeros(len(gs)), np.zeros(len(gs))
    V = np.zeros((len(gs), len(gs)))
    for u in np.unique(t[e == 1]):
        at = t >= u
        n = at.sum()
        d = ((t == u) & (e == 1)).sum()
        ng = np.array([(at & (group == g)).sum() for g in gs], float)
        dg = np.array([((t == u) & (e == 1) & (group == g)).sum() for g in gs], float)
        O += dg
        E += d * ng / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(ng / n) - np.outer(ng, ng) / n ** 2)
    k = len(gs) - 1
    diff = (O - E)[:k]
    stat = float(diff @ np.linalg.pinv(V[:k, :k]) @ diff)
    return _chi2_sf(stat, k)


def _chi2_sf(x: float, k: int) -> float:
    if k == 2:
        return math.exp(-x / 2)
    if k == 1:
        return math.erfc(math.sqrt(x / 2))
    # Regularised upper gamma via series for general k (k small here).
    a, z = k / 2, x / 2
    term, total, n = 1 / a, 1 / a, 0
    while term > 1e-12 * total and n < 500:
        n += 1
        term *= z / (a + n)
        total += term
    lower = math.exp(-z + a * math.log(z) - math.lgamma(a)) * total if z > 0 else 0.0
    return max(0.0, 1 - lower)


def fit_and_report(X: np.ndarray, t: np.ndarray, e: np.ndarray, seed: int = 2026, l2: float = 1.0,
                   repeats: int = 5, folds: int = 5, boots: int = 200) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(t)
    variants = {"full": None, "age_only": [0], "volume_only": [1], "age_and_resection": [0, 2, 3]}
    cv = {k: [] for k in variants}
    oof_lp = np.zeros((repeats, n))
    oof_median = np.full((repeats, n), np.nan)
    grid = np.linspace(0, float(np.percentile(t, 98)), 80)
    for r in range(repeats):
        perm = rng.permutation(n)
        for f in range(folds):
            te = perm[f::folds]
            tr = np.setdiff1d(perm, te)
            for name, cols in variants.items():
                m = Cox(l2, cols).fit(X[tr], t[tr], e[tr])
                lp = m.lp(X[te])
                cv[name].append(concordance(lp, t[te], e[te]))
                if name == "full":
                    oof_lp[r, te] = lp
                    oof_median[r, te] = m.median_survival(X[te], grid)
    lp_mean = oof_lp.mean(0)
    tert = np.digitize(lp_mean, np.percentile(lp_mean, [33.3, 66.7]))  # 0 low, 1 mid, 2 high risk
    km = {name: dict(zip(("t", "s"), kaplan_meier(t[tert == g], e[tert == g])))
          for g, name in enumerate(("low", "intermediate", "high"))}
    for g, name in enumerate(("low", "intermediate", "high")):
        km[name]["n"] = int((tert == g).sum())
    # BraTS survival classes from the out-of-fold predicted median (uncensored patients only, as in the challenge).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # patients whose curve never drops below 0.5
        med = np.nanmean(oof_median, 0)
    cls = lambda d: np.where(d < SHORT_DAYS, 0, np.where(d <= LONG_DAYS, 1, 2))  # noqa: E731
    ev = (e == 1) & ~np.isnan(med)
    class_acc = float((cls(med[ev]) == cls(t[ev])).mean()) if ev.any() else float("nan")
    majority = float(np.bincount(cls(t[ev])).max() / ev.sum()) if ev.any() else float("nan")

    final = Cox(l2).fit(X, t, e)
    boot_beta, boot_H0 = [], []
    for _ in range(boots):
        idx = rng.integers(0, n, n)
        m = Cox(l2).fit(X[idx], t[idx], e[idx])
        # Express each refit in the final model's standardisation so the browser uses one mu/sd.
        boot_beta.append((m.beta / m.sd * final.sd).tolist())
        shift = float((final.mu - m.mu) / m.sd @ m.beta)  # lp_m(x) = lp_final-scale(x) + shift
        boot_H0.append((m.H0_at(grid) * math.exp(shift)).tolist())
    return {
        "features": list(FEATURES), "n": int(n), "events": int(e.sum()), "l2": l2,
        "cv_c_index": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "folds": len(v)} for k, v in cv.items()},
        "risk_groups": {"km": km, "logrank_p": logrank_p(t, e, tert)},
        "survival_class_accuracy": {"accuracy": class_acc, "majority_class_baseline": majority, "n": int(ev.sum()),
                                    "classes_days": [SHORT_DAYS, LONG_DAYS]},
        "model": {"mu": final.mu.tolist(), "sd": final.sd.tolist(), "beta": final.beta.tolist(),
                  "grid_days": grid.tolist(), "H0": final.H0_at(grid).tolist(),
                  "hazard_ratio_per_sd": np.exp(final.beta).tolist(),
                  "lp_tertiles": np.percentile(final.lp(X), [100 / 3, 200 / 3]).tolist(),
                  "boot_beta": boot_beta, "boot_H0": boot_H0},
    }
