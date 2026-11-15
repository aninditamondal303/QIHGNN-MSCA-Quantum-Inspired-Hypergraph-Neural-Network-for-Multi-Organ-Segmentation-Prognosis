"""PyTorch composite objective (report, Section 4.8). Mirrors `reference.py` exactly."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

EPS_DICE = 1e-5
EPS_LOG = 1e-8


def merge_unsupervised(probs: torch.Tensor, supervised: torch.Tensor) -> torch.Tensor:
    """probs (B, C, ...), supervised (B, C) bool. Unsupervised mass is folded into channel 0."""
    sup = supervised.clone().bool()
    sup[:, 0] = True
    view = (probs.shape[0], probs.shape[1]) + (1,) * (probs.dim() - 2)
    supf = sup.to(probs.dtype).view(view)
    merged = probs * supf
    unsup_mass = (probs * (1 - supf)).sum(dim=1)
    merged = torch.cat([merged[:, :1] + unsup_mass.unsqueeze(1), merged[:, 1:]], dim=1)
    return merged


def marginal_loss(logits: torch.Tensor, target: torch.Tensor, supervised: torch.Tensor) -> torch.Tensor:
    """Dice + cross-entropy on the merged distribution, averaged over the batch.

    Unsupervised channels enter only through their summed mass, so their gradient is zero
    (dL/dp_c = 0 for c not in S): verified in tests/test_losses_torch.py.
    """
    probs = torch.softmax(logits.float(), dim=1)
    b, c = probs.shape[:2]
    sup = supervised.clone().bool()
    sup[:, 0] = True
    # A label inside an unsupervised channel cannot be trusted: treat it as background.
    target = target.long()
    label_sup = torch.gather(sup, 1, target.view(b, -1)).view_as(target)
    target = torch.where(label_sup, target, torch.zeros_like(target))
    merged = merge_unsupervised(probs, sup).view(b, c, -1)
    g = F.one_hot(target.view(b, -1), c).permute(0, 2, 1).to(merged.dtype)
    inter = (merged * g).sum(-1)
    dice = (2 * inter + EPS_DICE) / (merged.sum(-1) + g.sum(-1) + EPS_DICE)  # (B, C)
    supf = sup.to(dice.dtype)
    dice_mean = (dice * supf).sum(1) / supf.sum(1)
    picked = torch.gather(merged, 1, target.view(b, 1, -1)).squeeze(1)
    ce = -torch.log(picked.clamp_min(EPS_LOG)).mean(-1)
    return (1 - dice_mean + ce).mean()


def exclusion_loss(logits: torch.Tensor, activation: str = "softmax") -> torch.Tensor:
    """Penalise foreground mass above 1 (Eq. 4.11).

    Under a softmax head the foreground mass is at most 1 by construction, so this
    term is identically zero: the softmax already enforces exclusivity. It becomes
    active for heads with independent sigmoids (`activation="sigmoid"`).
    """
    x = logits.float()
    probs = torch.softmax(x, dim=1) if activation == "softmax" else torch.sigmoid(x)
    return F.relu(probs[:, 1:].sum(1) - 1).mean()


def cox_loss(log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood with Breslow ties and a within-batch risk set."""
    eta = log_risk.float().view(-1)
    t = time.float().view(-1)
    d = event.bool().view(-1)
    if not bool(d.any()):
        return eta.sum() * 0.0  # fully censored batch: legitimate, contributes no gradient
    at_risk = t.unsqueeze(0) >= t.unsqueeze(1)
    masked = eta.unsqueeze(0).expand(len(eta), -1).masked_fill(~at_risk, float("-inf"))
    lse = torch.logsumexp(masked, dim=1)
    return -(eta[d] - lse[d]).mean()


def horizon_loss(logits: torch.Tensor, time: torch.Tensor, event: torch.Tensor, horizons: Sequence[float]) -> torch.Tensor:
    tau = torch.as_tensor(horizons, dtype=torch.float32, device=logits.device).view(1, -1)
    t = time.float().view(-1, 1)
    d = event.bool().view(-1, 1)
    alive = t > tau
    determined = alive | (d & (t <= tau))
    if not bool(determined.any()):
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits.float(), alive.float(), reduction="none")
    return bce[determined].mean()


class UncertaintyWeighting(nn.Module):
    """L = sum_t exp(-s_t) L_t + s_t with one learned log-variance per task (Eq. 4.14)."""

    def __init__(self, tasks: Sequence[str]) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.log_vars = nn.Parameter(torch.zeros(len(self.tasks)))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = self.log_vars.new_zeros(())
        for i, name in enumerate(self.tasks):
            if name in losses:
                total = total + torch.exp(-self.log_vars[i]) * losses[name] + self.log_vars[i]
        return total


class CompositeObjective(nn.Module):
    TASKS = ("marginal", "exclusion", "cox", "horizon")

    def __init__(self, horizons_days: Sequence[float]) -> None:
        super().__init__()
        self.horizons = list(horizons_days)
        self.weighting = UncertaintyWeighting(self.TASKS)

    def forward(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        terms: Dict[str, torch.Tensor] = {
            "marginal": marginal_loss(out["seg_logits"], batch["label"], batch["supervised"]),
            "exclusion": exclusion_loss(out["seg_logits"]),
        }
        has_surv = batch["has_survival"].bool()
        if int(has_surv.sum()) >= 2 and bool(batch["event"][has_surv].bool().any()):
            terms["cox"] = cox_loss(out["log_risk"][has_surv], batch["time"][has_surv], batch["event"][has_surv])
            terms["horizon"] = horizon_loss(
                out["horizon_logits"][has_surv], batch["time"][has_surv], batch["event"][has_surv], self.horizons
            )
        # Cox term omitted (contributes zero) when the batch has no observed event (report, 4.9).
        total = self.weighting(terms)
        terms["total"] = total
        return terms
