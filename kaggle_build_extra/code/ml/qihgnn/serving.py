"""Inference with honest supervision accounting.

A checkpoint records which channels received gradient during training. The
serving layer zeroes every other channel before it returns anything, so the
system cannot report an organ it never learned (report, 4.2 and 5.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .config import LABELS


def suppress(probs: np.ndarray, unsupervised: Sequence[str]) -> np.ndarray:
    """probs (C, ...) -> copy with unsupervised channels set to 0. Background is never suppressed."""
    out = np.array(probs, dtype=np.float32, copy=True)
    for name in unsupervised:
        idx = LABELS.index(name)
        if idx == 0:
            raise ValueError("background cannot be suppressed")
        out[idx] = 0.0
    return out


def label_map(probs: np.ndarray, unsupervised: Sequence[str]) -> np.ndarray:
    return suppress(probs, unsupervised).argmax(axis=0).astype(np.uint8)


class Predictor:
    """Loads a training checkpoint and runs the network (requires torch)."""

    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        import torch

        from .config import ModelConfig
        from .model import QIHGNNMSCA

        self.torch = torch
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ckpt["model_config"].items()})
        self.unsupervised = list(ckpt["unsupervised_channels"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QIHGNNMSCA(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.meta = {k: ckpt.get(k) for k in ("epoch", "run_id", "dataset", "unsupervised_channels")}

    def predict(self, image: np.ndarray, covariates: Sequence[float] | None = None) -> Dict:
        torch = self.torch
        x = torch.as_tensor(image[None], dtype=torch.float32, device=self.device)
        cov = torch.as_tensor([list(covariates or [0.0] * self.cfg.num_covariates)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(x, cov)
        probs = torch.softmax(out["seg_logits"].float(), 1)[0].cpu().numpy()
        horizon = torch.sigmoid(out["horizon_logits"].float())[0].cpu().numpy()
        return {
            "label": label_map(probs, self.unsupervised),
            "suppressed": self.unsupervised,
            "log_risk": float(out["log_risk"][0]),
            "survival_probability": {"1y": float(horizon[0]), "3y": float(horizon[1]), "5y": float(horizon[2])},
            "aleatoric_sd": float(np.exp(0.5 * float(out["log_variance"][0]))),
        }
