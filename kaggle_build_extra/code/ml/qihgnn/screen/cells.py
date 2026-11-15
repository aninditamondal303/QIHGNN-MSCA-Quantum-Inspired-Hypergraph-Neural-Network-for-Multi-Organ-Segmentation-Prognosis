"""Cervical cell grading: an image classifier trained on SIPaKMeD, where cytologists graded every cell.

  python -m qihgnn.screen.cells --sipakmed <dir> --out <dir> [--probe]

Five classes as graded by the experts, and the three groups the website reports:
  normal    superficial-intermediate, parabasal
  benign    metaplastic
  abnormal  koilocytotic, dyskeratotic
Cells are split by the slide image they were cut from, so no test cell shares an image with a training cell.
The browser crops each cell the cell-outline model finds and runs this classifier on the crop.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .common import binary_auc, binary_report, confusion, ece, group_split, wilson, write_json

T0 = time.time()
CLASSES = ("superficial_intermediate", "parabasal", "metaplastic", "koilocytotic", "dyskeratotic")
GROUP = {"superficial_intermediate": "normal", "parabasal": "normal", "metaplastic": "benign", "koilocytotic": "abnormal", "dyskeratotic": "abnormal"}
SIZE = 128
FOLDER = {"superficial": "superficial_intermediate", "parabasal": "parabasal", "metaplastic": "metaplastic",
          "koilocytotic": "koilocytotic", "dyskeratotic": "dyskeratotic"}


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


def index(root: Path) -> List[Tuple[Path, int, str]]:
    """(cropped cell image, class index, group key = class + source image) for every graded cell."""
    out = []
    for p in sorted(Path(root).rglob("*.bmp")):
        if p.parent.name.upper() != "CROPPED":
            continue
        folder = p.parent.parent.name.lower()
        cls = next((v for k, v in FOLDER.items() if k in folder), None)
        if cls is None:
            continue
        m = re.match(r"(\d+)_", p.name)
        out.append((p, CLASSES.index(cls), f"{cls}:{m.group(1) if m else p.stem}"))
    return out


def square(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Pad to a square with the image's border colour, then resize: the same as cellCrop in the browser."""
    from PIL import Image

    h, w = img.shape[:2]
    s = max(h, w)
    edge = np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]), axis=0).astype(np.uint8)
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = edge
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BILINEAR))


def load(items) -> np.ndarray:
    from PIL import Image

    return np.stack([square(np.asarray(Image.open(p).convert("RGB"))) for p, _, _ in items])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sipakmed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--epochs", type=int, default=24)
    a = ap.parse_args()
    import torch
    import torchvision
    from torch import nn

    from .train import export_onnx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = index(Path(a.sipakmed))
    if a.probe:
        items = items[::40]
    split = {g: group_split(g) for _, _, g in items}
    parts = {k: [it for it in items if split[it[2]] == k] for k in ("train", "val", "test")}
    log("cells", {k: len(v) for k, v in parts.items()}, "per class", np.bincount([c for _, c, _ in items], minlength=5).tolist())
    X = {k: load(v) for k, v in parts.items()}
    Y = {k: np.array([c for _, c, _ in v]) for k, v in parts.items()}

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def tensor(x):
        return (torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255 - mean) / std

    try:
        net = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception as exc:
        print("ImageNet weights unavailable:", exc)
        net = torchvision.models.efficientnet_b0(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, len(CLASSES))
    net = net.to(device)
    counts = np.bincount(Y["train"], minlength=5).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / 5, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 1 if a.probe else a.epochs
    steps = int(np.ceil(len(X["train"]) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps, pct_start=0.15)
    rng = np.random.default_rng(0)

    def augment(x):
        # cells have no up or side: any flip or quarter turn; mild colour and stain variation
        x = np.rot90(x, int(rng.integers(4)), axes=(1, 2))
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
        x = x.astype(np.float32) * rng.uniform(0.85, 1.15, (len(x), 1, 1, 3)) + rng.uniform(-15, 15, (len(x), 1, 1, 3))
        return np.clip(x, 0, 255).astype(np.uint8)

    def logits(x, bs=128):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), bs):
                outs.append(net(tensor(np.ascontiguousarray(x[i:i + bs])).to(device)).float().cpu())
        return torch.cat(outs).numpy()

    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(X["train"]))
        losses = []
        for i in range(0, len(order), 64):
            b = order[i:i + 64]
            x = tensor(np.ascontiguousarray(augment(X["train"][b]))).to(device)
            y = torch.from_numpy(Y["train"][b]).to(device)
            loss = nn.functional.cross_entropy(net(x), y, weight=weights, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        acc = float((logits(X["val"]).argmax(1) == Y["val"]).mean())
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_accuracy": acc})
        log(f"cells epoch {ep}: loss {np.mean(losses):.4f} val acc {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    # temperature on validation, then everything measured once on the test cells
    lv = torch.from_numpy(logits(X["val"]))
    yv = torch.from_numpy(Y["val"])
    ts = np.linspace(0.5, 3, 26)
    temp = float(ts[int(np.argmin([float(nn.functional.cross_entropy(lv / t, yv)) for t in ts]))])
    pt = torch.softmax(torch.from_numpy(logits(X["test"])) / temp, 1).numpy()
    yt = Y["test"]
    pred = pt.argmax(1)
    ab = [CLASSES.index("koilocytotic"), CLASSES.index("dyskeratotic")]
    p_ab = pt[:, ab].sum(1)
    y_ab = np.isin(yt, ab).astype(int)
    pv = torch.softmax(lv / temp, 1).numpy()
    yv_ab = np.isin(Y["val"], ab)
    pv_ab = pv[:, ab].sum(1)
    # flag threshold: highest one that still finds 90% of abnormal validation cells
    cands = np.linspace(0.05, 0.95, 91)
    flag = float(max([t for t in cands if (pv_ab[yv_ab] >= t).mean() >= 0.9] or [0.5]))
    rep = binary_report(p_ab.tolist(), y_ab.tolist(), threshold=flag)
    rep["auc"] = binary_auc(p_ab.tolist(), y_ab.tolist())
    acc = float((pred == yt).mean())
    M = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "probe": a.probe,
        "data": {"source": "SIPaKMeD (cervical cells graded by cytologists)", "cells": len(items),
                 "splits": {k: len(v) for k, v in parts.items()}, "per_class": np.bincount([c for _, c, _ in items], minlength=5).tolist()},
        "classes": list(CLASSES), "groups": GROUP, "input": SIZE, "temperature": temp, "history": hist,
        "test": {"n": int(len(yt)), "accuracy": acc, "accuracy_ci": list(wilson(int((pred == yt).sum()), int(len(yt)))),
                 "confusion": confusion(yt.tolist(), pred.tolist(), len(CLASSES)).tolist(),
                 "recall": {c: float((pred[yt == i] == i).mean()) if (yt == i).any() else None for i, c in enumerate(CLASSES)},
                 "ece": ece(pt, yt), "abnormal": rep},
        "flag_threshold": flag,
        "note": "Accuracy is measured on SIPaKMeD cells. Slides from other laboratories (stain, magnification) can score differently.",
    }
    log("test", {"accuracy": acc, "abnormal sens/spec/auc": (rep["sensitivity"], rep["specificity"], rep["auc"]), "flag": flag})

    class Export(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.register_buffer("inv_t", torch.tensor(1.0 / temp))

        def forward(self, rgb):  # (B, 3, 128, 128) in [0, 1]
            return torch.softmax(self.m((rgb - self.mean) / self.std) * self.inv_t, 1)

    M["export"] = export_onnx(Export(net.cpu().eval()), torch.rand(2, 3, SIZE, SIZE), out / "cells.onnx", "rgb", ["probs"])
    write_json(out / "cells_metrics.json", M)
    # parity fixture for the browser crop
    from PIL import Image

    p0 = parts["test"][0][0] if parts["test"] else items[0][0]
    raw = np.asarray(Image.open(p0).convert("RGB"))
    Image.fromarray(raw).save(out / "cell_sample.png")
    write_json(out / "cell_sample.json", {"file": "cell_sample.png", "truth": CLASSES[(parts["test"] or items)[0][1]],
                                          "crop_mean": float(square(raw).mean())})
    log("done")


if __name__ == "__main__":
    main()
