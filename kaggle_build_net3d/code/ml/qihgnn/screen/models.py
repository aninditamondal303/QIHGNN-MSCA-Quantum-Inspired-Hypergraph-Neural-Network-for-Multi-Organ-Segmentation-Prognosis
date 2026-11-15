"""Screening networks (torch). Kept small enough to run in a browser tab through ONNX Runtime Web."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_classifier(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 (4.0 M parameters with this 5-way head). ImageNet weights when available."""
    import torchvision

    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        m = torchvision.models.efficientnet_b0(weights=weights)
    except Exception as exc:  # no internet: train from scratch rather than fail
        print("ImageNet weights unavailable, training from scratch:", exc)
        m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


class ClassifierExport(nn.Module):
    """Browser graph: grey image in [0, 1] (B, 1, 224, 224) -> calibrated probabilities and class activation maps.

    Normalisation and the calibration temperature live inside the graph, so the browser only resizes pixels.
    The maps are W_k . f(x, y) over the last 7x7 feature grid: where the evidence for class k sits.
    """

    def __init__(self, net: nn.Module, temperature: float):
        super().__init__()
        self.features, self.pool, self.fc = net.features, net.avgpool, net.classifier[1]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))

    def forward(self, gray: torch.Tensor):
        x = (gray.repeat(1, 3, 1, 1) - self.mean) / self.std
        f = self.features(x)
        logits = self.fc(torch.flatten(self.pool(f), 1))
        probs = torch.softmax(logits * self.inv_t, dim=1)
        cams = torch.einsum("bchw,kc->bkhw", f, self.fc.weight)
        return probs, cams


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    """Plain 2D U-Net. Defaults: 1 grey channel in, one tumour logit out; base=16 gives ~1.9 M parameters."""

    def __init__(self, base: int = 16, depth: int = 4, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_ch, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


class SegmenterExport(nn.Module):
    """Browser graph: percentile-normalised grey slice (B, 1, 192, 192) -> tumour probability map."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class SoftmaxExport(nn.Module):
    """Browser graph for multi-class segmenters: image in [0, 1] -> per-pixel class probabilities."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)
