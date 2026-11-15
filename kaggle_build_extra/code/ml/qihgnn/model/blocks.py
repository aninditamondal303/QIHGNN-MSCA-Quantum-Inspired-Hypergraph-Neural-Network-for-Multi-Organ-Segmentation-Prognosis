"""Conventional building blocks: the residual block used by the encoder and decoder."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock3d(nn.Module):
    """conv3x3 -> IN -> LeakyReLU -> conv3x3 -> IN, plus a 1x1x1 projection shortcut.

    Instance normalisation instead of batch normalisation: the batch size is 2,
    and statistics estimated over two volumes are unreliable (report, 4.4).
    Convolutions carry no bias because the affine norm that follows has one.
    """

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(cout, affine=True)
        self.conv2 = nn.Conv3d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(cout, affine=True)
        self.shortcut = nn.Conv3d(cin, cout, 1, stride=stride, bias=False)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))
