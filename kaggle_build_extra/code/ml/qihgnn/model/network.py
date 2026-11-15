"""QIHGNN-MSCA: the full eight-stage network."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from ..config import ModelConfig
from .stages import (
    ConvEncoder,
    HypergraphModule,
    MPOChannelMixer,
    MultiScaleCrossAttention,
    PrognosisHead,
    SegmentationHead,
    SharedRepresentation,
)


class QIHGNNMSCA(nn.Module):
    """Input (B, 4, 128, 128, 64) -> segmentation logits (B, 6, 128, 128, 64) + prognosis outputs.

    Setting `use_mpo`, `use_hypergraph` or `use_msca` to False in the config
    replaces that stage with identity. With all three off the network is the
    matched plain residual U-Net baseline used in the evaluation protocol.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.encoder = ConvEncoder(c)
        self.mpo = MPOChannelMixer(c.mpo_in_factors, c.mpo_out_factors, c.mpo_bond) if c.use_mpo else None
        self.hypergraph = HypergraphModule(c) if c.use_hypergraph else None
        self.msca = MultiScaleCrossAttention(c) if c.use_msca else None
        self.shared = SharedRepresentation(c)
        self.segmentation_head = SegmentationHead(c)
        self.prognosis_head = PrognosisHead(c)

    def stage_parameters(self) -> Dict[str, int]:
        def count(m: Optional[nn.Module]) -> int:
            return 0 if m is None else sum(p.numel() for p in m.parameters())

        return {
            "encoder": count(self.encoder),
            "mpo": count(self.mpo),
            "hypergraph": count(self.hypergraph),
            "msca": count(self.msca),
            "shared": count(self.shared),
            "segmentation_head": count(self.segmentation_head),
            "prognosis_head": count(self.prognosis_head),
        }

    def forward(self, x: torch.Tensor, covariates: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if covariates is None:
            covariates = x.new_zeros(x.shape[0], self.cfg.num_covariates)
        feats, skips = self.encoder(x)
        if self.mpo is not None:
            feats = self.mpo(feats)
        incidence = None
        if self.hypergraph is not None:
            feats, incidence = self.hypergraph(feats)
        if self.msca is not None:
            feats = self.msca(feats)
        shared = self.shared(feats)
        out = {"seg_logits": self.segmentation_head(feats, shared, skips), "shared": shared}
        out.update(self.prognosis_head(shared, covariates))
        if incidence is not None:
            out["incidence"] = incidence
        return out
