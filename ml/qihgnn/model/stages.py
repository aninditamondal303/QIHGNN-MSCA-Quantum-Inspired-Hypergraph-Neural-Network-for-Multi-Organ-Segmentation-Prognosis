"""The eight stages of QIHGNN-MSCA (report, Chapter 2 and Sections 4.4-4.7).

Tensor layout follows the report: (B, C, H, W, D) with H = W = 128 and D = 64.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from .blocks import ResidualBlock3d


# --------------------------------------------------------------------------- 1. encoder
class ConvEncoder(nn.Module):
    """Stem (stride 1) + three strided residual blocks: 4 -> 32 -> 64 -> 128 -> 256 channels."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        chs = (cfg.in_channels,) + tuple(cfg.encoder_channels)
        strides = [1] + [2] * (len(chs) - 2)
        self.blocks = nn.ModuleList(
            ResidualBlock3d(a, b, s) for a, b, s in zip(chs[:-1], chs[1:], strides)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        # Last activation is the encoder output; earlier ones feed the decoder.
        return x, skips[:-1]


# --------------------------------------------------------------------------- 2. MPO
class MPOChannelMixer(nn.Module):
    """Matrix-product-operator (tensor-train) factorisation of a dense C_in -> C_out channel map.

    y(b1..bd) = sum_{a1..ad} prod_k G_k[r_{k-1}, a_k, b_k, r_k] x(a1..ad) + bias      (Eq. 4.1)

    The cores are contracted into the full (C_in, C_out) matrix on every forward
    pass. For 256 x 256 this costs 65k multiply-adds, which is negligible next to
    applying the map at every voxel, and gradients reach every core.
    A residual connection and an affine normalisation surround the map (Sec. 4.4).
    """

    def __init__(self, in_factors: Sequence[int], out_factors: Sequence[int], bond: int) -> None:
        super().__init__()
        if len(in_factors) != len(out_factors):
            raise ValueError("in_factors and out_factors need the same number of cores")
        self.in_factors = tuple(in_factors)
        self.out_factors = tuple(out_factors)
        d = len(in_factors)
        ranks = [1] + [bond] * (d - 1) + [1]
        c_in, c_out = math.prod(in_factors), math.prod(out_factors)
        # Variance of each entry of W is prod_k var(G_k) * prod of inner bonds; aim for 1 / C_in.
        core_var = (1.0 / (c_in * bond ** (d - 1))) ** (1.0 / d)
        self.cores = nn.ParameterList(
            nn.Parameter(torch.randn(ranks[k], in_factors[k], out_factors[k], ranks[k + 1]) * math.sqrt(core_var))
            for k in range(d)
        )
        self.bias = nn.Parameter(torch.zeros(c_out))
        # The report names batch normalisation here; we use the affine instance norm used everywhere
        # else in the network for the same batch-size-2 reason. Parameter count is identical (2 * C).
        self.norm = nn.InstanceNorm3d(c_out, affine=True)
        self.c_in, self.c_out = c_in, c_out

    def dense_matrix(self) -> torch.Tensor:
        w = self.cores[0].squeeze(0)  # (i1, o1, r1)
        for core in self.cores[1:]:
            w = torch.einsum("...r,rios->...ios", w, core)
        w = w.squeeze(-1)  # (i1, o1, i2, o2, ..., id, od)
        d = len(self.in_factors)
        perm = [2 * k for k in range(d)] + [2 * k + 1 for k in range(d)]
        return w.permute(*perm).reshape(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dense_matrix().to(x.dtype)
        y = torch.einsum("bc...,co->bo...", x, w) + self.bias.to(x.dtype).view(1, -1, 1, 1, 1)
        return x + self.norm(y)


# --------------------------------------------------------------------------- 3. hypergraph
class HypergraphConv(nn.Module):
    """One hypergraph propagation step (Eq. 4.4).

    Z' = GELU( LN( Dv^-1/2 H W De^-1 H^T Dv^-1/2 Z Theta + Z ) )
    """

    def __init__(self, channels: int, hyperedges: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.theta = nn.Linear(channels, channels)
        # Learned diagonal hyperedge weighting W_e; softplus keeps it positive. Initialised to 1.
        self.edge_weight_raw = nn.Parameter(torch.full((hyperedges,), math.log(math.e - 1.0)))
        self.norm = nn.LayerNorm(channels)
        self.eps = eps

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # z: (B, N, C) node features; h: (B, N, E) soft incidence in [0, 1].
        w = F.softplus(self.edge_weight_raw)  # (E,)
        dv = (h * w).sum(-1)  # vertex degree (B, N)
        de = h.sum(1)  # hyperedge degree (B, E)
        dv_is = (dv + self.eps).rsqrt().unsqueeze(-1)  # (B, N, 1)
        de_inv = (w / (de + self.eps)).unsqueeze(-1)  # W_e De^-1, (B, E, 1)
        m = dv_is * self.theta(z)  # Dv^-1/2 Z Theta
        e = torch.bmm(h.transpose(1, 2), m) * de_inv  # nodes -> hyperedges
        n = dv_is * torch.bmm(h, e)  # hyperedges -> nodes
        return F.gelu(self.norm(n + z))


class HypergraphModule(nn.Module):
    """Pool to N = 256 region nodes, learn a soft node-to-organ incidence, propagate 3 times."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.encoder_channels[-1]
        self.node_grid = cfg.node_grid
        self.incidence = nn.Sequential(
            nn.Linear(c, cfg.incidence_hidden), nn.GELU(), nn.Linear(cfg.incidence_hidden, cfg.hyperedges)
        )
        self.layers = nn.ModuleList(HypergraphConv(c, cfg.hyperedges) for _ in range(cfg.hypergraph_layers))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c = x.shape[:2]
        grid = F.adaptive_avg_pool3d(x, self.node_grid)  # (B, C, 8, 8, 4)
        z = grid.flatten(2).transpose(1, 2)  # (B, N, C)
        h = z.new_zeros(b, z.shape[1], self.layers[0].edge_weight_raw.numel())
        for layer in self.layers:
            h = torch.sigmoid(self.incidence(z))  # incidence re-estimated from the current node features
            z = layer(z, h)
        z_grid = z.transpose(1, 2).reshape(b, c, *self.node_grid)
        up = F.interpolate(z_grid, size=x.shape[2:], mode="trilinear", align_corners=False)
        return x + up, h


# --------------------------------------------------------------------------- 4. multi-scale attention
class MultiScaleCrossAttention(nn.Module):
    """Global attention over fixed token budgets 256 / 32 / 4 / 1 (Eqs. 4.5-4.7).

    Self-attention runs within each scale; then, from coarse to fine, every scale
    attends to the next coarser one so global context conditions local detail.
    Cost is sum(n_s^2) = 66,577 entries per head, independent of input size.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.encoder_channels[-1]
        self.grids = tuple(tuple(g) for g in cfg.token_grids)
        s = len(self.grids)
        self.self_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(s))
        self.self_attn = nn.ModuleList(nn.MultiheadAttention(c, cfg.attention_heads, batch_first=True) for _ in range(s))
        self.cross_norm = nn.ModuleList(nn.LayerNorm(c) for _ in range(s - 1))
        self.cross_attn = nn.ModuleList(
            nn.MultiheadAttention(c, cfg.attention_heads, batch_first=True) for _ in range(s - 1)
        )
        self.fuse = nn.Conv3d(s * c, c, 1)

    def token_budgets(self) -> List[int]:
        return [math.prod(g) for g in self.grids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        tokens = [F.adaptive_avg_pool3d(x, g).flatten(2).transpose(1, 2) for g in self.grids]
        for i, (norm, attn) in enumerate(zip(self.self_norm, self.self_attn)):
            q = norm(tokens[i])
            tokens[i] = tokens[i] + attn(q, q, q, need_weights=False)[0]
        for i in reversed(range(len(self.grids) - 1)):  # organ <- modality, region <- organ, voxel <- region
            q = self.cross_norm[i](tokens[i])
            kv = tokens[i + 1]
            tokens[i] = tokens[i] + self.cross_attn[i](q, kv, kv, need_weights=False)[0]
        maps = [
            F.interpolate(t.transpose(1, 2).reshape(b, c, *g), size=x.shape[2:], mode="trilinear", align_corners=False)
            for t, g in zip(tokens, self.grids)
        ]
        return x + self.fuse(torch.cat(maps, dim=1))


# --------------------------------------------------------------------------- 5. shared representation
class SharedRepresentation(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.encoder_channels[-1], cfg.shared_dim)
        self.norm = nn.LayerNorm(cfg.shared_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x.mean(dim=(2, 3, 4))))


# --------------------------------------------------------------------------- 6. segmentation head
class SegmentationHead(nn.Module):
    """Transposed-convolution decoder with skip connections, ending in K + 1 = 6 channels.

    The 512-d shared vector conditions the decoder through a parameter-free FiLM:
    its two halves are used as a per-channel scale and shift on the 256-channel
    input, so both heads read the same representation (report, 2.6-2.7) without
    adding parameters to the budget.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        chs = list(cfg.encoder_channels)[::-1]  # 256, 128, 64, 32
        if cfg.shared_dim != 2 * chs[0]:
            raise ValueError("parameter-free FiLM needs shared_dim == 2 * bottleneck channels")
        self.ups = nn.ModuleList(nn.ConvTranspose3d(a, b, 2, stride=2) for a, b in zip(chs[:-1], chs[1:]))
        self.blocks = nn.ModuleList(ResidualBlock3d(2 * b, b) for b in chs[1:])
        self.out = nn.Conv3d(chs[-1], cfg.num_classes, 1)

    def forward(self, x: torch.Tensor, shared: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        gamma, beta = shared.chunk(2, dim=-1)
        x = x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        for up, block, skip in zip(self.ups, self.blocks, reversed(skips)):
            x = up(x)
            x = block(torch.cat([x, skip], dim=1))
        return self.out(x)


# --------------------------------------------------------------------------- 7. prognosis head
class PrognosisHead(nn.Module):
    """MLP over [shared (512), covariates (5)] -> log-risk, 1/3/5-year logits, log-variance."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h1, h2 = cfg.prognosis_hidden
        self.net = nn.Sequential(
            nn.Linear(cfg.shared_dim + cfg.num_covariates, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, cfg.prognosis_outputs),
        )

    def forward(self, shared: torch.Tensor, covariates: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(torch.cat([shared, covariates.to(shared.dtype)], dim=-1))
        # Raw logits are kept: the Cox term needs an unbounded log-risk (Sec. 4.7).
        return {
            "log_risk": out[:, 0],
            "horizon_logits": out[:, 1:4],
            "log_variance": out[:, 4],
        }
