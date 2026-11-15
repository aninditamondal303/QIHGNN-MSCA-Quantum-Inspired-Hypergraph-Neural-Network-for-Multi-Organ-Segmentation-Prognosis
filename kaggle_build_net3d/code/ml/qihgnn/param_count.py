"""Analytical parameter counter written in pure Python.

It mirrors the layer definitions in `qihgnn.model` without importing PyTorch,
so the parameter budget can be checked on any machine and the check does not
run through the code path it is meant to test. A drift test
(`tests/test_model_torch.py`) fails if this counter and `sum(p.numel())` of
the real network ever disagree.
"""
from __future__ import annotations

from math import prod
from typing import Dict

from .config import ModelConfig


def conv3d(cin: int, cout: int, k: int, bias: bool) -> int:
    return cin * cout * k ** 3 + (cout if bias else 0)


def conv_transpose3d(cin: int, cout: int, k: int, bias: bool) -> int:
    return cin * cout * k ** 3 + (cout if bias else 0)


def linear(fin: int, fout: int, bias: bool = True) -> int:
    return fin * fout + (fout if bias else 0)


def norm_affine(c: int) -> int:
    """InstanceNorm / LayerNorm / BatchNorm with a learned scale and shift."""
    return 2 * c


def residual_block(cin: int, cout: int) -> int:
    """Two 3x3x3 convolutions (no bias) with affine instance norm, plus a 1x1x1 projection shortcut."""
    return (
        conv3d(cin, cout, 3, False) + norm_affine(cout)
        + conv3d(cout, cout, 3, False) + norm_affine(cout)
        + conv3d(cin, cout, 1, False)
    )


def multihead_attention(dim: int) -> int:
    # nn.MultiheadAttention: packed q/k/v projection with bias + output projection with bias.
    return 3 * dim * dim + 3 * dim + linear(dim, dim)


def encoder(cfg: ModelConfig) -> int:
    chs = (cfg.in_channels,) + tuple(cfg.encoder_channels)
    return sum(residual_block(a, b) for a, b in zip(chs[:-1], chs[1:]))


def mpo_cores(in_factors, out_factors, bond: int) -> int:
    d = len(in_factors)
    ranks = [1] + [bond] * (d - 1) + [1]
    return sum(ranks[k] * in_factors[k] * out_factors[k] * ranks[k + 1] for k in range(d))


def mpo(cfg: ModelConfig) -> int:
    if not cfg.use_mpo:
        return 0
    c_out = prod(cfg.mpo_out_factors)
    return mpo_cores(cfg.mpo_in_factors, cfg.mpo_out_factors, cfg.mpo_bond) + c_out + norm_affine(c_out)


def hypergraph(cfg: ModelConfig) -> int:
    if not cfg.use_hypergraph:
        return 0
    c = cfg.encoder_channels[-1]
    incidence = linear(c, cfg.incidence_hidden) + linear(cfg.incidence_hidden, cfg.hyperedges)
    per_layer = linear(c, c) + norm_affine(c) + cfg.hyperedges  # Theta, LayerNorm, hyperedge weights W_e
    return incidence + cfg.hypergraph_layers * per_layer


def msca(cfg: ModelConfig) -> int:
    if not cfg.use_msca:
        return 0
    c = cfg.encoder_channels[-1]
    s = len(cfg.token_grids)
    self_attn = s * (multihead_attention(c) + norm_affine(c))
    cross_attn = (s - 1) * (multihead_attention(c) + norm_affine(c))
    fusion = conv3d(s * c, c, 1, True)
    return self_attn + cross_attn + fusion


def shared(cfg: ModelConfig) -> int:
    return linear(cfg.encoder_channels[-1], cfg.shared_dim) + norm_affine(cfg.shared_dim)


def segmentation_head(cfg: ModelConfig) -> int:
    chs = list(cfg.encoder_channels)[::-1]  # 256, 128, 64, 32
    total = 0
    for cin, cout in zip(chs[:-1], chs[1:]):
        total += conv_transpose3d(cin, cout, 2, True)
        total += residual_block(2 * cout, cout)  # after concatenating the skip connection
    total += conv3d(chs[-1], cfg.num_classes, 1, True)
    return total


def prognosis_head(cfg: ModelConfig) -> int:
    h1, h2 = cfg.prognosis_hidden
    return (
        linear(cfg.shared_dim + cfg.num_covariates, h1) + norm_affine(h1)
        + linear(h1, h2)
        + linear(h2, cfg.prognosis_outputs)
    )


def parameter_budget(cfg: ModelConfig | None = None) -> Dict[str, int]:
    cfg = cfg or ModelConfig()
    return {
        "encoder": encoder(cfg),
        "mpo": mpo(cfg),
        "hypergraph": hypergraph(cfg),
        "msca": msca(cfg),
        "shared": shared(cfg),
        "segmentation_head": segmentation_head(cfg),
        "prognosis_head": prognosis_head(cfg),
    }


def total_parameters(cfg: ModelConfig | None = None) -> int:
    return sum(parameter_budget(cfg).values())


def mpo_compression_table(bonds=(2, 4, 8, 16, 32, 48, 64, 128), in_factors=(16, 16), out_factors=(16, 16)):
    dense = prod(in_factors) * prod(out_factors)
    rows = []
    for r in bonds:
        p = mpo_cores(in_factors, out_factors, r)
        rows.append({"bond": r, "parameters": p, "dense": dense, "compression": dense / p})
    return rows


def attention_cost(token_budgets=(256, 32, 4, 1), shape=(128, 128, 64)) -> dict:
    voxels = prod(shape)
    multi = sum(n * n for n in token_budgets)
    naive = voxels * voxels
    return {"voxels": voxels, "multi_scale": multi, "naive": naive, "reduction": naive / multi}


if __name__ == "__main__":
    budget = parameter_budget()
    total = sum(budget.values())
    for k, v in budget.items():
        print(f"{k:<20}{v:>12,}  {100 * v / total:5.2f}%")
    print(f"{'total':<20}{total:>12,}  ({total * 4 / 2 ** 20:.2f} MB fp32)")
