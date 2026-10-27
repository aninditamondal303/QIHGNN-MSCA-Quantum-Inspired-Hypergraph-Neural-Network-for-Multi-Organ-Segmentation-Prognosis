"""Single source of truth for model and training hyper-parameters.

Every number here is taken from the project report (Chapters 2 and 4). The
pure-Python parameter counter (`param_count.py`) and the PyTorch network
(`model/network.py`) both read this file, so the two cannot silently drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

# Unified label space: background + K = 5 organ channels.
# Index order is fixed: it is written into checkpoints and read by the server.
LABELS: Tuple[str, ...] = ("background", "brain", "liver", "kidney", "cervix", "spleen")
LABEL_DESCRIPTIONS = {
    "background": "Everything that is not one of the five organ channels.",
    "brain": "BraTS whole-tumour region (labels 1, 2 and 4 merged).",
    "liver": "Parenchyma including lesions (LiTS labels 1 and 2 merged).",
    "kidney": "Supervised only when KiTS19 is registered.",
    "cervix": "Supervised only by a volumetric dataset with cervix masks.",
    "spleen": "No registered dataset supervises it.",
}
NUM_CLASSES = len(LABELS)  # K + 1 = 6


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 4
    input_shape: Tuple[int, int, int] = (128, 128, 64)
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    # Matrix product operator: 256 = 16 x 16 in and out, two cores, bond r.
    mpo_in_factors: Tuple[int, ...] = (16, 16)
    mpo_out_factors: Tuple[int, ...] = (16, 16)
    mpo_bond: int = 32
    # Hypergraph: N = 8*8*4 = 256 region nodes, E = 5 organ hyperedges.
    node_grid: Tuple[int, int, int] = (8, 8, 4)
    hyperedges: int = 5
    hypergraph_layers: int = 3
    incidence_hidden: int = 128
    # Multi-scale cross-attention token grids (voxel, region, organ, modality).
    token_grids: Tuple[Tuple[int, int, int], ...] = ((8, 8, 4), (4, 4, 2), (2, 2, 1), (1, 1, 1))
    attention_heads: int = 8
    shared_dim: int = 512
    num_covariates: int = 5
    num_classes: int = NUM_CLASSES
    prognosis_hidden: Tuple[int, int] = (256, 128)
    prognosis_outputs: int = 5  # log-risk, 3 horizon logits, log-variance
    # Ablation switches (P6): each replaces a proposed stage with identity.
    use_mpo: bool = True
    use_hypergraph: bool = True
    use_msca: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 2
    grad_accumulation: int = 8
    peak_lr: float = 3e-4
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 2026
    horizons_years: Tuple[float, ...] = (1.0, 3.0, 5.0)
    min_events_per_batch: int = 2
    num_workers: int = 2
    cache_dir: str = "data/cache"
    run_dir: str = "runs"
    dataset: str = "cache"  # "cache" (preprocessed corpora) or "phantom" (synthetic smoke test)
    phantom_cases: int = 12
    max_steps_per_epoch: int = 0  # 0 = full epoch
    extra: dict = field(default_factory=dict)


# Parameter counts per stage as printed in the project report (Table 5.7).
REPORT_PARAMETER_BUDGET = {
    "encoder": 3_559_808,
    "mpo": 17_152,
    "hypergraph": 232_468,
    "msca": 2_108_160,
    "shared": 132_608,
    "segmentation_head": 2_130_214,
    "prognosis_head": 166_661,
}
REPORT_TOTAL_PARAMETERS = 8_347_071
