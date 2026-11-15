"""Synthetic anatomical phantoms.

They exist so that the website, the API and the smoke-training run work before any
real data is downloaded, and so that CI can run the whole pipeline without data.
Every phantom is labelled SYNTHETIC wherever it appears. They are not used to
report any segmentation or prognosis performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import ndimage

from .config import LABELS
from .data.preprocess import CT_WINDOWS, ct_to_channels, supervision_vector

SHAPE = (128, 128, 64)


@dataclass
class Phantom:
    case_id: str
    title: str
    modality: str  # "MRI" or "CT"
    analog_of: str  # which corpus this phantom imitates
    channels: List[str]
    image: np.ndarray  # (4, H, W, D) float32 in [0, 1] for display / roughly normalised for the model
    label: np.ndarray  # (H, W, D) uint8, unified label space
    supervised: np.ndarray  # (C,) bool
    note: str = ""
    covariates: List[float] = field(default_factory=lambda: [0.0] * 5)
    survival_days: float | None = None
    event: int | None = None


def _grid(shape=SHAPE):
    """Return x (left-right), y (anterior-posterior), z (inferior-superior) on an (H, W, D) array.

    Array rows (H) run anterior -> posterior and columns (W) run patient-right -> patient-left,
    so an axial slice vol[:, :, k] displays in the radiological convention.
    """
    ys = np.linspace(-1, 1, shape[0], dtype=np.float32)
    xs = np.linspace(-1, 1, shape[1], dtype=np.float32)
    zs = np.linspace(-1, 1, shape[2], dtype=np.float32)
    y, x, z = np.meshgrid(ys, xs, zs, indexing="ij")
    return x, y, z


def _ellipsoid(x, y, z, c, r):
    return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2


def _smooth_noise(rng, shape, sigma):
    n = ndimage.gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma)
    return n / (np.abs(n).max() + 1e-8)


def brain_mri(seed: int = 1) -> Phantom:
    rng = np.random.default_rng(seed)
    x, y, z = _grid()
    head = _ellipsoid(x, y, z, (0, 0.02, 0), (0.70, 0.86, 0.95))
    brain = head < 0.78
    scalp = (head < 1.0) & (head >= 0.9)
    skull = (head < 0.9) & ~brain
    fold = _smooth_noise(rng, SHAPE, 2.2)
    tex = _smooth_noise(rng, SHAPE, 3.5)
    inner = _ellipsoid(x, y, z, (0, 0.02, 0), (0.56, 0.70, 0.78)) + 0.22 * fold
    white = brain & (inner < 1.0)
    sulci = brain & ~white & (np.abs(fold) < 0.08) & (head > 0.45)
    falx = brain & (np.abs(x) < 0.012 + 0.01 * np.abs(fold)) & (y < 0.5)
    vent = (
        _ellipsoid(np.abs(x), y, z, (0.1, -0.05, 0.08), (0.07, 0.30, 0.32)) + 0.15 * tex < 1
    ) & brain
    warp = _smooth_noise(rng, SHAPE, 5) * 0.45
    d_t = _ellipsoid(x, y, z, (-0.3, -0.22, 0.18), (0.19, 0.21, 0.30)) + warp
    core = (d_t < 0.75) & brain
    necrosis = (d_t < 0.28) & brain
    edema = (_ellipsoid(x, y, z, (-0.3, -0.2, 0.18), (0.30, 0.36, 0.45)) + 1.2 * warp < 1.0) & white & ~core
    grey = brain & ~white
    csf = sulci | falx | vent
    bias = 1.0 + 0.08 * x - 0.05 * y  # smooth coil-sensitivity bias field

    def seq(vals: Dict[str, float], noise=0.025):
        v = np.zeros(SHAPE, np.float32)
        v[scalp] = vals["scalp"]
        v[skull] = vals["skull"]
        v[grey] = vals["grey"]
        v[white] = vals["white"]
        v[csf] = vals["csf"]
        v[edema] = vals["edema"]
        v[core] = vals["core"]
        v[necrosis] = vals["necrosis"]
        v = ndimage.gaussian_filter(v, 0.7) * bias
        v += (head < 1.0) * (rng.normal(0, noise, SHAPE).astype(np.float32) + 0.035 * tex)
        return np.clip(v, 0, 1)

    t1 = seq(dict(scalp=0.7, skull=0.08, grey=0.45, white=0.64, csf=0.14, edema=0.40, core=0.38, necrosis=0.16))
    t1ce = seq(dict(scalp=0.72, skull=0.08, grey=0.46, white=0.62, csf=0.14, edema=0.42, core=0.95, necrosis=0.15))
    t2 = seq(dict(scalp=0.55, skull=0.05, grey=0.60, white=0.42, csf=0.95, edema=0.80, core=0.62, necrosis=0.90))
    flair = seq(dict(scalp=0.5, skull=0.05, grey=0.55, white=0.42, csf=0.07, edema=0.92, core=0.70, necrosis=0.25))
    label = np.zeros(SHAPE, np.uint8)
    label[edema | core] = LABELS.index("brain")
    return Phantom(
        case_id="PH-BRAIN-01",
        title="Synthetic brain MRI with enhancing lesion",
        modality="MRI",
        analog_of="BraTS 2020",
        channels=["T1", "T1ce", "T2", "FLAIR"],
        image=np.stack([t1, t1ce, t2, flair]),
        label=label,
        supervised=supervision_vector(["brain"]),
        note="Lesion, oedema and necrotic core are warped ellipsoids inside folded cortex. Only the brain channel is supervised, as in BraTS.",
        covariates=[0.4, 1.0, 0.0, 0.0, 1.0],
        survival_days=410.0,
        event=1,
    )


def _hu_volume(parts, base=-1000.0, noise=12.0, rng=None):
    hu = np.full(SHAPE, base, np.float32)
    for mask, value in parts:
        hu[mask] = value
    hu = ndimage.gaussian_filter(hu, 0.7)
    if rng is not None:
        hu += rng.normal(0, noise, SHAPE).astype(np.float32)
    return hu


def abdomen_ct(seed: int = 2) -> Phantom:
    rng = np.random.default_rng(seed)
    x, y, z = _grid()
    body = _ellipsoid(x, y, z * 0.0, (0, 0.05, 0), (0.92, 0.66, 10)) < 1
    fat = body & (_ellipsoid(x, y, z * 0.0, (0, 0.05, 0), (0.84, 0.58, 10)) >= 1)
    muscle_wall = body & ~fat & (_ellipsoid(x, y, z * 0.0, (0, 0.05, 0), (0.78, 0.52, 10)) >= 1)
    warp = _smooth_noise(rng, SHAPE, 8) * 0.25
    liver = body & (_ellipsoid(x, y, z, (-0.32, -0.12, 0.05), (0.42, 0.34, 0.8)) + warp < 1) & (x < 0.15)
    lesion = liver & (_ellipsoid(x, y, z, (-0.42, -0.2, 0.15), (0.09, 0.09, 0.14)) < 1)
    spleen = body & (_ellipsoid(x, y, z, (0.5, 0.1, 0.1), (0.13, 0.2, 0.42)) + warp * 0.6 < 1)
    kidney_l = body & (_ellipsoid(x, y, z, (0.3, 0.36, -0.15), (0.1, 0.13, 0.38)) < 1)
    kidney_r = body & (_ellipsoid(x, y, z, (-0.3, 0.36, -0.25), (0.1, 0.13, 0.38)) < 1)
    kidney = (kidney_l | kidney_r) & ~liver
    spine = body & (_ellipsoid(x, y, z * 0.0, (0, 0.5, 0), (0.09, 0.1, 10)) < 1)
    aorta = body & (_ellipsoid(x, y, z * 0.0, (0.06, 0.32, 0), (0.045, 0.045, 10)) < 1)
    bowel_noise = _smooth_noise(rng, SHAPE, 4)
    bowel = body & ~liver & ~spleen & ~kidney & ~spine & (_ellipsoid(x, y, z * 0.0, (0.05, -0.15, 0), (0.45, 0.32, 10)) < 1)
    gas = bowel & (bowel_noise > 0.45)
    parts = [
        (body, 35.0), (fat, -95.0), (muscle_wall, 50.0), (bowel, 25.0), (gas, -850.0),
        (liver, 62.0), (lesion, 18.0), (spleen, 48.0), (kidney, 150.0), (spine, 650.0), (aorta, 180.0),
    ]
    hu = _hu_volume(parts, rng=rng)
    label = np.zeros(SHAPE, np.uint8)
    label[kidney] = LABELS.index("kidney")
    label[spleen] = LABELS.index("spleen")
    label[liver] = LABELS.index("liver")
    return Phantom(
        case_id="PH-ABDOMEN-01",
        title="Synthetic abdominal CT, liver with one lesion",
        modality="CT",
        analog_of="LiTS",
        channels=[f"{k.title()} window" for k in CT_WINDOWS],
        image=ct_to_channels(hu),
        label=label,
        supervised=supervision_vector(["liver"]),
        note="Kidneys and spleen are present in the anatomy but this corpus annotates only the liver, so their channels are unsupervised for this case.",
    )


def pelvis_ct(seed: int = 3) -> Phantom:
    rng = np.random.default_rng(seed)
    x, y, z = _grid()
    body = _ellipsoid(x, y, z * 0.0, (0, 0.02, 0), (0.94, 0.7, 10)) < 1
    fat = body & (_ellipsoid(x, y, z * 0.0, (0, 0.02, 0), (0.85, 0.6, 10)) >= 1)
    ilium_l = body & (_ellipsoid(x, y, z, (0.58, 0.18, 0.2), (0.16, 0.36, 0.7)) < 1) & (
        _ellipsoid(x, y, z, (0.58, 0.18, 0.2), (0.11, 0.3, 0.6)) >= 1)
    ilium_r = body & (_ellipsoid(x, y, z, (-0.58, 0.18, 0.2), (0.16, 0.36, 0.7)) < 1) & (
        _ellipsoid(x, y, z, (-0.58, 0.18, 0.2), (0.11, 0.3, 0.6)) >= 1)
    sacrum = body & (_ellipsoid(x, y, z, (0, 0.5, 0.1), (0.18, 0.1, 0.8)) < 1)
    bladder = body & (_ellipsoid(x, y, z, (0, -0.32, -0.1), (0.24, 0.2, 0.32)) < 1)
    warp = _smooth_noise(rng, SHAPE, 6) * 0.2
    uterus = body & (_ellipsoid(x, y, z, (0, 0.0, 0.3), (0.2, 0.17, 0.42)) + warp < 1)
    cervix = body & (_ellipsoid(x, y, z, (0, 0.06, -0.25), (0.12, 0.11, 0.24)) + warp < 1)
    rectum = body & (_ellipsoid(x, y, z * 0.0, (0, 0.33, 0), (0.1, 0.09, 10)) < 1)
    gas = rectum & (_smooth_noise(rng, SHAPE, 3) > 0.3)
    parts = [
        (body, 40.0), (fat, -100.0), (ilium_l | ilium_r | sacrum, 700.0), (bladder, 8.0),
        (uterus, 55.0), (cervix, 70.0), (rectum, 35.0), (gas, -800.0),
    ]
    hu = _hu_volume(parts, rng=rng)
    label = np.zeros(SHAPE, np.uint8)
    label[cervix] = LABELS.index("cervix")
    return Phantom(
        case_id="PH-PELVIS-01",
        title="Synthetic pelvic CT with visible cervix",
        modality="CT",
        analog_of="no corpus supervises the cervix",
        channels=[f"{k.title()} window" for k in CT_WINDOWS],
        image=ct_to_channels(hu),
        label=label,
        supervised=supervision_vector([]),
        note="The cervix is drawn in the anatomy, but no registered corpus supervises the cervix channel, so the server suppresses it.",
    )


def all_phantoms() -> List[Phantom]:
    return [brain_mri(), abdomen_ct(), pelvis_ct()]


def random_training_phantom(rng: np.random.Generator, shape=SHAPE) -> Phantom:
    """Cheap varied phantom for the smoke-training run (different seeds -> different anatomy)."""
    seed = int(rng.integers(0, 2 ** 31))
    maker = [brain_mri, abdomen_ct][seed % 2]
    return maker(seed)
