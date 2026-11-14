"""Liver tumours, second stage: a tumour network that looks only at the liver, trained on cleaned labels.

  python -m qihgnn.screen.liver_cascade --lits <dir> --stage1 liver.onnx --out <dir> [--probe]

Why a second stage. The first run segmented the liver well (whole-scan Dice 0.93) but tumours poorly. Two causes:
1. The LiTS mirror stores its label masks as JPEG. Compression leaves 1-2-pixel specks of the tumour value along liver
   edges and vessels: hundreds of fake "lesions" per patient. Tumour regions that do not contain a solid 3x3 block are
   relabelled liver (morphological reconstruction from a 3x3 opening), for training and for testing alike.
2. Tumours are a few percent of the liver. Stage 2 crops a square around the liver, resizes it to 192 px and decides
   tumour / not tumour inside it, so the network spends its capacity where tumours can be.

Stage 1 is the published liver model (liver.onnx), run with ONNX Runtime exactly as the browser runs it.
Everything below (crop box, resizing, clean-up) is mirrored in web/src/lib/organs.ts and checked by its tests.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .common import binary_auc, binary_report, dice, group_split, summarise, write_json
from .organs import LIVER, _gray, clean_liver_mask, lits_index, train_multiclass

T0 = time.time()
TUM = 192            # stage-2 input size
BOX_MARGIN = 8       # pixels around the liver, in the 256 px slice
BOX_MIN = 48         # smallest crop side
MIN_TUMOUR_PX = 8    # 2D tumour components smaller than this are dropped (8-connected)
MIN_LESION_VOXELS = 30
GT_SLICE_TUMOUR_PX = 30  # a slice counts as "has tumour" for detection when it has at least this much (cleaned) tumour


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


# ============================================================================ geometry shared with the browser
def clean_tumour(m: np.ndarray) -> np.ndarray:
    """0/1/2 slice -> same, with tumour regions that hold no solid 3x3 block turned into liver (JPEG specks)."""
    from scipy import ndimage

    t = m == 2
    if not t.any():
        return m
    core = ndimage.binary_erosion(t, structure=np.ones((3, 3)), border_value=0)
    lab, _ = ndimage.label(t, structure=np.ones((3, 3)))
    ids = np.unique(lab[core])
    keep = np.isin(lab, ids[ids > 0])
    out = m.copy()
    out[t & ~keep] = 1
    return out


def liver_box(liver: np.ndarray, margin: int = BOX_MARGIN, min_side: int = BOX_MIN):
    """Square box (y0, x0, side) around the liver pixels; None when there is no liver."""
    ys, xs = np.nonzero(liver)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0) + 2 * margin
    side = max(side, min_side)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side


def resize_bilinear(a: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """Half-pixel-centre bilinear resize with edge clamping, float32. Same formula as resizeBilinear in organs.ts."""
    h, w = a.shape
    a = a.astype(np.float32)
    sy, sx = h / oh, w / ow
    fy = np.clip((np.arange(oh) + 0.5) * sy - 0.5, 0, h - 1)
    fx = np.clip((np.arange(ow) + 0.5) * sx - 0.5, 0, w - 1)
    y0 = np.floor(fy).astype(int); x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, h - 1); x1 = np.minimum(x0 + 1, w - 1)
    wy = (fy - y0).astype(np.float32)[:, None]; wx = (fx - x0).astype(np.float32)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def resize_nearest(a: np.ndarray, oh: int, ow: int) -> np.ndarray:
    h, w = a.shape
    iy = np.minimum((((np.arange(oh) + 0.5) * h) / oh).astype(int), h - 1)
    ix = np.minimum((((np.arange(ow) + 0.5) * w) / ow).astype(int), w - 1)
    return a[iy][:, ix]


def crop(a: np.ndarray, box) -> np.ndarray:
    """Square crop with zero padding where the box leaves the slice."""
    y0, x0, s = box
    h, w = a.shape
    out = np.zeros((s, s), a.dtype)
    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(h, y0 + s), min(w, x0 + s)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = a[ys0:ys1, xs0:xs1]
    return out


def stage2_input(img_u8: np.ndarray, box) -> np.ndarray:
    """(TUM, TUM) float32 in [0, 1]."""
    return resize_bilinear(crop(img_u8, box).astype(np.float32) / 255.0, TUM, TUM)


def uncrop(prob: np.ndarray, box, h: int = LIVER, w: int = LIVER) -> np.ndarray:
    """Stage-2 probability (TUM, TUM) back onto the full slice; zero outside the box."""
    y0, x0, s = box
    big = resize_bilinear(prob, s, s)
    out = np.zeros((h, w), np.float32)
    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(h, y0 + s), min(w, x0 + s)
    out[ys0:ys1, xs0:xs1] = big[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return out


def drop_small(mask: np.ndarray, min_px: int = MIN_TUMOUR_PX) -> np.ndarray:
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    ok = sizes >= min_px
    ok[0] = False
    return ok[lab]


def combine(stage1_cls: np.ndarray, tumour_prob: np.ndarray | None, thr: float) -> np.ndarray:
    """Final 0/1/2 map: liver where stage 1 says liver or tumour; tumour where stage 2 > thr inside that liver."""
    liver = stage1_cls >= 1
    out = liver.astype(np.uint8)
    if tumour_prob is not None:
        t = drop_small((tumour_prob > thr) & liver)
        out[t] = 2
    return out


# ============================================================================ data
def load_volumes(index, volumes: List[int], workers: int = 16, liver_only: bool = False) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Slices of each volume (all, or only those with liver): images (n, 256, 256) uint8, cleaned masks (n, 256, 256) uint8."""
    from concurrent.futures import ThreadPoolExecutor

    from .common import letterbox

    def read(pair):
        img = letterbox(_gray(pair[0]), LIVER)[0]
        msk = clean_tumour(letterbox(clean_liver_mask(_gray(pair[1])), LIVER, order=0)[0])
        return img, msk

    out = {}
    with ThreadPoolExecutor(workers) as pool:
        for v in volumes:
            sl = sorted(index[v])
            if liver_only:
                has = list(pool.map(lambda s, v=v: bool(clean_liver_mask(_gray(index[v][s][1])).any()), sl))
                sl = [s for s, h in zip(sl, has) if h]
                if not sl:
                    continue
            pairs = list(pool.map(read, [index[v][s] for s in sl]))
            out[v] = (np.stack([a for a, _ in pairs]), np.stack([b for _, b in pairs]))
    return out


HU_LO, HU_HI = -160.0, 240.0  # abdominal window, the same as ctSlice in organs.ts


def hu_to_u8(hu: np.ndarray) -> np.ndarray:
    return np.floor(np.clip((hu - HU_LO) / (HU_HI - HU_LO), 0, 1) * 255 + 0.5).astype(np.uint8)


def nifti_index(root: Path) -> Dict[int, Tuple[Path, Path]]:
    """{volume number: (volume-N.nii, segmentation-N.nii)} from the original LiTS release."""
    import re

    vols, segs = {}, {}
    for p in root.rglob("*.nii*"):
        m = re.match(r"(volume|segmentation)-(\d+)\.nii(\.gz)?$", p.name)
        if m:
            (vols if m.group(1) == "volume" else segs)[int(m.group(2))] = p
    return {v: (vols[v], segs[v]) for v in sorted(vols) if v in segs}


def read_nifti_case(vol_path: Path, seg_path: Path):
    """Slices as the browser sees them: x fastest, rows = y. Returns HU (z, y, x) float32, labels (z, y, x) uint8,
    voxel size (x, y, z) in mm."""
    import nibabel as nib

    im = nib.load(str(vol_path))
    hu = im.get_fdata(dtype=np.float32)                   # (x, y, z), scaling applied
    seg = np.asarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)
    zooms = [float(abs(z)) or 1.0 for z in im.header.get_zooms()[:3]]
    return np.ascontiguousarray(hu.transpose(2, 1, 0)), np.ascontiguousarray(seg.transpose(2, 1, 0)), zooms


def load_nifti(index, volumes: List[int], train: bool, rng: np.random.Generator, workers: int = 4):
    """Per volume: windowed images (n, 256, 256) uint8, labels (n, 256, 256) uint8 (0 bg, 1 liver, 2 tumour) and the
    voxel size of the 256 px grid (x, y, z) in mm. Training volumes keep their liver slices plus a quarter as many
    liver-free slices; validation and test volumes keep every slice."""
    from concurrent.futures import ThreadPoolExecutor

    from .common import letterbox

    def one(v):
        hu, seg, (sx, sy, sz) = read_nifti_case(*index[v])
        seg[seg > 2] = 0
        zs = np.arange(len(hu))
        if train:
            has = seg.reshape(len(seg), -1).any(1)
            pos = zs[has]
            neg = zs[~has]
            r = np.random.default_rng(int(rng.integers(1 << 30)) + v)
            neg = r.choice(neg, min(len(neg), max(2, len(pos) // 4)), replace=False) if len(neg) else neg
            zs = np.sort(np.concatenate([pos, neg]).astype(int))
        imgs, msks, scale = [], [], 1.0
        for z in zs:
            g, lb = letterbox(hu_to_u8(hu[z]), LIVER)
            imgs.append(g)
            msks.append(letterbox(seg[z], LIVER, order=0)[0])
            scale = lb["scale"]
        return v, (np.stack(imgs), np.stack(msks), (sx / scale, sy / scale, sz))

    out = {}
    with ThreadPoolExecutor(workers) as pool:
        for v, d in pool.map(one, volumes):
            out[v] = d
    return out


def write_nifti_i16(path: Path, vol_zyx: np.ndarray, spacing_xyz) -> None:
    """NIfTI-1, int16 Hounsfield units stored [z][y][x]."""
    import gzip
    import struct

    z, y, x = vol_zyx.shape
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, x, y, z, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 4, 16)
    struct.pack_into("<8f", hdr, 76, 1.0, *spacing_xyz, 0, 0, 0, 0)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<ff", hdr, 112, 1.0, 0.0)
    struct.pack_into("<B", hdr, 123, 2)
    hdr[344:348] = b"n+1\x00"
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(bytes(hdr) + b"\x00" * 4 + np.ascontiguousarray(vol_zyx, "<i2").tobytes())


def stage1(sess, imgs: np.ndarray, batch: int = 32) -> np.ndarray:
    """Liver model over (n, 256, 256) uint8 -> argmax classes (n, 256, 256)."""
    outs = []
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch, None].astype(np.float32) / 255.0
        p = sess.run(None, {"slice": x})[0]
        outs.append(p.argmax(1).astype(np.uint8))
    return np.concatenate(outs) if outs else np.zeros((0, LIVER, LIVER), np.uint8)


def tumour_dataset(imgs: np.ndarray, masks: np.ndarray, items: List[int], train: bool):
    import torch
    from torch.utils.data import Dataset

    class DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, k):
            i = items[k]
            img, m = imgs[i], masks[i]
            rng = np.random.default_rng()
            margin = int(rng.integers(3, 17)) if train else BOX_MARGIN
            y0, x0, s = liver_box(m >= 1, margin=margin)
            if train:
                j = max(2, s // 12)
                y0 += int(rng.integers(-j, j + 1)); x0 += int(rng.integers(-j, j + 1))
            box = (y0, x0, s)
            x = stage2_input(img, box)
            y = (resize_nearest(crop(m, box), TUM, TUM) == 2).astype(np.int64)
            if train:
                if rng.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                gamma = float(np.exp(rng.uniform(-0.25, 0.25)))
                x = np.clip((x ** gamma) * rng.uniform(0.9, 1.1) + rng.uniform(-0.04, 0.04), 0, 1)
                x = x + rng.normal(0, 0.01, x.shape).astype(np.float32)
            return torch.from_numpy(np.ascontiguousarray(x[None]).astype(np.float32)), torch.from_numpy(np.ascontiguousarray(y))

    return DS()


def run_stage2(net, imgs: np.ndarray, s1: np.ndarray, device: str, batch: int = 32) -> List[np.ndarray | None]:
    """Tumour probability (256, 256) per slice where stage 1 found liver, else None."""
    import torch

    boxes = [liver_box(c >= 1) for c in s1]
    idx = [i for i, b in enumerate(boxes) if b is not None]
    probs: List[np.ndarray | None] = [None] * len(imgs)
    net.eval()
    with torch.no_grad():
        for k in range(0, len(idx), batch):
            ii = idx[k:k + batch]
            x = np.stack([stage2_input(imgs[i], boxes[i]) for i in ii])[:, None]
            p = torch.softmax(net(torch.from_numpy(x).to(device)), 1)[:, 1].float().cpu().numpy()
            for i, pi in zip(ii, p):
                probs[i] = uncrop(pi, boxes[i])
    return probs


def volume_rows(data, s1s, probs_all, thr: float) -> List[Dict]:
    from scipy import ndimage

    rows = []
    for v, d in data.items():
        imgs, msks = d[0], d[1]
        vox = (d[2][0] * d[2][1] * d[2][2] / 1000.0) if len(d) > 2 else None  # cm3 per voxel of the 256 px grid
        s1, probs = s1s[v], probs_all[v]
        pred = np.stack([combine(s1[i], probs[i], thr) for i in range(len(imgs))])
        gt_t, pr_t = msks == 2, pred == 2
        lt, nt = ndimage.label(gt_t)
        lp, npr = ndimage.label(pr_t)
        gt_sizes = np.bincount(lt.ravel())[1:] if nt else np.array([], int)
        pr_sizes = np.bincount(lp.ravel())[1:] if npr else np.array([], int)
        gt_ids = [k + 1 for k, s in enumerate(gt_sizes) if s >= MIN_LESION_VOXELS]
        pr_ids = [k + 1 for k, s in enumerate(pr_sizes) if s >= MIN_LESION_VOXELS]
        hit_gt = set(np.unique(lt[pr_t]).tolist()) - {0}
        hit_pr = set(np.unique(lp[gt_t]).tolist()) - {0}
        lesions = [{"voxels": int(gt_sizes[k - 1]), "found": k in hit_gt} for k in gt_ids]
        fp = sum(1 for k in pr_ids if k not in hit_pr)
        slices = []
        for i in range(len(imgs)):
            if not (msks[i] >= 1).any():
                continue
            g = int(gt_t[i].sum())
            if 0 < g < GT_SLICE_TUMOUR_PX:
                continue  # too little tumour to call either way
            slices.append((int(pr_t[i].sum()), int(g >= GT_SLICE_TUMOUR_PX)))
        liver_t, liver_p = (msks >= 1).sum(), (pred >= 1).sum()
        rows.append({"volume": int(v), "dice_liver": dice(pred >= 1, msks >= 1), "dice_tumour": dice(pr_t, gt_t) if gt_t.any() else None,
                     "burden_true": float(gt_t.sum() / max(1, liver_t)), "burden_pred": float(pr_t.sum() / max(1, liver_p)),
                     "lesions_true": len(gt_ids), "lesions_pred": len(pr_ids), "lesions": lesions, "false_lesions": fp,
                     "slices": slices, "has_tumour": int(len(gt_ids) > 0),
                     "liver_cc": [float(liver_t * vox), float(liver_p * vox)] if vox else None,
                     "tumour_cc": [float(gt_t.sum() * vox), float(pr_t.sum() * vox)] if vox else None})
    return rows


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nifti", default="", help="original LiTS release (volume-N.nii, segmentation-N.nii): trains both stages")
    ap.add_argument("--lits", default="", help="JPEG mirror of LiTS (used with --stage1 when --nifti is not given)")
    ap.add_argument("--stage1", default="", help="published liver.onnx, for the mirror route")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--s1-epochs", type=int, default=20)
    ap.add_argument("--s1-steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--base", type=int, default=24)
    a = ap.parse_args()
    import torch

    from .models import SoftmaxExport, UNet2D, count
    from .organs import liver_dataset, predict
    from .train import export_onnx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    P = a.probe
    rng = np.random.default_rng(2026)
    NIFTI = bool(a.nifti)

    if NIFTI:
        index = nifti_index(Path(a.nifti))
        n_slices = None
    else:
        index = lits_index(Path(a.lits))
        n_slices = sum(len(s) for s in index.values())
    vols = sorted(index)
    split = {v: group_split(f"lits:{v}") for v in vols}  # same per-patient split as every earlier liver run
    if P:
        vols = [v for v in vols if split[v] == "train"][:3] + [v for v in vols if split[v] == "val"][:1] + [v for v in vols if split[v] == "test"][:2]
    pick = lambda k: [v for v in vols if split[v] == k]
    if NIFTI:
        tr, va, te = load_nifti(index, pick("train"), True, rng), load_nifti(index, pick("val"), False, rng), load_nifti(index, pick("test"), False, rng)
    else:
        tr = load_volumes(index, pick("train"), liver_only=True)
        va, te = load_volumes(index, pick("val")), load_volumes(index, pick("test"))
    log("volumes loaded", {k: (len(d), sum(len(x[0]) for x in d.values())) for k, d in (("train", tr), ("val", va), ("test", te))})
    imgs = np.concatenate([d[0] for d in tr.values()])
    msks = np.concatenate([d[1] for d in tr.values()])

    # ---- stage 1: liver (trained here on real CT; on the mirror route the published model is reused)
    s1_info = None
    if NIFTI:
        vx = np.concatenate([d[0] for d in va.values()])
        vy = np.concatenate([d[1] for d in va.values()])
        keep = (vy >= 1).reshape(len(vy), -1).any(1)
        vx, vy = vx[keep][:, None].astype(np.float32) / 255.0, vy[keep]

        def liver_val(net):
            pred = predict(net, vx, device).argmax(1)
            return float(np.mean([dice(p >= 1, y >= 1) for p, y in zip(pred, vy)]))

        s1net, s1hist = train_multiclass(UNet2D(base=16, in_ch=1, out_ch=3), liver_dataset(list(imgs), list(msks), True), liver_val,
                                         epochs=1 if P else a.s1_epochs, max_steps=3 if P else a.s1_steps, device=device, batch=24,
                                         lr=2e-3, class_weights=[0.5, 1.0, 3.0], name="liver")
        s1fn = lambda x: predict(s1net, x[:, None].astype(np.float32) / 255.0, device).argmax(1).astype(np.uint8)
        s1_info = {"parameters": count(s1net), "history": s1hist}
    else:
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        sess = ort.InferenceSession(a.stage1, providers=providers)
        s1fn = lambda x: stage1(sess, x)

    # ---- stage 2: tumour inside the liver; every slice with liver, slices with tumour twice
    has_liver = (msks >= 1).reshape(len(msks), -1).any(1)
    has_tum = (msks == 2).reshape(len(msks), -1).sum(1) >= 1
    items = [int(i) for i in np.nonzero(has_liver)[0]] + [int(i) for i in np.nonzero(has_liver & has_tum)[0]]
    log("stage-2 training slices", int(has_liver.sum()), "with tumour", int((has_liver & has_tum).sum()))
    ds = tumour_dataset(imgs, msks, items, True)
    va_s1 = {v: s1fn(d[0]) for v, d in va.items()}

    def val_score(net, thr=0.5):
        probs = {v: run_stage2(net, d[0], va_s1[v], device) for v, d in va.items()}
        rows = volume_rows(va, va_s1, probs, thr)
        return float(np.mean([r["dice_tumour"] for r in rows if r["dice_tumour"] is not None] or [0])), probs

    net, hist = train_multiclass(UNet2D(base=a.base, in_ch=1, out_ch=2), ds, lambda n: val_score(n)[0],
                                 epochs=1 if P else a.epochs, max_steps=3 if P else a.steps, device=device, batch=32,
                                 lr=2e-3, class_weights=[1.0, 2.0], name="tumour")

    # ---- operating points chosen on validation patients only
    _, va_probs = val_score(net)
    thr_scores = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        rr = volume_rows(va, va_s1, va_probs, t)
        thr_scores[t] = float(np.mean([r["dice_tumour"] for r in rr if r["dice_tumour"] is not None] or [0]))
    thr = max(thr_scores, key=thr_scores.get)
    va_sl = [x for r in volume_rows(va, va_s1, va_probs, thr) for x in r["slices"]]
    sp = np.array([p for p, _ in va_sl]); sy = np.array([y for _, y in va_sl], bool)

    def youden(c):
        pred = sp >= c
        return (pred & sy).sum() / max(1, sy.sum()) + (~pred & ~sy).sum() / max(1, (~sy).sum())

    cands = sorted({int(x) for x in sp if x > 0})
    if len(cands) > 400:
        cands = sorted({int(x) for x in np.percentile(cands, np.linspace(0, 100, 400))})
    slice_px = max(cands or [1], key=youden)
    log("validation: tumour threshold", thr, thr_scores, "slice rule: tumour px >=", slice_px)

    # ---- test patients, every slice
    te_s1 = {v: s1fn(d[0]) for v, d in te.items()}
    te_probs = {v: run_stage2(net, d[0], te_s1[v], device) for v, d in te.items()}
    rows = volume_rows(te, te_s1, te_probs, thr)
    sl = [x for r in rows for x in r["slices"]]
    det = binary_report([p for p, _ in sl], [y for _, y in sl], threshold=slice_px) if sl else None
    if det is not None:
        det["auc"] = binary_auc([p for p, _ in sl], [y for _, y in sl])
    les = [l for r in rows for l in r["lesions"]]
    big = [l for l in les if l["voxels"] >= 1000]
    small = [l for l in les if l["voxels"] < 1000]
    bt = np.array([r["burden_true"] for r in rows]); bp = np.array([r["burden_pred"] for r in rows])
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None
    M = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "probe": P}
    M["liver_data"] = {"source": "LiTS, original CT volumes (NIfTI, Hounsfield units)" if NIFTI else "LiTS, JPEG mirror",
                       "volumes": len(index), "slices": n_slices,
                       "splits": {k: sum(split[v] == k for v in index) for k in ("train", "val", "test")}}
    L = {
        "architecture": "Two stages: a 2D U-Net finds the liver (one CT slice in, abdominal window); a second U-Net looks only at a square around the liver and outlines tumours",
        "stage1": s1_info,
        "stage2": {"input": TUM, "box_margin": BOX_MARGIN, "box_min": BOX_MIN, "parameters": count(net), "history": hist},
        "window_hu": [HU_LO, HU_HI],
        "tumour_threshold": thr, "tumour_threshold_val_dice": thr_scores, "min_tumour_px": MIN_TUMOUR_PX, "min_lesion_voxels": MIN_LESION_VOXELS,
        "slice_rule_tumour_px": int(slice_px),
        "test_patients": len(rows), "patients_with_tumour": int(sum(r["has_tumour"] for r in rows)),
        "volume_dice_liver": summarise([r["dice_liver"] for r in rows]),
        "volume_dice_tumour": summarise([r["dice_tumour"] for r in rows if r["dice_tumour"] is not None]),
        "slice_detection": det,
        "lesions": {"n": len(les), "found": sum(l["found"] for l in les),
                    "recall": sum(l["found"] for l in les) / max(1, len(les)),
                    "recall_large": sum(l["found"] for l in big) / max(1, len(big)), "n_large": len(big),
                    "recall_small": sum(l["found"] for l in small) / max(1, len(small)), "n_small": len(small),
                    "false_per_patient": summarise([r["false_lesions"] for r in rows])},
        "burden_pearson_r": corr(bt, bp),
        "burden_pairs": [[round(x, 4), round(y, 4)] for x, y in zip(bt, bp)],
        "lesion_count_pairs": [[r["lesions_true"], r["lesions_pred"]] for r in rows],
        "per_patient": [{k: r[k] for k in ("volume", "dice_liver", "dice_tumour", "lesions_true", "lesions_pred", "false_lesions", "liver_cc", "tumour_cc")} for r in rows],
    }
    if NIFTI:
        lc = np.array([r["liver_cc"] for r in rows]); tc = np.array([r["tumour_cc"] for r in rows])
        L["liver_volume_error_cc"] = summarise(np.abs(lc[:, 0] - lc[:, 1]))
        L["liver_volume_error_pct"] = summarise(100 * np.abs(lc[:, 0] - lc[:, 1]) / np.maximum(lc[:, 0], 1e-6))
        L["liver_volume_r"] = corr(lc[:, 0], lc[:, 1])
        L["tumour_volume_error_cc"] = summarise(np.abs(tc[:, 0] - tc[:, 1]))
        L["tumour_volume_r"] = corr(tc[:, 0], tc[:, 1])
        L["note"] = "Volumes use each scan's own voxel size. Lesion size bands are in voxels of the 256 px grid (1000 voxels is roughly 2-4 cm3)."
    else:
        L["label_cleaning"] = "The LiTS mirror stores masks as JPEG; tumour regions with no solid 3x3 block (compression specks) are relabelled liver in training and test data alike."
        L["note"] = "Slice spacing is not stored in this mirror, so sizes are in pixels and relative to the liver; NIfTI uploads use their own voxel size."
    M["liver"] = L
    log("test", {"liver": L["volume_dice_liver"]["mean"], "tumour": L["volume_dice_tumour"]["mean"], "tumour_median": L["volume_dice_tumour"]["median"],
                 "lesion_recall": L["lesions"]["recall"], "recall_large": L["lesions"]["recall_large"], "fp_per_patient": L["lesions"]["false_per_patient"]["mean"],
                 "slice_sens_spec_auc": det and (det["sensitivity"], det["specificity"], det["auc"]),
                 "liver_cc_err_pct": L.get("liver_volume_error_pct", {}).get("median"), "tumour_cc_r": L.get("tumour_volume_r")})

    M["export"] = {"liver_tumour": export_onnx(SoftmaxExport(net), torch.rand(1, 1, TUM, TUM), out / "liver_tumour.onnx", "crop", ["prob"])}
    if NIFTI:
        M["export"]["liver"] = export_onnx(SoftmaxExport(s1net), torch.rand(1, 1, LIVER, LIVER), out / "liver.onnx", "slice", ["prob"])

    # ---- samples from typical test patients (closest to the median tumour Dice), not the best ones
    from PIL import Image

    samples = []
    with_t = [r for r in rows if r["dice_tumour"] is not None]
    med = float(np.median([r["dice_tumour"] for r in with_t])) if with_t else 0.0
    order = sorted(with_t, key=lambda r: abs(r["dice_tumour"] - med))
    for r in order[:6]:
        imgs_v, msks_v = te[r["volume"]][0], te[r["volume"]][1]
        tum = [i for i in range(len(msks_v)) if (msks_v[i] == 2).sum() > 150]
        if tum and sum(x["truth"] == "liver_tumour" for x in samples) < 2:
            i = tum[len(tum) // 2]
            name = f"liver_tumour_{r['volume']}_{i}.png"
            Image.fromarray(imgs_v[i]).save(out / "samples" / name)
            samples.append({"file": name, "truth": "liver_tumour", "kind": "liver", "source": "LiTS (test patient of this project)", "tumour_px": int((msks_v[i] == 2).sum())})
    for r in rows:
        imgs_v, msks_v = te[r["volume"]][0], te[r["volume"]][1]
        clean = [i for i in range(len(msks_v)) if (msks_v[i] == 1).sum() > 2500 and not (msks_v[i] == 2).any()]
        if clean:
            i = clean[len(clean) // 2]
            name = f"liver_no_tumour_{r['volume']}_{i}.png"
            Image.fromarray(imgs_v[i]).save(out / "samples" / name)
            samples.append({"file": name, "truth": "liver_no_tumour", "kind": "liver", "source": "LiTS (test patient of this project)"})
            break
    if order:
        pick_r = order[0]
        v = pick_r["volume"]
        msks_v = te[v][1]
        zs = np.nonzero((msks_v >= 1).reshape(len(msks_v), -1).any(1))[0]
        lo, hi = max(0, int(zs.min()) - 6), min(len(msks_v), int(zs.max()) + 7)
        entry = {"truth": "liver_tumour", "kind": "liver-volume", "true_lesions": pick_r["lesions_true"]}
        if NIFTI:
            from .common import letterbox as _lb

            hu, _, (sx, sy, sz) = read_nifti_case(*index[v])
            sl = []
            for z in range(lo, hi):
                from PIL import Image as _I

                im = _I.fromarray(hu[z]).resize((LIVER, LIVER), _I.BILINEAR)
                sl.append(np.clip(np.rint(np.asarray(im)), -1024, 3071).astype(np.int16))
            sp3 = (sx * hu.shape[2] / LIVER, sy * hu.shape[1] / LIVER, sz)
            name = f"liver_ct_volume_{v}.nii.gz"
            write_nifti_i16(out / "samples" / name, np.stack(sl), sp3)
            vox = sp3[0] * sp3[1] * sp3[2] / 1000
            entry.update({"file": name, "source": "LiTS (test patient of this project with the median tumour Dice): liver slices only, in-plane resolution halved to keep the file small; Hounsfield units and voxel size kept",
                          "true_tumour_cc": float((msks_v[lo:hi] == 2).sum() * vox), "true_liver_cc": float((msks_v[lo:hi] >= 1).sum() * vox)})
        else:
            from .train import write_nifti_gz

            name = f"liver_ct_volume_{v}.nii.gz"
            write_nifti_gz(out / "samples" / name, te[v][0][lo:hi], (1.0, 1.0, 1.0))
            entry.update({"file": name, "source": "LiTS (test patient of this project with the median tumour Dice), liver slices only; voxel size not stored in this mirror",
                          "true_tumour_px": int((msks_v[lo:hi] == 2).sum()), "true_liver_px": int((msks_v[lo:hi] >= 1).sum())})
        samples.append(entry)
    write_json(out / "samples" / "liver_index.json", samples)
    M["minutes"] = round((time.time() - T0) / 60, 1)
    write_json(out / "organs_metrics.json", M)
    log("done", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
