"""Train, evaluate and export the screening models.

  python -m qihgnn.screen.train --raw <dir with one sub-folder per dataset> --out <dir> [--probe]

--probe runs every step on a handful of cases for a few iterations (CPU, minutes), so that the
whole pipeline, including ONNX export and the browser-parity check, is exercised before a GPU run.
"""
from __future__ import annotations

import argparse
import gzip
import struct
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import (CLASSES, CLS_SIZE, NOT_BRAIN, SEG_SIZE, binary_report, confusion, dhash_canonical, dice, duplicate_groups, ece,
                     group_split, letterbox, seg_normalise, summarise, wilson, write_json)
from . import data as D
from .survival import design, fit_and_report

T0 = time.time()
STORE = 256  # classifier images are kept letterboxed at 256 px and randomly cropped to 224


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


# ============================================================================ datasets (torch)
def _torch():
    import torch
    from torch.utils.data import DataLoader, Dataset
    return torch, DataLoader, Dataset


def augment_gray(img: np.ndarray, rng: np.random.Generator, out: int, mask: np.ndarray | None = None):
    """Rotation, scale/crop, flip and intensity jitter. The same geometry is applied to a mask if given."""
    from PIL import Image

    s = img.shape[0]
    ang = float(rng.uniform(-12, 12))
    scale = float(rng.uniform(0.85, 1.0))
    c = int(round(s * scale))
    x0, y0 = int(rng.integers(0, s - c + 1)), int(rng.integers(0, s - c + 1))
    flip = rng.random() < 0.5

    def geo(a, resample):
        im = Image.fromarray(a).rotate(ang, resample=resample)
        im = im.crop((x0, y0, x0 + c, y0 + c)).resize((out, out), resample)
        return np.asarray(im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im)

    g = geo(img, Image.BILINEAR).astype(np.float32) / 255.0
    gamma = float(np.exp(rng.uniform(-0.3, 0.3)))
    g = np.clip((g ** gamma) * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06), 0, 1)
    return (g, geo(mask, Image.NEAREST)) if mask is not None else g


def make_cls_dataset(images: List[np.ndarray], labels: List[int], train: bool):
    torch, _, Dataset = _torch()

    class DS(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, i):
            rng = np.random.default_rng()
            if train:
                g = augment_gray(images[i], rng, CLS_SIZE)
            else:
                g = letterbox(images[i], CLS_SIZE)[0].astype(np.float32) / 255.0
            return torch.from_numpy(g[None].astype(np.float32)), int(labels[i])

    return DS()


def make_seg_dataset(xs: List[np.ndarray], ys: List[np.ndarray], train: bool):
    """xs: uint8 (C, S, S) with C sequences (BraTS) or 1 (figshare); a random sequence is drawn per sample."""
    torch, _, Dataset = _torch()

    class DS(Dataset):
        def __len__(self):
            return len(xs)

        def __getitem__(self, i):
            rng = np.random.default_rng()
            x = xs[i][int(rng.integers(0, xs[i].shape[0]))] if train else xs[i][-1]
            y = ys[i]
            if train:
                x, y = augment_gray(x, rng, SEG_SIZE, y * 255)
                x = (x * 255).astype(np.uint8)
                y = (y > 127).astype(np.uint8)
            return torch.from_numpy(seg_normalise(x)[None]), torch.from_numpy(y[None].astype(np.float32))

    return DS()


# ============================================================================ classifier
def train_classifier(tr, va, epochs: int, max_steps: int, device: str, batch: int = 48) -> Dict:
    torch, DataLoader, _ = _torch()
    from .models import build_classifier

    net = build_classifier(pretrained=True).to(device)
    counts = np.bincount(tr[1], minlength=len(CLASSES)).astype(float)
    weights = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1)) / np.sqrt(len(CLASSES)), dtype=torch.float32, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    dl = DataLoader(make_cls_dataset(*tr, train=True), batch_size=batch, shuffle=True, num_workers=4 if device == "cuda" else 0, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    total = epochs * (min(len(dl), max_steps) if max_steps else len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=max(1, total), pct_start=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        net.train()
        run = []
        for step, (x, y) in enumerate(dl):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(net(_imnorm(x)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(float(loss.detach()))
        logits, _ = predict_logits(net, va[0], device)
        pred = logits.argmax(1)
        f1 = macro_f1(np.asarray(va[1]), pred)
        hist.append({"epoch": ep, "train_loss": float(np.mean(run)) if run else None, "val_macro_f1": f1,
                     "val_acc": float((pred == np.asarray(va[1])).mean())})
        log(f"classifier epoch {ep}: loss {hist[-1]['train_loss']:.4f}  val acc {hist[-1]['val_acc']:.4f}  macro-F1 {f1:.4f}")
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return {"net": net, "history": hist}


def _imnorm(x):
    import torch
    from .models import IMAGENET_MEAN, IMAGENET_STD
    m = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x.repeat(1, 3, 1, 1) - m) / s


def predict_logits(net, images: List[np.ndarray], device: str, batch: int = 64):
    torch, DataLoader, _ = _torch()
    dl = DataLoader(make_cls_dataset(images, [0] * len(images), train=False), batch_size=batch, num_workers=0)
    net.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(net(_imnorm(x.to(device))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES))), None


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for k in range(len(CLASSES)):
        tp = ((p == k) & (y == k)).sum()
        fp = ((p == k) & (y != k)).sum()
        fn = ((p != k) & (y == k)).sum()
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 120))
    nll = [-np.log(softmax(logits, t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def tumour_prob(p: np.ndarray) -> np.ndarray:
    brain = p[:, :NOT_BRAIN].sum(1)
    return p[:, 1:NOT_BRAIN].sum(1) / np.maximum(brain, 1e-9)


def choose_threshold(p: np.ndarray, y: np.ndarray, target: float = 0.98) -> float:
    """Smallest confidence threshold whose accepted predictions reach `target` accuracy on validation."""
    conf, pred = p.max(1), p.argmax(1)
    for t in np.linspace(0.40, 0.95, 56):
        m = conf >= t
        if m.sum() >= 10 and (pred[m] == y[m]).mean() >= target:
            return float(t)
    return 0.95


def classifier_report(p: np.ndarray, y: np.ndarray, sources: List[str], thr: float) -> Dict:
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    cm = confusion(y, pred, len(CLASSES))
    per = {}
    for k, name in enumerate(CLASSES):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        per[name] = {"n": int(cm[k].sum()), "precision": float(tp / max(1, tp + fp)), "recall": float(tp / max(1, tp + fn)),
                     "recall_ci": wilson(int(tp), int(tp + fn))}
    brain = y != NOT_BRAIN
    bin_ = binary_report(tumour_prob(p[brain]), (y[brain] > 0).astype(int))
    nb = binary_report(p[:, NOT_BRAIN], (y == NOT_BRAIN).astype(int))
    covered = p.max(1) >= thr
    by_source = {}
    for s in sorted(set(sources)):
        m = np.asarray([x == s for x in sources])
        by_source[s] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
    return {"n": int(len(y)), "accuracy": acc, "accuracy_ci": wilson(int((pred == y).sum()), len(y)),
            "macro_f1": macro_f1(y, pred), "per_class": per, "confusion": cm.tolist(), "labels": list(CLASSES),
            "tumour_vs_no_tumour": bin_, "not_brain_mri_detection": nb, "by_source": by_source,
            "ece": ece(p, y), "abstention": {"threshold": thr, "coverage": float(covered.mean()),
                                             "accuracy_when_answered": float((pred[covered] == y[covered]).mean()) if covered.any() else None}}


# ============================================================================ segmenter
def train_segmenter(tr, va, epochs: int, max_steps: int, device: str, batch: int = 32) -> Dict:
    torch, DataLoader, _ = _torch()
    from .models import UNet2D

    net = UNet2D().to(device)
    dl = DataLoader(make_seg_dataset(*tr, train=True), batch_size=batch, shuffle=True, num_workers=4 if device == "cuda" else 0, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    total = epochs * (min(len(dl), max_steps) if max_steps else len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(1, total), pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        net.train()
        run = []
        for step, (x, y) in enumerate(dl):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            z = net(x)
            p = torch.sigmoid(z)
            inter = (p * y).sum((1, 2, 3))
            soft_dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = bce(z, y) + soft_dice.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run.append(float(loss.detach()))
        d = np.mean([dice(a, b) for a, b in zip(seg_predict(net, [x[-1] for x in va[0]], device) > 0.5, va[1]) if b.any()] or [0.0])
        hist.append({"epoch": ep, "train_loss": float(np.mean(run)) if run else None, "val_dice_tumour_slices": float(d)})
        log(f"segmenter epoch {ep}: loss {hist[-1]['train_loss']:.4f}  val Dice {d:.4f}")
        if d > best:
            best, best_state = d, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return {"net": net, "history": hist}


def seg_predict(net, slices_u8: List[np.ndarray], device: str, batch: int = 64) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(slices_u8), batch):
            x = torch.from_numpy(np.stack([seg_normalise(s) for s in slices_u8[i:i + batch]])[:, None]).to(device)
            out.append(torch.sigmoid(net(x)).float().cpu().numpy()[:, 0])
    return np.concatenate(out) if out else np.zeros((0, SEG_SIZE, SEG_SIZE), np.float32)


# ============================================================================ export
def export_onnx(module, example, path: Path, input_name: str, output_names: List[str]) -> Dict:
    import torch

    module = module.cpu().eval()
    kw = dict(input_names=[input_name], output_names=output_names, opset_version=17,
              dynamic_axes={input_name: {0: "batch"}, **{o: {0: "batch"} for o in output_names}})
    try:
        torch.onnx.export(module, (example,), str(path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(module, (example,), str(path), **kw)
    report = {"file": path.name, "bytes": path.stat().st_size}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = module(example)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        got = sess.run(None, {input_name: example.numpy()})
        report["max_abs_diff_vs_torch"] = float(max(np.abs(a - b.numpy()).max() for a, b in zip(got, ref)))
    except Exception as exc:
        report["parity_check_error"] = f"{type(exc).__name__}: {exc}"
    return report


def write_nifti_gz(path: Path, vol_zyx: np.ndarray, spacing_xyz) -> None:
    """Minimal NIfTI-1 writer for a uint8 volume stored [z][y][x] (x varies fastest, as NIfTI expects)."""
    z, y, x = vol_zyx.shape
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, x, y, z, 1, 1, 1, 1)
    struct.pack_into("<hh", hdr, 70, 2, 8)                              # datatype uint8, bitpix 8
    struct.pack_into("<8f", hdr, 76, 1.0, *spacing_xyz, 0, 0, 0, 0)     # pixdim
    struct.pack_into("<f", hdr, 108, 352.0)                             # vox_offset
    struct.pack_into("<ff", hdr, 112, 1.0, 0.0)                         # scl_slope, scl_inter
    struct.pack_into("<B", hdr, 123, 2)                                 # xyzt_units: mm
    hdr[344:348] = b"n+1\x00"
    with gzip.open(path, "wb") as f:
        f.write(bytes(hdr) + b"\x00" * 4 + np.ascontiguousarray(vol_zyx, np.uint8).tobytes())


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mri4", required=True, help="Brain Tumor MRI Dataset (4 classes incl. no tumour)")
    ap.add_argument("--external", default="", help="Brain MRI Images for Brain Tumor Detection (yes/no), external test only")
    ap.add_argument("--figshare", default="", help="2D brain tumour segmentation set (images + masks)")
    ap.add_argument("--brats", required=True)
    ap.add_argument("--ct", default="", help="abdominal CT slices (not-brain examples)")
    ap.add_argument("--photos", default="", help="colposcopy photographs (not-brain examples)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cls-epochs", type=int, default=14)
    ap.add_argument("--seg-epochs", type=int, default=30)
    a = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    P = a.probe
    log("device", device, "| probe" if P else "| full run", "| torch", torch.__version__)
    metrics: Dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "probe": P}

    # ---------------------------------------------------------------- BraTS pass
    D.print_head(Path(a.brats))
    vols, binfo = D.brats_volumes(Path(a.brats), limit=6 if P else 0, workers=4 if P else 8)
    metrics["brats"] = {**binfo, "volumes_used": len(vols), "splits": {s: sum(v.split == s for v in vols) for s in ("train", "val", "test")}}
    log("BraTS volumes", binfo, metrics["brats"]["splits"])

    # ---------------------------------------------------------------- classifier data
    items: List[D.Item] = D.masoud_items(Path(a.mri4), limit=240 if P else 0)
    for v in vols:
        for img, lab, seq in v.cls:
            items.append(D.Item("brats-slices", lab, f"brats:{v.patient_id}", array=img, meta={"sequence": seq}))
    if a.ct:
        items += D.ct_items(Path(a.ct), per_volume=10, max_volumes=3 if P else 0)
    if a.photos:
        items += D.photo_items(Path(a.photos), limit=20 if P else 0)
    items += D.synthetic_items(30 if P else 400)
    log("classifier items", len(items), {s: sum(i.source == s for i in items) for s in sorted({i.source for i in items})})
    hashes = D.hash_items(items)
    bad = {i for i, h in hashes.items() if h < 0}
    items = [it for i, it in enumerate(items) if i not in bad]
    hashes = {j: hashes[i] for j, i in enumerate(i for i in range(len(hashes)) if i not in bad)}
    metrics["dedup_audit"] = D.assign_groups(items, hashes)
    metrics["dedup_audit"]["unreadable_files_dropped"] = len(bad)
    log("dedup audit", metrics["dedup_audit"])
    split = D.split_items(items)
    imgs = {s: [letterbox(im, STORE)[0] if max(im.shape) != STORE else im for im in D.load_images(split[s])] for s in split}
    labels = {s: [it.label for it in split[s]] for s in split}
    metrics["classifier_data"] = {s: {"n": len(split[s]), "per_class": np.bincount(labels[s], minlength=len(CLASSES)).tolist()} for s in split}

    # external test set, with anything that also appears in our data removed
    ext = D.external_items(Path(a.external)) if a.external else []
    ext_report = None
    if ext:
        eh = D.hash_items(ext)
        ours = list(hashes.values())
        allh = ours + [eh[i] for i in range(len(ext))]
        gid = duplicate_groups(allh)
        seen = set(gid[: len(ours)])
        # The external set ships every image twice (yes/no and brain_tumor_dataset/yes|no): keep one per cluster,
        # then drop any cluster that also contains one of our images.
        first: Dict[int, int] = {}
        for i in range(len(ext)):
            if eh[i] >= 0:
                first.setdefault(gid[len(ours) + i], i)
        ext_internal_dupes = len(ext) - len(first)
        keep = [i for g, i in first.items() if g not in seen]
        ext_overlap = len(first) - len(keep)
        ext = [ext[i] for i in sorted(keep)]
        log("external test images", len(ext), "| internal duplicates removed:", ext_internal_dupes, "| copies of our data removed:", ext_overlap)

    # ---------------------------------------------------------------- classifier
    res = train_classifier((imgs["train"], labels["train"]), (imgs["val"], labels["val"]),
                           epochs=1 if P else a.cls_epochs, max_steps=3 if P else 0, device=device)
    net = res["net"]
    zv, _ = predict_logits(net, imgs["val"], device)
    yv = np.asarray(labels["val"])
    T = fit_temperature(zv, yv)
    thr = choose_threshold(softmax(zv, T), yv)
    zt, _ = predict_logits(net, imgs["test"], device)
    yt = np.asarray(labels["test"])
    pt = softmax(zt, T)
    rep = classifier_report(pt, yt, [it.source for it in split["test"]], thr)
    rep["ece_before_calibration"] = ece(softmax(zt, 1.0), yt)
    metrics["classifier"] = {"architecture": "EfficientNet-B0, ImageNet-initialised, 5 classes", "temperature": T,
                             "history": res["history"], "test": rep}
    if ext:
        ze, _ = predict_logits(net, D.load_images(ext), device)
        pe = softmax(ze, T)
        ye = np.asarray([it.label for it in ext])
        ext_report = binary_report(tumour_prob(pe), ye)
        ext_report["rejected_as_not_brain_mri"] = int((pe.argmax(1) == NOT_BRAIN).sum())
        ext_report["removed_as_copies_of_training_data"] = ext_overlap
        ext_report["internal_duplicates_removed"] = ext_internal_dupes
        metrics["classifier"]["external_test"] = ext_report
    log("classifier test", {k: rep[k] for k in ("accuracy", "macro_f1", "ece")}, "tumour/no:", rep["tumour_vs_no_tumour"]["sensitivity"], rep["tumour_vs_no_tumour"]["specificity"])
    if ext_report:
        log("external", {k: ext_report[k] for k in ("n", "sensitivity", "specificity", "auc")})

    # ---------------------------------------------------------------- segmenter data
    seg = {s: ([], []) for s in ("train", "val", "test")}
    for v in vols:
        for x, y in zip(v.seg_x, v.seg_y):
            seg[v.split][0].append(x)
            seg[v.split][1].append(y)
    fig_eval = {"train": [], "val": [], "test": []}
    if a.figshare:
        pairs = D.figshare_pairs(Path(a.figshare), limit=40 if P else 0)
        fh = [dhash_canonical(D.load_gray(p)) for p, _ in pairs]
        gid = duplicate_groups(fh)
        for (ip, mp), g in zip(pairs, gid):
            s = group_split(f"fig:{fh[g]:016x}")
            x = letterbox(D.load_gray(ip), SEG_SIZE)[0]
            y = (letterbox(D.load_gray(mp), SEG_SIZE, order=0)[0] > 127).astype(np.uint8)
            seg[s][0].append(x[None])
            seg[s][1].append(y)
            fig_eval[s].append(len(seg[s][0]) - 1)
        metrics["figshare"] = {"pairs": len(pairs), "splits": {s: len(fig_eval[s]) for s in fig_eval}}
    log("segmenter slices", {s: len(seg[s][0]) for s in seg})
    sres = train_segmenter(seg["train"], seg["val"], epochs=1 if P else a.seg_epochs, max_steps=3 if P else 0, device=device)
    snet = sres["net"]

    # slice-level test metrics per BraTS sequence and on figshare
    seg_rep: Dict = {"architecture": "2D U-Net, 1.9 M parameters, one grey slice in", "history": sres["history"], "brats_slices": {}}
    bx = [(x, y) for v in vols if v.split == "test" for x, y in zip(v.seg_x, v.seg_y)]
    for c, name in enumerate(D.SEQUENCES):
        if not bx:
            break
        pr = seg_predict(snet, [x[c] for x, _ in bx], device) > 0.5
        tum = [dice(p, y) for p, (_, y) in zip(pr, bx) if y.any()]
        free = [p.mean() > 0.001 for p, (_, y) in zip(pr, bx) if not y.any()]
        seg_rep["brats_slices"][name] = {"dice_on_tumour_slices": summarise(tum),
                                         "false_alarm_rate_on_tumour_free_slices": float(np.mean(free)) if free else None,
                                         "tumour_free_slices": len(free)}
    if fig_eval["test"]:
        xs = [seg["test"][0][i][0] for i in fig_eval["test"]]
        ys = [seg["test"][1][i] for i in fig_eval["test"]]
        pr = seg_predict(snet, xs, device) > 0.5
        seg_rep["figshare_test"] = {"dice": summarise([dice(p, y) for p, y in zip(pr, ys)]),
                                    "note": "Image-level split: this set has no patient IDs, so neighbouring slices of one patient can fall on both sides."}

    # 3D: whole-volume tumour volume from slice-by-slice segmentation, test patients only
    vol3d = {}
    spacing = 240 / SEG_SIZE
    for s in ("FLAIR", "T2"):
        rows = []
        for v in vols:
            if v.full is None:
                continue
            pr = seg_predict(snet, list(v.full[s]), device) > 0.5
            est = pr.sum() * spacing * spacing / 1000.0
            true = v.full["mask"].sum() * spacing * spacing / 1000.0
            rows.append({"patient": v.patient_id, "dice3d": dice(pr, v.full["mask"]), "est_cc": est, "true_cc": true})
        if rows:
            est, true = np.array([r["est_cc"] for r in rows]), np.array([r["true_cc"] for r in rows])
            vol3d[s] = {"patients": len(rows), "dice3d": summarise([r["dice3d"] for r in rows]),
                        "volume_abs_error_cc": summarise(np.abs(est - true)),
                        "volume_rel_error": summarise(np.abs(est - true) / np.maximum(true, 1e-6)),
                        "volume_pearson_r": float(np.corrcoef(est, true)[0, 1]) if len(rows) > 2 else None,
                        "pairs": [[round(r["true_cc"], 2), round(r["est_cc"], 2)] for r in rows]}
    seg_rep["brats_3d_test"] = vol3d
    metrics["segmenter"] = seg_rep
    log("segmenter", {k: v["dice_on_tumour_slices"]["mean"] for k, v in seg_rep["brats_slices"].items()}, {k: v["dice3d"]["mean"] for k, v in vol3d.items()})

    # ---------------------------------------------------------------- survival
    surv = D.brats_tables(Path(a.brats))
    recs = [(v, surv[v.patient_id]) for v in vols if v.patient_id in surv]
    metrics["survival_matching"] = {"survival_rows": len(surv), "matched_volumes": len(recs)}
    log("survival rows", len(surv), "matched to volumes", len(recs))
    if len(recs) >= 8:
        X = design([r["age"] for _, r in recs], [v.wt_cc for v, _ in recs], [r["resection"] for _, r in recs])
        t = np.array([r["time"] for _, r in recs], float)
        e = np.array([r["event"] for _, r in recs], int)
        kw = dict(repeats=1, folds=2, boots=3) if P else {}
        srep = fit_and_report(X, t, e, **kw)
        srep["volume_source"] = "Whole-tumour volume from the expert segmentation (BraTS ground truth)."
        srep["ages"] = summarise([r["age"] for _, r in recs])
        srep["wt_volume_cc"] = summarise([v.wt_cc for v, _ in recs])
        write_json(out / "survival.json", {k: srep[k] for k in ("features", "model", "n", "events")})
        metrics["survival"] = {k: v for k, v in srep.items() if k != "model"}
        metrics["survival"]["hazard_ratio_per_sd"] = srep["model"]["hazard_ratio_per_sd"]
        log("survival c-index", {k: round(v["mean"], 3) for k, v in srep["cv_c_index"].items()})

    # ---------------------------------------------------------------- export
    import torch as _t
    from .models import ClassifierExport, SegmenterExport, count

    metrics["export"] = {
        "classifier": export_onnx(ClassifierExport(net, T), _t.rand(1, 1, CLS_SIZE, CLS_SIZE), out / "classifier.onnx", "image", ["probs", "cams"]),
        "segmenter": export_onnx(SegmenterExport(snet), _t.rand(1, 1, SEG_SIZE, SEG_SIZE), out / "segmenter.onnx", "slice", ["prob"]),
    }
    metrics["export"]["classifier"]["parameters"] = count(net)
    metrics["export"]["segmenter"]["parameters"] = count(snet)
    log("export", metrics["export"])

    # samples for the website: held-out test images only
    samples = []
    rng = np.random.default_rng(3)
    for k in range(4):
        pool = [i for i, it in enumerate(split["test"]) if it.label == k and it.source == "brain-mri-4class"]
        for i in rng.choice(pool, min(3, len(pool)), replace=False) if pool else []:
            name = f"test_{CLASSES[k]}_{len(samples)}.jpg"
            from PIL import Image
            Image.fromarray(imgs["test"][i]).save(out / "samples" / name, quality=90)
            samples.append({"file": name, "truth": CLASSES[k], "source": "Brain Tumor MRI Dataset (test split of this project)"})
    test_vols = [v for v in vols if v.full is not None]
    if test_vols:
        v = max(test_vols, key=lambda v: v.wt_cc)
        write_nifti_gz(out / "samples" / "brats_test_flair.nii.gz", v.full["FLAIR"], (spacing, spacing, 1.0))
        samples.append({"file": "brats_test_flair.nii.gz", "truth": "glioma", "kind": "volume",
                        "true_wt_cc": round(float(v.full["mask"].sum() * spacing * spacing / 1000), 1),
                        "source": "BraTS 2020 (test split of this project), FLAIR"})
    write_json(out / "samples" / "index.json", samples)
    metrics["minutes"] = round((time.time() - T0) / 60, 1)
    write_json(out / "screen_metrics.json", metrics)
    log("done; wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
