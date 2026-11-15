"""Liver tumour segmentation (CT) and cervical cell analysis (microscope images), trained, evaluated and exported.

  python -m qihgnn.screen.organs --lits <dir> --cervix <dir> --out <dir> [--probe]

Liver: LiTS slices keep their original labels (0 background, 1 liver, 2 tumour). A 3-class 2D U-Net outlines
liver and tumour; on whole CT volumes it gives liver and tumour size, tumour burden and lesion count, and decides
per patient whether a tumour is present. Splits are per patient.

Cervix: the dataset is Papanicolaou-stained cytology (microscope images) with every cell and nucleus outlined.
A 3-class U-Net (background, cytoplasm, nucleus) outlines them; cells are separated by growing each nucleus
through its cell, and each cell's nucleus-to-cytoplasm ratio is measured. There are no diagnosis labels, so this
is a measurement tool, not a cancer classifier. Splits are per slide folder.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..data.listing import list_files
from .common import binary_report, dice, group_split, letterbox, summarise, write_json

T0 = time.time()
LIVER = 256
CERV_W, CERV_H = 512, 384
MIN_NUCLEUS_PX = 12
SLICE_RE = re.compile(r"volume-(\d+)_slice_(\d+)\.(jpg|png)$", re.I)


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)


# ============================================================================ liver data
def lits_index(root: Path) -> Dict[int, Dict[int, List[Path]]]:
    """{volume: {slice: [image, mask]}} from the mirror's train_images / train_masks folders."""
    imgs, masks = defaultdict(dict), defaultdict(dict)
    for p in list_files(root, (".jpg", ".png")):
        m = SLICE_RE.search(p.name)
        if not m:
            continue
        v, s = int(m.group(1)), int(m.group(2))
        (masks if "mask" in p.parent.as_posix().lower() else imgs)[v][s] = p
    return {v: {s: [imgs[v][s], masks[v][s]] for s in imgs[v] if s in masks[v]} for v in imgs}


def _gray(p: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(p) as im:
        return np.asarray(im.convert("L"))


def clean_liver_mask(m: np.ndarray) -> np.ndarray:
    """0 background, 1 liver, 2 tumour. JPEG noise produces the odd 3 next to tumour edges."""
    out = np.zeros(m.shape, np.uint8)
    out[m == 1] = 1
    out[m >= 2] = 2
    out[m > 10] = 0  # anything far from the label range is not a label
    return out


def load_liver(index, volumes: List[int], all_slices: bool, rng: np.random.Generator, workers: int = 16):
    """Returns per volume: (images (n, 256, 256) uint8, masks (n, 256, 256) uint8, slice numbers)."""
    out = {}

    def read(pair):
        img = letterbox(_gray(pair[0]), LIVER)[0]
        msk = letterbox(clean_liver_mask(_gray(pair[1])), LIVER, order=0)[0]
        return img, msk

    with ThreadPoolExecutor(workers) as pool:
        for v in volumes:
            sl = sorted(index[v])
            if not all_slices:
                # masks first (small files) to choose slices: liver slices every 2nd plus 20% liver-free slices
                ms = list(pool.map(lambda s, v=v: clean_liver_mask(_gray(index[v][s][1])), sl))
                has = np.array([m.any() for m in ms])
                pos = [s for s, h in zip(sl, has) if h][::2]
                neg_pool = [s for s, h in zip(sl, has) if not h]
                neg = list(rng.choice(neg_pool, min(len(neg_pool), max(2, len(pos) // 5)), replace=False)) if neg_pool else []
                sl = sorted(pos + [int(x) for x in neg])
            pairs = list(pool.map(read, [index[v][s] for s in sl]))
            out[v] = (np.stack([a for a, _ in pairs]), np.stack([b for _, b in pairs]), sl)
    return out


# ============================================================================ cervix data
def cervix_items(root: Path) -> List[Dict]:
    """PNG images with labelme JSON polygons ('cell', 'kernel' = nucleus). Group = slide folder in imagePath."""
    pngs = {p.stem: p for p in list_files(root, (".png",)) if "png" in p.parent.as_posix().lower()}
    items = []
    for jp in list_files(root, (".json",)):
        if jp.stem not in pngs:
            continue
        d = json.loads(jp.read_text(errors="replace"))
        ip = str(d.get("imagePath", "")).replace("\\", "/").split("/")
        group = ip[-2] if len(ip) >= 2 and ip[-2] not in ("..", "") else jp.stem
        items.append({"png": pngs[jp.stem], "json": jp, "group": f"slide:{group}", "w": d.get("imageWidth"), "h": d.get("imageHeight"),
                      "shapes": [(s.get("label"), s.get("points")) for s in d.get("shapes", []) if s.get("points")]})
    return items


def rasterise_cervix(item: Dict) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """RGB (384, 512, 3) uint8, class map (0 bg, 1 cytoplasm, 2 nucleus) and per-cell ground-truth N/C ratios."""
    from PIL import Image, ImageDraw

    with Image.open(item["png"]) as im:
        rgb = im.convert("RGB")
        W, H = rgb.size
    sx, sy = CERV_W / W, CERV_H / H
    rgb = np.asarray(rgb.resize((CERV_W, CERV_H), Image.BILINEAR))
    cell_img, nuc_img = Image.new("L", (CERV_W, CERV_H), 0), Image.new("L", (CERV_W, CERV_H), 0)
    dc, dn = ImageDraw.Draw(cell_img), ImageDraw.Draw(nuc_img)
    cells, nuclei = [], []
    for label, pts in item["shapes"]:
        poly = [(x * sx, y * sy) for x, y in pts]
        if len(poly) < 3:
            continue
        if label == "cell":
            dc.polygon(poly, fill=255)
            cells.append(poly)
        elif label == "kernel":
            dn.polygon(poly, fill=255)
            nuclei.append(poly)
    cell, nuc = np.asarray(cell_img) > 0, np.asarray(nuc_img) > 0
    cls = np.zeros((CERV_H, CERV_W), np.uint8)
    cls[cell | nuc] = 1
    cls[nuc] = 2
    # Ground-truth N/C per annotated cell: nucleus polygons whose centroid lies inside the cell polygon.
    ratios = []
    for poly in cells:
        ci = Image.new("L", (CERV_W, CERV_H), 0)
        ImageDraw.Draw(ci).polygon(poly, fill=1)
        cm = np.asarray(ci) > 0
        n_area = (cm & nuc).sum()
        c_area = cm.sum() - n_area
        if c_area > 0 and n_area > 0:
            ratios.append(float(n_area / c_area))
    return rgb, cls, ratios


def cells_from_classes(cls: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Separate cells: label nuclei (8-connected, >= MIN_NUCLEUS_PX), then grow every nucleus through cell pixels by
    breadth-first search (4-connected), so each cell pixel joins the nucleus it is reached from first.
    Mirrors web/src/lib/organs.ts exactly. Returns the cell label map and per-cell measurements."""
    from scipy import ndimage

    h, w = cls.shape
    nuc = cls == 2
    lab, n = ndimage.label(nuc, structure=np.ones((3, 3)))
    sizes = ndimage.sum(nuc, lab, index=np.arange(1, n + 1)) if n else np.array([])
    keep = {i + 1 for i, s in enumerate(sizes) if s >= MIN_NUCLEUS_PX}
    owner = np.zeros((h, w), np.int32)
    q = deque()
    ys, xs = np.nonzero(nuc)
    for y, x in zip(ys, xs):
        if lab[y, x] in keep:
            owner[y, x] = lab[y, x]
            q.append((y, x))
    cellpx = cls > 0
    while q:
        y, x = q.popleft()
        o = owner[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and cellpx[yy, xx] and owner[yy, xx] == 0:
                owner[yy, xx] = o
                q.append((yy, xx))
    cells = []
    for k in sorted(keep):
        m = owner == k
        n_area = int((m & nuc).sum())
        c_area = int(m.sum()) - n_area
        if c_area <= 0:
            continue
        yy, xx = np.nonzero(m)
        cells.append({"id": int(k), "nucleus_px": n_area, "cytoplasm_px": c_area, "nc": n_area / c_area,
                      "cx": float(xx.mean()), "cy": float(yy.mean())})
    return owner, cells


# ============================================================================ training
def _torch():
    import torch
    from torch.utils.data import DataLoader, Dataset
    return torch, DataLoader, Dataset


def liver_dataset(imgs: List[np.ndarray], masks: List[np.ndarray], train: bool):
    torch, _, Dataset = _torch()
    from .train import augment_gray

    class DS(Dataset):
        def __len__(self):
            return len(imgs)

        def __getitem__(self, i):
            x, y = imgs[i], masks[i]
            if train:
                rng = np.random.default_rng()
                g, y = augment_gray(x, rng, LIVER, y)
                if rng.random() < 0.5:  # scans are stored either way up; uploads may be displayed either way
                    g, y = np.ascontiguousarray(g[::-1]), np.ascontiguousarray(y[::-1])
            else:
                g = x.astype(np.float32) / 255.0
            return torch.from_numpy(g[None].astype(np.float32)), torch.from_numpy(y.astype(np.int64))

    return DS()


def cervix_dataset(imgs: List[np.ndarray], masks: List[np.ndarray], train: bool, crop: int = 256):
    torch, _, Dataset = _torch()

    class DS(Dataset):
        def __len__(self):
            return len(imgs) * (6 if train else 1)

        def __getitem__(self, i):
            i = i % len(imgs)
            x, y = imgs[i].astype(np.float32) / 255.0, masks[i]
            if train:
                rng = np.random.default_rng()
                y0, x0 = int(rng.integers(0, CERV_H - crop + 1)), int(rng.integers(0, CERV_W - crop + 1))
                x, y = x[y0:y0 + crop, x0:x0 + crop], y[y0:y0 + crop, x0:x0 + crop]
                k = int(rng.integers(0, 4))
                x, y = np.rot90(x, k), np.rot90(y, k)
                if rng.random() < 0.5:
                    x, y = x[:, ::-1], y[:, ::-1]
                # stain variation: per-channel gain and offset, brightness and contrast
                x = x * rng.uniform(0.85, 1.15, 3) + rng.uniform(-0.05, 0.05, 3)
                x = (x - x.mean()) * rng.uniform(0.85, 1.15) + x.mean()
                x = np.clip(x, 0, 1)
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32)), torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))

    return DS()


def train_multiclass(net, train_ds, val_fn, epochs: int, max_steps: int, device: str, batch: int, lr: float, class_weights, name: str):
    torch, DataLoader, _ = _torch()
    net = net.to(device)
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4 if device == "cuda" else 0, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    total = max(1, epochs * (min(len(dl), max_steps) if max_steps else len(dl)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        net.train()
        run = []
        for step, (x, y) in enumerate(dl):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            z = net(x)
            p = torch.softmax(z, 1)
            oh = torch.nn.functional.one_hot(y, z.shape[1]).permute(0, 3, 1, 2).float()
            inter = (p * oh).sum((0, 2, 3))
            dsc = 1 - (2 * inter + 1) / (p.sum((0, 2, 3)) + oh.sum((0, 2, 3)) + 1)
            loss = ce(z, y) + dsc[1:].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run.append(float(loss.detach()))
        score = val_fn(net)
        hist.append({"epoch": ep, "train_loss": float(np.mean(run)) if run else None, "val_score": score})
        log(f"{name} epoch {ep}: loss {hist[-1]['train_loss']:.4f}  val {score:.4f}")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, hist


def predict(net, x: np.ndarray, device: str, batch: int = 32) -> np.ndarray:
    """x: (n, C, H, W) float32 -> argmax classes (n, H, W) and class probabilities."""
    import torch

    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(torch.softmax(net(torch.from_numpy(x[i:i + batch]).to(device)), 1).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0,) + x.shape[1:])


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lits", required=True)
    ap.add_argument("--cervix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--liver-epochs", type=int, default=30)
    ap.add_argument("--cervix-epochs", type=int, default=60)
    a = ap.parse_args()
    import torch
    from scipy import ndimage

    from .models import SoftmaxExport, UNet2D, count
    from .train import export_onnx, write_nifti_gz

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    P = a.probe
    rng = np.random.default_rng(2026)
    M: Dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "device": device, "probe": P}

    # ------------------------------------------------------------------ liver
    index = lits_index(Path(a.lits))
    vols = sorted(index)
    split = {v: group_split(f"lits:{v}") for v in vols}
    if P:
        vols = [v for v in vols if split[v] == "train"][:3] + [v for v in vols if split[v] == "val"][:1] + [v for v in vols if split[v] == "test"][:2]
    M["liver_data"] = {"volumes": len(index), "slices": sum(len(s) for s in index.values()),
                       "splits": {k: sum(split[v] == k for v in index) for k in ("train", "val", "test")}}
    log("LiTS", M["liver_data"])
    tr = load_liver(index, [v for v in vols if split[v] == "train"], False, rng)
    va = load_liver(index, [v for v in vols if split[v] == "val"], False, rng)
    te = load_liver(index, [v for v in vols if split[v] == "test"], True, rng)
    log("liver slices loaded", {k: sum(len(x[2]) for x in d.values()) for k, d in (("train", tr), ("val", va), ("test", te))})
    tri = [im for d in tr.values() for im in d[0]]
    trm = [m for d in tr.values() for m in d[1]]
    vx = np.concatenate([d[0] for d in va.values()])[:, None].astype(np.float32) / 255.0
    vy = np.concatenate([d[1] for d in va.values()])

    def liver_val(net):
        pred = predict(net, vx, device).argmax(1)
        dl = [dice(p == 1, y == 1) for p, y in zip(pred, vy) if (y == 1).any()]
        dt = [dice(p == 2, y == 2) for p, y in zip(pred, vy) if (y == 2).any()]
        return float(np.mean(dl or [0]) * 0.5 + np.mean(dt or [0]) * 0.5)

    lnet, lhist = train_multiclass(UNet2D(base=16, in_ch=1, out_ch=3), liver_dataset(tri, trm, True), liver_val,
                                   epochs=1 if P else a.liver_epochs, max_steps=3 if P else 0, device=device, batch=24, lr=2e-3,
                                   class_weights=[0.5, 1.0, 3.0], name="liver")

    # validation patients: choose the tumour-burden threshold that best separates patients with and without tumour
    def patient_stats(data):
        rows = []
        for v, (imgs, msks, _sl) in data.items():
            pr = predict(lnet, imgs[:, None].astype(np.float32) / 255.0, device).argmax(1)
            liver_t, tum_t = (msks >= 1).sum(), (msks == 2).sum()
            liver_p, tum_p = (pr >= 1).sum(), (pr == 2).sum()
            lt, nt = ndimage.label(msks == 2)
            lp, npred = ndimage.label(pr == 2)
            sizes = ndimage.sum(pr == 2, lp, index=np.arange(1, npred + 1)) if npred else np.array([])
            rows.append({"volume": int(v), "dice_liver": dice(pr >= 1, msks >= 1), "dice_tumour": dice(pr == 2, msks == 2) if tum_t else None,
                         "has_tumour": int(tum_t > 0), "burden_true": float(tum_t / max(1, liver_t)), "burden_pred": float(tum_p / max(1, liver_p)),
                         "lesions_true": int(nt), "lesions_pred": int((sizes >= 30).sum()), "slices": int(len(imgs))})
        return rows

    val_rows = patient_stats(va)
    cands = sorted({0.0} | {r["burden_pred"] for r in val_rows})
    def youden(t, rows):
        tp = sum(r["burden_pred"] > t and r["has_tumour"] for r in rows); fn = sum(r["burden_pred"] <= t and r["has_tumour"] for r in rows)
        tn = sum(r["burden_pred"] <= t and not r["has_tumour"] for r in rows); fp = sum(r["burden_pred"] > t and not r["has_tumour"] for r in rows)
        return tp / max(1, tp + fn) + tn / max(1, tn + fp)
    thr = max(cands, key=lambda t: youden(t, val_rows)) if val_rows else 0.0
    test_rows = patient_stats(te)
    # slice-level metrics on test patients
    sl_l, sl_t = [], []
    for imgs, msks, _ in te.values():
        pr = predict(lnet, imgs[:, None].astype(np.float32) / 255.0, device).argmax(1)
        sl_l += [dice(p >= 1, y >= 1) for p, y in zip(pr, msks) if (y >= 1).any()]
        sl_t += [dice(p == 2, y == 2) for p, y in zip(pr, msks) if (y == 2).any()]
    det = binary_report([r["burden_pred"] for r in test_rows], [r["has_tumour"] for r in test_rows], threshold=thr + 1e-12) if test_rows else None
    bt = np.array([r["burden_true"] for r in test_rows]); bp = np.array([r["burden_pred"] for r in test_rows])
    M["liver"] = {
        "architecture": "2D U-Net, 3 classes (background, liver, tumour), one grey CT slice in", "history": lhist,
        "parameters": count(lnet), "patient_threshold_burden": thr,
        "test_patients": len(test_rows), "patients_with_tumour": int(sum(r["has_tumour"] for r in test_rows)),
        "slice_dice_liver": summarise(sl_l), "slice_dice_tumour": summarise(sl_t),
        "volume_dice_liver": summarise([r["dice_liver"] for r in test_rows]),
        "volume_dice_tumour": summarise([r["dice_tumour"] for r in test_rows if r["dice_tumour"] is not None]),
        "patient_detection": det,
        "burden_pearson_r": float(np.corrcoef(bt, bp)[0, 1]) if len(bt) > 2 else None,
        "burden_pairs": [[round(x, 4), round(y, 4)] for x, y in zip(bt, bp)],
        "lesion_count_pairs": [[r["lesions_true"], r["lesions_pred"]] for r in test_rows],
        "note": "Slice spacing is not stored in this mirror, so sizes are reported relative to the liver; NIfTI uploads use their own voxel size.",
    }
    log("liver", {k: M["liver"][k]["mean"] for k in ("slice_dice_liver", "slice_dice_tumour", "volume_dice_liver", "volume_dice_tumour")}, "detection", det and (det["sensitivity"], det["specificity"]))

    # ------------------------------------------------------------------ cervix
    items = cervix_items(Path(a.cervix))
    if P:
        items = items[:24]
    cs = {"train": [], "val": [], "test": []}
    with ThreadPoolExecutor(8) as pool:
        rast = list(pool.map(rasterise_cervix, items))
    for it, (rgb, cls, ratios) in zip(items, rast):
        cs[group_split(it["group"])].append((rgb, cls, ratios, it))
    groups = sorted({it["group"] for it in items})
    M["cervix_data"] = {"images": len(items), "slide_groups": len(groups), "splits": {k: len(v) for k, v in cs.items()},
                        "cells_annotated": int(sum(sum(1 for lab, _ in it["shapes"] if lab == "cell") for it in items)),
                        "nuclei_annotated": int(sum(sum(1 for lab, _ in it["shapes"] if lab == "kernel") for it in items))}
    log("cervix", M["cervix_data"])
    if not cs["val"]:
        cs["val"] = cs["train"][-2:]
    cvx = np.stack([c[0] for c in cs["val"]]).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    cvy = np.stack([c[1] for c in cs["val"]])

    def cervix_val(net):
        pred = predict(net, cvx, device, batch=4).argmax(1)
        return float(np.mean([dice(p == 2, y == 2) for p, y in zip(pred, cvy)]) * 0.5 + np.mean([dice(p > 0, y > 0) for p, y in zip(pred, cvy)]) * 0.5)

    cnet, chist = train_multiclass(UNet2D(base=16, in_ch=3, out_ch=3), cervix_dataset([c[0] for c in cs["train"]], [c[1] for c in cs["train"]], True),
                                   cervix_val, epochs=1 if P else a.cervix_epochs, max_steps=3 if P else 0, device=device, batch=16, lr=2e-3,
                                   class_weights=[0.5, 1.0, 2.0], name="cervix")
    rows = []
    gt_all = [r for c in cs["train"] for r in c[2]]
    for rgb, cls, ratios, it in cs["test"]:
        pr = predict(cnet, rgb[None].astype(np.float32).transpose(0, 3, 1, 2) / 255.0, device).argmax(1)[0].astype(np.uint8)
        _, cells = cells_from_classes(pr)
        rows.append({"dice_nucleus": dice(pr == 2, cls == 2), "dice_cell": dice(pr > 0, cls > 0),
                     "nuclei_true": sum(1 for lab, _ in it["shapes"] if lab == "kernel"), "nuclei_pred": len(cells),
                     "nc_true": float(np.median(ratios)) if ratios else None, "nc_pred": float(np.median([c["nc"] for c in cells])) if cells else None})
    nt = np.array([r["nuclei_true"] for r in rows], float); npd = np.array([r["nuclei_pred"] for r in rows], float)
    pairs_nc = [(r["nc_true"], r["nc_pred"]) for r in rows if r["nc_true"] is not None and r["nc_pred"] is not None]
    M["cervix"] = {
        "architecture": "2D U-Net, 3 classes (background, cytoplasm, nucleus), RGB microscope image in", "history": chist, "parameters": count(cnet),
        "test_images": len(rows), "dice_nucleus": summarise([r["dice_nucleus"] for r in rows]), "dice_cell": summarise([r["dice_cell"] for r in rows]),
        "nucleus_count_abs_error": summarise(np.abs(nt - npd)), "nucleus_count_pearson_r": float(np.corrcoef(nt, npd)[0, 1]) if len(rows) > 2 else None,
        "nucleus_count_pairs": [[int(x), int(y)] for x, y in zip(nt, npd)],
        "median_nc_abs_error": summarise([abs(x - y) for x, y in pairs_nc]),
        "median_nc_pearson_r": float(np.corrcoef(*zip(*pairs_nc))[0, 1]) if len(pairs_nc) > 2 else None,
        "training_nc_distribution": summarise(gt_all),
        "unusual_nc_threshold": float(np.percentile(gt_all, 99)) if gt_all else None,
        "note": "No diagnosis labels: cells with an unusually high nucleus-to-cytoplasm ratio are flagged for review, not classified.",
    }
    log("cervix", {k: M["cervix"][k]["mean"] for k in ("dice_nucleus", "dice_cell", "nucleus_count_abs_error")}, "N/C r", M["cervix"]["median_nc_pearson_r"])

    # ------------------------------------------------------------------ export and samples
    M["export"] = {
        "liver": export_onnx(SoftmaxExport(lnet), torch.rand(1, 1, LIVER, LIVER), out / "liver.onnx", "slice", ["prob"]),
        "cervix": export_onnx(SoftmaxExport(cnet), torch.rand(1, 3, CERV_H, CERV_W), out / "cervix.onnx", "image", ["prob"]),
    }
    from PIL import Image

    samples = []
    for v, (imgs, msks, sl) in list(te.items())[:2]:
        tum = [i for i in range(len(msks)) if (msks[i] == 2).sum() > 200]
        clean = [i for i in range(len(msks)) if (msks[i] == 1).sum() > 2000 and not (msks[i] == 2).any()]
        for i, truth in [(tum[len(tum) // 2], "liver_tumour") if tum else (None, None), (clean[len(clean) // 2], "liver_no_tumour") if clean else (None, None)]:
            if i is not None:
                name = f"liver_{truth}_{v}_{sl[i]}.png"
                Image.fromarray(imgs[i]).save(out / "samples" / name)
                samples.append({"file": name, "truth": truth, "kind": "liver", "source": "LiTS (test patient of this project)"})
    if te:
        v = max(te, key=lambda k: (te[k][1] == 2).sum())
        write_nifti_gz(out / "samples" / f"liver_ct_volume_{v}.nii.gz", te[v][0], (1.0, 1.0, 1.0))
        samples.append({"file": f"liver_ct_volume_{v}.nii.gz", "truth": "liver_tumour", "kind": "liver-volume",
                        "source": "LiTS (test patient of this project); voxel size not stored in this mirror"})
    for rgb, _cls, _ratios, it in cs["test"][:4]:
        name = f"cervical_cells_{it['png'].stem}.jpg"
        Image.fromarray(rgb).save(out / "samples" / name, quality=92)
        samples.append({"file": name, "truth": "cervical_cells", "kind": "cervix", "source": "Cervical cytology (test slide of this project)",
                        "nuclei_annotated": sum(1 for lab, _ in it["shapes"] if lab == "kernel")})
    write_json(out / "samples" / "organs_index.json", samples)
    M["minutes"] = round((time.time() - T0) / 60, 1)
    write_json(out / "organs_metrics.json", M)
    log("done", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
