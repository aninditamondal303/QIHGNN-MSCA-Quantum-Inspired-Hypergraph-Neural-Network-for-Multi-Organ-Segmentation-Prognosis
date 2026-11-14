"""Kaggle kernel entry for the liver (CT) and cervix (colposcopy) models. kaggle_run.ps1 -Job organs.

probe: CPU. Inspects what the two datasets actually contain (mask levels, array shapes, annotation labels).
train: GPU. Trains, evaluates and exports both models (qihgnn.screen.organs).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

MODE = "__MODE__"
OUT = "/kaggle/working/output"
REPO = "/tmp/repo"
T0 = time.time()


def sh(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}   [{(time.time() - T0) / 60:.1f} min]", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")


def find_input(slug):
    for base in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*/*"):
        if os.path.basename(base.rstrip("/")) == slug:
            return base
    return ""


def restore_code():
    named = [d for d in glob.glob("/kaggle/input/**/qihgnn-msca-code", recursive=True) if os.path.isdir(d)]
    root, stage = named[0], "/tmp/code_stage"
    for cur, _, files in os.walk(root):
        for f in files:
            src = os.path.join(cur, f)
            rel = os.path.relpath(src, root).replace("\\", "/")
            if rel.endswith(".zip"):
                with zipfile.ZipFile(src) as z:
                    for info in z.infolist():
                        name = info.filename.replace("\\", "/")
                        if not name.endswith("/"):
                            dst = os.path.join(stage, os.path.dirname(rel), os.path.splitext(os.path.basename(rel))[0], name)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with z.open(info) as fi, open(dst, "wb") as fo:
                                shutil.copyfileobj(fi, fo)
                continue
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    found = sorted(glob.glob(f"{stage}/**/qihgnn/__init__.py", recursive=True), key=len)
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), f"{REPO}/ml", dirs_exist_ok=True)
    return f"{REPO}/ml"


def inspect(lits, cerv):
    import numpy as np
    from PIL import Image

    report = {}
    # ---- LiTS: which grey levels do the masks use, and how many volumes have a second (tumour) level?
    masks = sorted(glob.glob(f"{lits}/**/train_masks/**/*.jpg", recursive=True))
    print("LiTS mask files:", len(masks))
    by_vol = {}
    for p in masks:
        v = os.path.basename(p).split("_slice_")[0]
        by_vol.setdefault(v, []).append(p)
    hist = np.zeros(256, np.int64)
    vols = sorted(by_vol, key=lambda s: int(s.split("-")[1]))
    per_vol = {}
    rng = np.random.default_rng(0)
    for v in vols:
        ps = by_vol[v]
        pick = [ps[i] for i in rng.choice(len(ps), min(40, len(ps)), replace=False)]
        hv = np.zeros(256, np.int64)
        for p in pick:
            a = np.asarray(Image.open(p).convert("L"))
            hv += np.bincount(a.ravel(), minlength=256)
        hist += hv
        per_vol[v] = {"slices": len(ps), "frac_100_160": float(hv[100:160].sum() / max(1, hv[1:].sum())), "frac_200_256": float(hv[200:].sum() / max(1, hv[1:].sum()))}
    nz = hist.copy(); nz[0] = 0
    peaks = np.argsort(nz)[::-1][:12]
    print("LiTS mask grey-level histogram, top values:", [(int(k), int(nz[k])) for k in peaks])
    print("LiTS volumes:", len(vols), "first/last:", vols[:3], vols[-3:])
    two_level = [v for v, d in per_vol.items() if d["frac_100_160"] > 0.05 and d["frac_200_256"] > 0.01]
    print("volumes whose masks show both a mid (100-160) and a bright (200+) level:", len(two_level))
    imgs = sorted(glob.glob(f"{lits}/**/train_images/**/*.jpg", recursive=True))[:3]
    for p in imgs:
        im = Image.open(p); a = np.asarray(im.convert("L"))
        print("LiTS image", os.path.basename(p), im.size, im.mode, "min/max/mean", a.min(), a.max(), round(float(a.mean()), 1))
    report["lits"] = {"mask_hist_top": [(int(k), int(nz[k])) for k in peaks], "volumes": len(vols), "two_level_volumes": len(two_level)}

    # ---- cervical: arrays, masks, annotations
    for name in ("images_train.npy", "masks_train.npy", "images_test.npy", "masks_test.npy"):
        p = os.path.join(cerv, name)
        if os.path.exists(p):
            a = np.load(p, mmap_mode="r")
            sl = np.asarray(a[: min(8, len(a))])
            print(f"cervical {name}: shape {a.shape} dtype {a.dtype} sample min/max {sl.min()}/{sl.max()} unique(first 8) {np.unique(sl)[:12]}")
            report[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    js = sorted(glob.glob(f"{cerv}/**/*.json", recursive=True))
    labels = {}
    shapes = {}
    for p in js:
        try:
            d = json.load(open(p))
        except Exception as exc:
            print("bad json", p, exc)
            continue
        if p == js[0]:
            print("first JSON keys:", list(d.keys())[:20])
            print("first JSON (trimmed):", json.dumps({k: (v if not isinstance(v, (list, dict)) else str(v)[:300]) for k, v in d.items()})[:1500])
        for s in d.get("shapes", []) if isinstance(d, dict) else []:
            labels[s.get("label")] = labels.get(s.get("label"), 0) + 1
            shapes[s.get("shape_type")] = shapes.get(s.get("shape_type"), 0) + 1
    print("cervical JSON files:", len(js), "shape labels:", labels, "shape types:", shapes)
    report["cervical_labels"] = labels
    pngs = sorted(glob.glob(f"{cerv}/**/png/*.png", recursive=True))
    print("cervical PNGs:", len(pngs), "first names:", [os.path.basename(p) for p in pngs[:5]])
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(f"{OUT}/organs_inspection.json", "w"), indent=1)
    # small visual samples to look at
    for i, p in enumerate(pngs[:3]):
        Image.open(p).convert("RGB").resize((512, 384)).save(f"{OUT}/cervix_sample_{i}.jpg", quality=85)
    m = np.load(os.path.join(cerv, "masks_train.npy"), mmap_mode="r") if os.path.exists(os.path.join(cerv, "masks_train.npy")) else None
    x = np.load(os.path.join(cerv, "images_train.npy"), mmap_mode="r") if os.path.exists(os.path.join(cerv, "images_train.npy")) else None
    if m is not None and x is not None:
        for i in range(3):
            xi = np.asarray(x[i]); mi = np.asarray(m[i])
            xi8 = (255 * (xi - xi.min()) / max(1e-6, float(xi.max() - xi.min()))).astype(np.uint8)
            mi8 = (255 * (mi > (mi.max() / 2 if mi.max() > 0 else 0.5))).astype(np.uint8)
            Image.fromarray(xi8.squeeze() if xi8.ndim == 3 and xi8.shape[-1] == 1 else xi8).save(f"{OUT}/cervix_npy_img_{i}.png")
            Image.fromarray(mi8.squeeze() if mi8.ndim == 3 and mi8.shape[-1] == 1 else (mi8[..., 0] if mi8.ndim == 3 else mi8)).save(f"{OUT}/cervix_npy_mask_{i}.png")
    for v in [vols[len(vols) // 2]] if vols else []:
        ps = by_vol[v]
        ps.sort(key=lambda s: int(s.split("_slice_")[1].split(".")[0]))
        best = max(ps, key=lambda p: (np.asarray(Image.open(p).convert("L")) > 30).sum())
        Image.open(best).save(f"{OUT}/lits_mask_{v}.jpg")
        Image.open(best.replace("train_masks", "train_images")).save(f"{OUT}/lits_image_{v}.jpg")


ML = restore_code()
os.makedirs(OUT, exist_ok=True)
sh(f"{sys.executable} -m pip install -q onnx onnxruntime", check=False)
lits, cerv = find_input("lits-dataset"), find_input("cervical-cancer-segmentation")
print("lits:", lits, "| cervical:", cerv)
if MODE == "probe":
    inspect(lits, cerv)
    print("\nORGANS PROBE OK")
    sys.exit(0)
if MODE == "train":
    sh(f"{sys.executable} -c \"import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))\"")
sh(f"{sys.executable} -m qihgnn.screen.organs --lits '{lits}' --cervix '{cerv}' --out {OUT} {'--probe' if MODE == 'probe-train' else ''}", cwd=ML)
print(f"\nORGANS {MODE.upper()} OK in {(time.time() - T0) / 60:.1f} min")
