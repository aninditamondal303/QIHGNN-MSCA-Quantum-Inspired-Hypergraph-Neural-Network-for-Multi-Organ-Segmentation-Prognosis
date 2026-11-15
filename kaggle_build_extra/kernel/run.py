"""Kaggle kernel for the extra data: healthy livers (CHAOS CT) and graded cervical cells (SIPaKMeD).
kaggle_run.ps1 -Job extra -Mode probe (CPU: inspect) | train (GPU: liver with healthy livers + cell classifier).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

MODE = "probe-train"
OUT = "/kaggle/working/output"
REPO = "/tmp/repo"
T0 = time.time()


def sh(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}   [{(time.time() - T0) / 60:.1f} min]", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def tree(root, depth=3, per_dir=8):
    """Print a compact listing: up to `per_dir` entries per folder, with counts."""
    for cur, dirs, files in os.walk(root):
        level = cur[len(root):].count(os.sep)
        if level >= depth:
            dirs[:] = []
            continue
        dirs.sort()
        files.sort()
        ind = "  " * level
        print(f"{ind}{os.path.basename(cur) or cur}/  ({len(dirs)} dirs, {len(files)} files)")
        for f in files[:per_dir]:
            print(f"{ind}  {f}  {os.path.getsize(os.path.join(cur, f)) // 1024} KB")
        if len(files) > per_dir:
            print(f"{ind}  ... {len(files) - per_dir} more files")
        dirs[:] = dirs[:per_dir]


def find_input(slug):
    for base in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*/*"):
        if os.path.basename(base.rstrip("/")) == slug:
            return base
    return None


# ---------------------------------------------------------------- code
def restore_code():
    """Copy the qihgnn code dataset to REPO, whatever layout Kaggle gave it.

    Datasets uploaded from Windows can arrive as a folder, as a zip, or as flat files whose
    names contain backslashes ("qihgnn\\__init__.py"); all three are normalised here.
    """
    import zipfile
    marks = glob.glob("/kaggle/input/**/requirements.txt", recursive=True)
    named = [d for d in glob.glob("/kaggle/input/**/qihgnn-msca-code", recursive=True) if os.path.isdir(d)]
    roots = named or [os.path.dirname(m) for m in marks]
    if not roots:
        tree("/kaggle/input", depth=5)
        raise SystemExit("qihgnn code dataset not found under /kaggle/input")
    root = roots[0]
    stage = "/tmp/code_stage"
    print("code dataset at", root)
    tree(root, depth=4)
    for cur, _, files in os.walk(root):
        for f in files:
            srcf = os.path.join(cur, f)
            rel = os.path.relpath(srcf, root).replace("\\", "/")
            if rel.endswith(".zip"):
                base = os.path.dirname(rel)
                stem = os.path.splitext(os.path.basename(rel))[0]
                with zipfile.ZipFile(srcf) as z:
                    for info in z.infolist():
                        name = info.filename.replace("\\", "/")
                        if name.endswith("/"):
                            continue
                        dst = os.path.join(stage, base, stem, name)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with z.open(info) as fi, open(dst, "wb") as fo:
                            shutil.copyfileobj(fi, fo)
                continue
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(srcf, dst)
    found = sorted(glob.glob(f"{stage}/**/qihgnn/__init__.py", recursive=True), key=len)
    if not found:
        tree(stage, depth=5)
        raise SystemExit("qihgnn package not found after restoring the code dataset")
    # The folder holding qihgnn/ (and configs/) is the repo's ml/ folder, whatever it was called on Kaggle.
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), f"{REPO}/ml", dirs_exist_ok=True)
    return root, f"{REPO}/ml"




def probe():
    import numpy as np
    from PIL import Image

    sh(f"{sys.executable} -m pip install -q pydicom", check=False)
    import pydicom

    chaos = find_input("chaos-combined-ct-mr-healthy-abdominal-organ")
    sip = find_input("cervical-cancer-largest-dataset-sipakmed")
    print("chaos:", chaos, "| sipakmed:", sip)
    if chaos:
        tree(chaos, depth=6, per_dir=4)
        dcm = sorted(glob.glob(f"{chaos}/**/CT/**/*.dcm", recursive=True))
        print("CT dicom files:", len(dcm))
        pats = sorted({p.split("/CT/")[1].split("/")[0] for p in dcm})
        print("CT patients:", len(pats), pats[:50])
        if dcm:
            d = pydicom.dcmread(dcm[0])
            a = d.pixel_array
            print("dicom", dcm[0], a.shape, a.dtype, a.min(), a.max(), "slope", getattr(d, "RescaleSlope", None), "inter", getattr(d, "RescaleIntercept", None),
                  "spacing", getattr(d, "PixelSpacing", None), "thick", getattr(d, "SliceThickness", None), "pos", getattr(d, "ImagePositionPatient", None),
                  "instance", getattr(d, "InstanceNumber", None))
        pngs = sorted(glob.glob(f"{chaos}/**/CT/**/*.png", recursive=True))
        print("CT pngs:", len(pngs), pngs[:3])
        if pngs:
            m = np.asarray(Image.open(pngs[len(pngs) // 2]))
            print("mask", pngs[len(pngs) // 2], m.shape, m.dtype, np.unique(m)[:10])
    if sip:
        tree(sip, depth=5, per_dir=4)
        exts = {}
        for _cur, _, fs in os.walk(sip):
            for f in fs:
                e = os.path.splitext(f)[1].lower()
                exts[e] = exts.get(e, 0) + 1
        print("sipakmed extensions:", exts)
        for d in sorted(glob.glob(f"{sip}/**/", recursive=True))[:40]:
            n = len([f for f in os.listdir(d) if f.lower().endswith((".bmp", ".jpg", ".png"))])
            if n:
                print("  dir", d, n)
        imgs = sorted(glob.glob(f"{sip}/**/*.bmp", recursive=True))[:3]
        for p in imgs:
            print("  img", p, Image.open(p).size)
        dats = sorted(glob.glob(f"{sip}/**/*.dat", recursive=True))
        print("dat files:", len(dats), dats[:3])
        if dats:
            print(open(dats[0]).read()[:400])


if MODE == "probe":
    probe()
    print("EXTRA PROBE OK")
    sys.exit(0)

src, ML = restore_code()
print("code from", src, "| mode", MODE, "| python", sys.version.split()[0])
sh(f"{sys.executable} -m pip install -q onnx onnxruntime nibabel pydicom", check=False)
if MODE == "train":
    sh(f"{sys.executable} -c \"import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))\"")
os.makedirs(OUT, exist_ok=True)
probe_flag = "--probe" if MODE == "probe-train" else ""
sip = find_input("cervical-cancer-largest-dataset-sipakmed")
chaos = find_input("chaos-combined-ct-mr-healthy-abdominal-organ")
nifti = ",".join(r for r in (find_input("liver-tumor-segmentation"), find_input("liver-tumor-segmentation-part-2")) if r)
print("sipakmed", sip, "| chaos", chaos, "| lits", nifti)

# 1. cervical cell grading (minutes)
sh(f"{sys.executable} -m qihgnn.screen.cells --sipakmed '{sip}' --out {OUT}/cells {probe_flag}", cwd=ML)
# 2. liver: original LiTS CT + CHAOS healthy livers (about 2.5 h on a T4)
sh(f"{sys.executable} -m qihgnn.screen.liver_cascade --nifti '{nifti}' --chaos '{chaos}' --out {OUT}/liver {probe_flag}", cwd=ML)
print(f"\nEXTRA {MODE.upper()} OK in {(time.time() - T0) / 60:.1f} min")

