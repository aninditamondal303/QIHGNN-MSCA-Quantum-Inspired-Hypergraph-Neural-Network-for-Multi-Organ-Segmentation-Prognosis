"""Kaggle kernel entry for the screening models. scripts/kaggle_run.ps1 -Job screen replaces __MODE__.

probe: CPU. Prints each dataset's layout and runs the whole pipeline on a handful of cases.
train: GPU. Full data, full training, evaluation and ONNX export to /kaggle/working/output.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

MODE = "__MODE__"
OUT = "/kaggle/working/output"
REPO = "/tmp/repo"
T0 = time.time()


def sh(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}   [{(time.time() - T0) / 60:.1f} min]", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def tree(root, depth=3, per_dir=6):
    for cur, dirs, files in os.walk(root):
        level = cur[len(root):].count(os.sep)
        if level >= depth:
            dirs[:] = []
            continue
        dirs.sort()
        files.sort()
        print(f"{'  ' * level}{os.path.basename(cur) or cur}/  ({len(dirs)} dirs, {len(files)} files)")
        for f in files[:per_dir]:
            print(f"{'  ' * level}  {f}")
        dirs[:] = dirs[:per_dir]


def find_input(slug):
    for base in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*/*"):
        if os.path.basename(base.rstrip("/")) == slug:
            return base
    return ""


def restore_code():
    import zipfile
    named = [d for d in glob.glob("/kaggle/input/**/qihgnn-msca-code", recursive=True) if os.path.isdir(d)]
    if not named:
        tree("/kaggle/input", 4)
        raise SystemExit("code dataset not found")
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
    if not found:
        raise SystemExit("qihgnn package not found in the code dataset")
    shutil.copytree(os.path.dirname(os.path.dirname(found[0])), f"{REPO}/ml", dirs_exist_ok=True)
    return f"{REPO}/ml"


ML = restore_code()
os.makedirs(OUT, exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
sh(f"{sys.executable} -m pip install -q h5py onnx onnxruntime", check=False)
if MODE == "train":
    sh(f"{sys.executable} -c \"import torch; assert torch.cuda.is_available(), 'no GPU in this session'; print(torch.cuda.get_device_name(0))\"")

paths = {
    "mri4": find_input("brain-tumor-mri-dataset"),
    "external": find_input("brain-mri-images-for-brain-tumor-detection"),
    "figshare": find_input("brain-tumor-segmentation"),
    "brats": find_input("brats2020-training-data"),
    "ct": find_input("lits-dataset"),
    "photos": find_input("cervical-cancer-segmentation"),
}
for k, v in paths.items():
    print(f"\n==== {k}: {v or 'NOT ATTACHED'}")
    if v and MODE == "probe" and k != "brats":
        tree(v, 4)

args = " ".join(f"--{k} '{v}'" for k, v in paths.items() if v)
sh(f"{sys.executable} -m qihgnn.screen.train {args} --out {OUT} {'--probe' if MODE == 'probe' else ''}", cwd=ML)
print(f"\nSCREEN {MODE.upper()} OK in {(time.time() - T0) / 60:.1f} min")
