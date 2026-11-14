"""Kaggle kernel entry point. scripts/kaggle_run.ps1 replaces __MODE__ with "probe" or "train".

probe: CPU only. Prints the real layout of each attached dataset, runs the inspection, and
       preprocesses two cases per admitted dataset, so loader problems show up in minutes.
train: GPU. Inspection, full preprocessing, full model + matched baseline training,
       evaluation of both on the held-out test split, and export of the results.

The repository is copied to /tmp so the multi-GB cache is not saved as kernel output;
only /kaggle/working/output is kept.
"""
import glob
import json
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


src, ML = restore_code()
os.makedirs(f"{REPO}/results", exist_ok=True)
print("code from", src, "| mode", MODE, "| python", sys.version.split()[0])
sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
sh(f"{sys.executable} -m pip install -q nibabel h5py", check=False)

# ---------------------------------------------------------------- data
raw = f"{REPO}/data/raw"
os.makedirs(raw, exist_ok=True)
for key, slug in [("brats2020", "brats2020-training-data"), ("lits", "lits-dataset"), ("cervical", "cervical-cancer-segmentation")]:
    path = find_input(slug)
    print(f"\n==== {key}: {path}")
    if path:
        os.symlink(path, f"{raw}/{key}")
        if MODE == "probe":
            tree(path, depth=4)

sh(f"{sys.executable} -m qihgnn.data.inspection", cwd=ML)
with open(f"{REPO}/data/inspection_report.json") as f:
    print(json.dumps(json.load(f)["decisions"], indent=1))

os.makedirs(OUT, exist_ok=True)
shutil.copy(f"{REPO}/data/inspection_report.json", OUT)

if MODE == "probe":
    sh(f"{sys.executable} -m qihgnn.data.cache --limit 8", cwd=ML)
    with open(f"{REPO}/data/cache/index.json") as f:
        idx = json.load(f)
    print(json.dumps({"counts": idx["counts"], "cases": [{k: c[k] for k in ("patient_id", "source", "split", "has_survival", "voxels")} for c in idx["cases"]]}, indent=1))
    shutil.copy(f"{REPO}/data/cache/index.json", f"{OUT}/probe_index.json")
    # Rehearsal of the real-data path on CPU: two optimiser steps, then evaluation of that checkpoint.
    rehearsal = {"model": {}, "train": {"dataset": "cache", "epochs": 1, "warmup_epochs": 0, "max_steps_per_epoch": 2,
                                        "grad_accumulation": 1, "num_workers": 0, "amp": False}}
    json.dump(rehearsal, open(f"{ML}/configs/_rehearsal.json", "w"))
    sh(f"{sys.executable} -m qihgnn.train --config configs/_rehearsal.json --device cpu", cwd=ML)
    run = open(f"{REPO}/runs/LATEST").read().strip()
    split = "test" if idx["counts"]["test"] else "val"
    sh(f"{sys.executable} -m qihgnn.evaluate --checkpoint {REPO}/runs/{run}/last.pt --label rehearsal --split {split} --limit 2", cwd=ML)
    print("\nPROBE OK")
    sys.exit(0)

# ---------------------------------------------------------------- train
# One forward/backward pass on the GPU first, so an out-of-memory problem costs seconds, not the
# hour of preprocessing that follows.
GPU_CHECK = """
import torch
from qihgnn.config import ModelConfig
from qihgnn.model import QIHGNNMSCA
assert torch.cuda.is_available(), "no GPU in this session"
m = QIHGNNMSCA(ModelConfig()).cuda()
x = torch.randn(2, 4, 128, 128, 64, device="cuda")
c = torch.zeros(2, 5, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    out = m(x, c)
sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.is_floating_point()).backward()
print("GPU check OK:", torch.cuda.get_device_name(0), round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB peak")
"""
open(f"{ML}/_gpu_check.py", "w").write(GPU_CHECK)
sh(f"{sys.executable} _gpu_check.py", cwd=ML)

sh(f"{sys.executable} -m qihgnn.data.cache --workers 8", cwd=ML)  # I/O-bound: more cases in flight than CPUs
shutil.copy(f"{REPO}/data/cache/index.json", f"{OUT}/cache_index.json")
sh(f"{sys.executable} -m qihgnn.verify", cwd=ML)

# Kaggle stops a session at 12 h. Whatever preprocessing left, minus a margin for evaluation and
# saving, is split between the full model (55%) and the baseline (45%).
SESSION_HOURS, MARGIN_HOURS = 12.0, 1.25
left = SESSION_HOURS - MARGIN_HOURS - (time.time() - T0) / 3600
budget = {"full": round(max(0.5, 0.55 * left), 2), "baseline": round(max(0.4, 0.45 * left), 2)}
print("training budget (h):", budget, flush=True)

runs = {}
for label, base_cfg in [("full", "configs/kaggle_full.json"), ("baseline", "configs/kaggle_baseline.json")]:
    conf = json.load(open(f"{ML}/{base_cfg}"))
    conf["train"]["time_limit_hours"] = min(conf["train"].get("time_limit_hours") or 99, budget[label])
    cfg = f"configs/_kaggle_{label}_budgeted.json"
    json.dump(conf, open(f"{ML}/{cfg}", "w"), indent=1)
    sh(f"{sys.executable} -m qihgnn.train --config {cfg}", cwd=ML)
    runs[label] = open(f"{REPO}/runs/LATEST").read().strip()
    shutil.copy(f"{REPO}/results/training_history.json", f"{OUT}/training_history_{label}.json")

cks = " ".join(f"{REPO}/runs/{runs[l]}/best.pt" for l in ("full", "baseline"))
sh(f"{sys.executable} -m qihgnn.evaluate --checkpoint {cks} --label full baseline", cwd=ML)

shutil.copytree(f"{REPO}/results", f"{OUT}/results", dirs_exist_ok=True)
shutil.copy(f"{OUT}/training_history_full.json", f"{OUT}/results/training_history.json")
for label, run in runs.items():
    dst = f"{OUT}/runs/{run}"
    os.makedirs(dst, exist_ok=True)
    for name in ("config.json", "history.csv", "best.pt"):
        p = f"{REPO}/runs/{run}/{name}"
        if os.path.exists(p) and (name != "best.pt" or label == "full"):
            shutil.copy(p, dst)
json.dump(runs, open(f"{OUT}/runs.json", "w"))
print(f"\nTRAIN OK in {(time.time() - T0) / 3600:.2f} h", runs)
