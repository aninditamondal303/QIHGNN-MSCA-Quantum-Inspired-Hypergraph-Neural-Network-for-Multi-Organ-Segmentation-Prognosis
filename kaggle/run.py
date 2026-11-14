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
inits = glob.glob("/kaggle/input/**/ml/qihgnn/__init__.py", recursive=True)
if not inits:
    tree("/kaggle/input", depth=4)
    raise SystemExit("qihgnn code dataset not found under /kaggle/input")
src = os.path.dirname(os.path.dirname(os.path.dirname(inits[0])))
shutil.copytree(src, REPO, dirs_exist_ok=True)
os.makedirs(f"{REPO}/results", exist_ok=True)
ML = f"{REPO}/ml"
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
    sh(f"{sys.executable} -m qihgnn.data.cache --limit 2", cwd=ML)
    with open(f"{REPO}/data/cache/index.json") as f:
        idx = json.load(f)
    print(json.dumps({"counts": idx["counts"], "cases": [{k: c[k] for k in ("patient_id", "source", "has_survival", "voxels")} for c in idx["cases"]]}, indent=1))
    shutil.copy(f"{REPO}/data/cache/index.json", f"{OUT}/probe_index.json")
    print("\nPROBE OK")
    sys.exit(0)

# ---------------------------------------------------------------- train
sh(f"{sys.executable} -m qihgnn.data.cache", cwd=ML)
shutil.copy(f"{REPO}/data/cache/index.json", f"{OUT}/cache_index.json")
sh(f"{sys.executable} -m qihgnn.verify", cwd=ML)

runs = {}
for label, cfg in [("full", "configs/kaggle_full.json"), ("baseline", "configs/kaggle_baseline.json")]:
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
