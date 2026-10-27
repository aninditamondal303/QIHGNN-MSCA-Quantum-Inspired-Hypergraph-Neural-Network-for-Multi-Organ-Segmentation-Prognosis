"""Download the three Kaggle datasets into data/raw/<key>/.

Needs a Kaggle API token:
  1. kaggle.com -> Settings -> API -> "Create New Token" downloads kaggle.json
  2. Windows: put it in C:\\Users\\<you>\\.kaggle\\kaggle.json
     macOS/Linux: ~/.kaggle/kaggle.json (then: chmod 600 ~/.kaggle/kaggle.json)
  3. pip install kaggle   (already in ml/requirements.txt)

Run from the repository root:  python scripts/download_data.py [--only brats2020 lits cervical]
The archives are large (BraTS 2020 is several GB). Each dataset is skipped if its folder is not empty.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASETS = {
    "brats2020": "awsaf49/brats2020-training-data",
    "lits": "harshwardhanbhangale/lits-dataset",
    "cervical": "balajikartheek/cervical-cancer-segmentation",
}


def download(key: str, slug: str) -> None:
    target = RAW / key
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {key}: {target} is not empty")
        return
    target.mkdir(parents=True, exist_ok=True)
    print(f"[get ] {slug} -> {target}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(slug, path=str(target), unzip=True, quiet=False)
    except ImportError:
        exe = shutil.which("kaggle")
        if not exe:
            sys.exit("The kaggle package is not installed. Run: pip install kaggle")
        subprocess.run([exe, "datasets", "download", "-d", slug, "-p", str(target), "--unzip"], check=True)
    print(f"[done] {key}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), help="download a subset")
    args = ap.parse_args()
    for key in args.only or DATASETS:
        download(key, DATASETS[key])
    print("\nNext: cd ml && python -m qihgnn.data.inspection")


if __name__ == "__main__":
    main()
