"""Kaggle kernel: put the trained 3D QIHGNN-MSCA network on the website. kaggle_run.ps1 -Job net3d -Mode probe (CPU).

1. Preprocesses a few patients per corpus exactly as training did, and keeps held-out (test-split) ones.
2. Loads the trained checkpoint from the earlier training kernel's output and exports the whole network to ONNX:
   image (1, 4, 128, 128, 64) + covariates (1, 5) -> label map (unsupervised channels suppressed) and prognosis.
3. Checks ONNX Runtime against PyTorch, and runs the network on each case from the 8-bit PNG copy the website
   uses, scoring it against the expert labels, so the site can show how well it did on that very patient.
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
print("code from", src, "| python", sys.version.split()[0])
sh(f"{sys.executable} -m pip install -q nibabel h5py onnx onnxruntime", check=False)
raw = f"{REPO}/data/raw"
os.makedirs(raw, exist_ok=True)
for key, slug in [("brats2020", "brats2020-training-data"), ("lits", "lits-dataset"), ("cervical", "cervical-cancer-segmentation")]:
    path = find_input(slug)
    print(f"==== {key}: {path}")
    if path:
        os.symlink(path, f"{raw}/{key}")
sh(f"{sys.executable} -m qihgnn.data.inspection", cwd=ML)
os.makedirs(OUT, exist_ok=True)
sh(f"{sys.executable} -m qihgnn.data.cache --limit 40 --workers 8", cwd=ML)

ckpts = sorted(glob.glob("/kaggle/input/**/best.pt", recursive=True))
print("checkpoints:", ckpts)
if not ckpts:
    tree("/kaggle/input", depth=6)
    raise SystemExit("no best.pt under /kaggle/input: attach the training kernel's output")

STEP2 = r"""
import json, sys, time, math
import numpy as np, torch, onnxruntime as ort
from pathlib import Path
from PIL import Image
sys.path.insert(0, '.')
from qihgnn.config import LABELS, ModelConfig
from qihgnn.model import QIHGNNMSCA
from qihgnn.export_web import mosaic, to_uint8

OUT = Path(sys.argv[1]); CK = sys.argv[2:]
cache = Path('../data/cache'); idx = json.loads((cache / 'index.json').read_text())

# The full model's checkpoint (all three stages on), whichever folder it came from.
model = None
for ck in CK:
    c = torch.load(ck, map_location='cpu', weights_only=False)
    mc = c['model_config']
    if mc.get('use_mpo', True) and mc.get('use_hypergraph', True) and mc.get('use_msca', True):
        model = c; print('using', ck, 'epoch', c.get('epoch'), 'run', c.get('run_id')); break
assert model is not None, 'full-model checkpoint not found'
cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in model['model_config'].items()})
net = QIHGNNMSCA(cfg).eval(); net.load_state_dict(model['model'])
unsup = [LABELS.index(n) for n in model['unsupervised_channels']]
print('parameters', sum(p.numel() for p in net.parameters()), 'suppressed', model['unsupervised_channels'])

class Export(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        keep = torch.ones(len(LABELS)); keep[unsup] = 0
        s.register_buffer('keep', keep.view(1, -1, 1, 1, 1))
    def forward(s, image, covariates):
        o = s.m(image, covariates)
        p = torch.softmax(o['seg_logits'].float(), 1) * s.keep
        label = p.argmax(1).to(torch.uint8)
        return label, o['log_risk'].float().reshape(1, 1), torch.sigmoid(o['horizon_logits'].float()).reshape(1, -1), o['log_variance'].float().reshape(1, 1)

ex = Export(net).eval()
H, W, D = cfg.input_shape
x0 = torch.randn(1, cfg.in_channels, H, W, D); c0 = torch.zeros(1, cfg.num_covariates)
with torch.no_grad():
    ref = [t.clone() for t in ex(x0, c0)]

# ONNX cannot export adaptive_avg_pool3d when the input size is not static. Every grid here divides the feature
# map exactly, so the same result is a reshape and a mean, which exports cleanly. Checked against the original below.
_orig_pool = torch.nn.functional.adaptive_avg_pool3d
def _pool(x, size):
    g = tuple(size) if isinstance(size, (tuple, list)) else (size,) * 3
    b, c, d0, d1, d2 = x.shape
    return x.reshape(b, c, g[0], d0 // g[0], g[1], d1 // g[1], g[2], d2 // g[2]).mean(dim=(3, 5, 7))
torch.nn.functional.adaptive_avg_pool3d = _pool
with torch.no_grad():
    new = ex(x0, c0)
print('pool replacement: label agreement', float((new[0] == ref[0]).float().mean()), 'max risk diff', float((new[1] - ref[1]).abs().max()))
assert float((new[0] == ref[0]).float().mean()) > 0.9999
onnx_path = OUT / 'net3d.onnx'
torch.onnx.export(ex, (x0, c0), str(onnx_path), input_names=['image', 'covariates'],
                  output_names=['label', 'log_risk', 'horizon', 'log_variance'], opset_version=17, dynamo=False)
print('onnx MB', round(onnx_path.stat().st_size / 2**20, 1))
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])

def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else None

tests = [e for e in idx['cases'] if e['split'] == 'test']
picks = [e for e in tests if e['source'].startswith('brats')][:3] + [e for e in tests if e['source'].startswith('lits')][:2]
print('test cases available', len(tests), 'picked', [(e['source'], e['patient_id']) for e in picks])
cases_dir = OUT / 'cases'; cases_dir.mkdir(parents=True, exist_ok=True)
index = []
for e in picks:
    z = np.load(cache / e['file'])
    img = z['image'].astype(np.float32); lab = z['label']; sup = z['supervised']
    cov = np.array([e.get('covariates') or [0.0] * cfg.num_covariates], np.float32)
    with torch.no_grad():
        tl, trisk, thor, tvar = ex(torch.from_numpy(img[None]), torch.from_numpy(cov))
    # what the browser feeds: 8-bit PNG copy, mapped back with the stored range
    u8, ranges = [], []
    for ch in img:
        lo, hi = np.percentile(ch, [0.5, 99.5]); ranges.append([float(lo), float(hi)])
        u8.append(to_uint8(ch))
    rec = np.stack([lo + u.astype(np.float32) / 255 * (hi - lo) for u, (lo, hi) in zip(u8, ranges)])
    t0 = time.time()
    ol, orisk, ohor, ovar = sess.run(None, {'image': rec[None].astype(np.float32), 'covariates': cov})
    secs = time.time() - t0
    organs = [LABELS[i] for i in np.flatnonzero(sup) if i > 0]
    d_torch = {o: dice(tl[0].numpy() == LABELS.index(o), lab == LABELS.index(o)) for o in organs}
    d_web = {o: dice(ol[0] == LABELS.index(o), lab == LABELS.index(o)) for o in organs}
    agree = float((ol[0] == tl[0].numpy()).mean())
    print(e['source'], e['patient_id'], 'dice torch', d_torch, 'dice browser-input', d_web, 'agreement', round(agree, 4), 'ort 1-thread s', round(secs, 1))
    cid = f"{e['source'].upper()}-{e['patient_id']}"[:40]
    d = cases_dir / cid; d.mkdir(exist_ok=True)
    chans = ['T1', 'T1ce', 'T2', 'FLAIR'] if e['modality'] == 'MRI' else ['Abdomen window', 'Liver window', 'Bone window', 'Wide window']
    files = []
    cols = 0
    for i, (name, u) in enumerate(zip(chans, u8)):
        m, cols = mosaic(u); Image.fromarray(m, mode='L').save(d / f'ch{i}.png', optimize=True)
        files.append({'name': name, 'file': f'{cid}/ch{i}.png'})
    m, _ = mosaic(lab.astype(np.uint8)); Image.fromarray(m, mode='L').save(d / 'label.png', optimize=True)
    present = sorted({int(v) for v in np.unique(lab)} - {0})
    index.append({
        'id': cid, 'title': f"{'Brain MRI' if e['modality'] == 'MRI' else 'Abdominal CT'}, held-out patient {e['patient_id']}",
        'modality': e['modality'], 'analogOf': 'BraTS 2020' if e['source'].startswith('brats') else 'LiTS',
        'synthetic': False, 'shape': list(lab.shape), 'mosaicColumns': cols, 'channels': files, 'channelRanges': ranges,
        'label': f'{cid}/label.png', 'spacing': [1, 1, round(lab.shape[0] / lab.shape[2], 3)],
        'supervised': [LABELS[i] for i in np.flatnonzero(sup)], 'present': [LABELS[i] for i in present],
        'voxels': {LABELS[i]: int((lab == i).sum()) for i in present},
        'covariates': cov[0].tolist(),
        'networkDice': {k: (round(v, 3) if v is not None else None) for k, v in d_web.items()},
        'note': 'Held-out test patient: never used for training. Preprocessed exactly as in training.',
    })
json.dump({'labels': list(LABELS), 'cases': index, 'model': {'run_id': model.get('run_id'), 'epoch': model.get('epoch'),
           'suppressed': model['unsupervised_channels'], 'onnx_mb': round(onnx_path.stat().st_size / 2**20, 1)}},
          open(OUT / 'net3d_cases.json', 'w'), indent=1)
print('NET3D OK', len(index), 'cases')
"""
open(f"{ML}/_net3d.py", "w").write(STEP2)
sh(f"{sys.executable} _net3d.py {OUT} " + " ".join(ckpts), cwd=ML)
print(f"\nNET3D DONE in {(time.time() - T0) / 60:.1f} min")
