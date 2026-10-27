# data/

Nothing in here is committed except this note.

- `raw/<dataset>/` is filled by `python scripts/download_data.py`.
- `inspection_report.json` is written by `python -m qihgnn.data.inspection` (from `ml/`).
- `cache/` holds preprocessed `.npz` volumes and `index.json` (patient-level splits), written by `python -m qihgnn.data.cache`.
