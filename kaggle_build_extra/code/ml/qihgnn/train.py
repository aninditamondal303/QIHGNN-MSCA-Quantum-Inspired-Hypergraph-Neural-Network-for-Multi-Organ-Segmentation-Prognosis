"""End-to-end training (report, 4.9).

  python -m qihgnn.train --config configs/smoke.json      # minutes on CPU, synthetic phantoms
  python -m qihgnn.train --config configs/full.json       # 200 epochs on preprocessed corpora (GPU)
  python -m qihgnn.train --config configs/full.json --resume runs/<id>/last.pt

Writes runs/<run_id>/{config.json, history.csv, last.pt, best.pt} and results/training_history.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import LABELS, ModelConfig, TrainConfig
from .data.dataset import CacheDataset, EventAwareBatchSampler, PhantomDataset, collate
from .data.registry import supervised_channels
from .losses import CompositeObjective, marginal_loss
from .model import QIHGNNMSCA
from .verify import ROOT, _git_hash

DAYS_PER_YEAR = 365.25


def load_configs(path: str | None):
    raw = json.loads(Path(path).read_text()) if path else {}
    mc = ModelConfig(**{k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
                        for k, v in raw.get("model", {}).items()})
    tc = TrainConfig(**raw.get("train", {}))
    return mc, tc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_groups(model: torch.nn.Module, extra: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for module in (model, extra):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # No weight decay on normalisation, bias, or the loss log-variances (report, 4.9).
            (no_decay if p.ndim <= 1 or "norm" in name or name.endswith("bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_data(mc: ModelConfig, tc: TrainConfig):
    if tc.dataset == "phantom":
        train = PhantomDataset(tc.phantom_cases, mc.input_shape, seed=tc.seed)
        val = PhantomDataset(max(2, tc.phantom_cases // 4), mc.input_shape, seed=tc.seed + 1)
        unsupervised = ["kidney", "cervix", "spleen"]  # phantoms supervise brain and liver only
        return train, val, unsupervised
    cache = ROOT / tc.cache_dir
    train = CacheDataset(cache, "train", augment=True)
    val = CacheDataset(cache, "val")
    sup = supervised_channels(ROOT)
    return train, val, [l for l in LABELS if l not in sup]


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    vals = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device), batch["covariates"].to(device))
            vals.append(float(marginal_loss(out["seg_logits"], batch["label"].to(device), batch["supervised"].to(device))))
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "smoke.json"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    mc, tc = load_configs(args.config)
    seed_everything(tc.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tc.amp and device == "cuda"
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{tc.dataset}"
    run_dir = ROOT / tc.run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, unsupervised = build_data(mc, tc)
    sampler = EventAwareBatchSampler(
        [bool(train_ds[i]["has_survival"]) for i in range(len(train_ds))] if tc.dataset == "phantom" else [e["has_survival"] for e in train_ds.entries],
        [int(train_ds[i]["event"]) for i in range(len(train_ds))] if tc.dataset == "phantom" else [int(e["event"] or 0) for e in train_ds.entries],
        tc.batch_size, tc.min_events_per_batch, tc.seed,
    )
    workers = 0 if tc.dataset == "phantom" else tc.num_workers
    train_dl = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=workers)
    val_dl = DataLoader(val_ds, batch_size=tc.batch_size, collate_fn=collate, num_workers=workers)

    model = QIHGNNMSCA(mc).to(device)
    objective = CompositeObjective([y * DAYS_PER_YEAR for y in tc.horizons_years]).to(device)
    opt = torch.optim.AdamW(param_groups(model, objective, tc.weight_decay), lr=tc.peak_lr)
    steps_per_epoch = max(1, (tc.max_steps_per_epoch or len(sampler)) // tc.grad_accumulation)
    total_steps = steps_per_epoch * tc.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, tc.warmup_epochs * steps_per_epoch, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best = 0, float("inf")

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        objective.load_state_dict(ck["objective"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best_val", best)

    meta = {"run_id": run_id, "git": _git_hash(), "device": device, "torch": torch.__version__,
            "model_config": mc.to_dict(), "train_config": asdict(tc), "unsupervised_channels": unsupervised,
            "train_cases": len(train_ds), "val_cases": len(val_ds)}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"run {run_id} on {device}: {len(train_ds)} train / {len(val_ds)} val, unsupervised={unsupervised}")

    history = []
    started = time.time()
    log = (run_dir / "history.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_total", "train_marginal", "train_cox", "val_marginal", "lr", "skipped", "seconds"])
    model.train()
    for epoch in range(start_epoch, tc.epochs):
        t0, sums, counts, skipped = time.time(), {}, {}, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl):
            if tc.max_steps_per_epoch and step >= tc.max_steps_per_epoch:
                break
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=use_amp):
                out = model(b["image"], b["covariates"])
            terms = objective(out, b)
            loss = terms["total"] / tc.grad_accumulation
            if not torch.isfinite(loss):
                skipped += 1  # never let a non-finite batch touch the weights
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if (step + 1) % tc.grad_accumulation == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(objective.parameters()), tc.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
                counts[k] = counts.get(k, 0) + 1
        avg = {k: sums[k] / counts[k] for k in sums}
        val = evaluate_loss(model, val_dl, device)
        row = {"epoch": epoch, "train_total": avg.get("total"), "train_marginal": avg.get("marginal"),
               "train_cox": avg.get("cox"), "val_marginal": val, "lr": opt.param_groups[0]["lr"],
               "skipped": skipped, "seconds": round(time.time() - t0, 1)}
        history.append(row)
        writer.writerow(list(row.values()))
        log.flush()
        print(f"epoch {epoch:>3}  train {row['train_total']:.4f}  marginal {row['train_marginal']:.4f}  val {val:.4f}  ({row['seconds']}s)")
        ck = {"model": model.state_dict(), "objective": objective.state_dict(), "optimizer": opt.state_dict(),
              "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_val": min(best, val),
              "run_id": run_id, "dataset": tc.dataset, "model_config": mc.to_dict(), "unsupervised_channels": unsupervised}
        torch.save(ck, run_dir / "last.pt")
        if val < best:
            best = val
            torch.save(ck, run_dir / "best.pt")
        if tc.time_limit_hours and (time.time() - started) / 3600 > tc.time_limit_hours:
            print(f"time limit of {tc.time_limit_hours} h reached after epoch {epoch}; stopping")
            break
    log.close()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "training_history.json").write_text(json.dumps({**meta, "history": history}, indent=1))
    latest = ROOT / tc.run_dir / "LATEST"
    latest.write_text(run_id)
    print(f"done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
