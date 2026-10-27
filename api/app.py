"""HTTP API for the QIHGNN-MSCA platform.

Built directly on Starlette (the ASGI toolkit FastAPI is built on) with pydantic
models for request validation and response schemas. One process serves both the
JSON API under /api and the built website at /.

  uvicorn api.app:app --reload --port 8000        (from the repository root)
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from qihgnn.config import LABEL_DESCRIPTIONS, LABELS, ModelConfig  # noqa: E402
from qihgnn.data.registry import corpus_summary  # noqa: E402
from qihgnn.param_count import attention_cost, mpo_compression_table, parameter_budget  # noqa: E402
from qihgnn.verify import build_report, to_json  # noqa: E402

VERSION = "0.4.0"
STARTED = time.time()
MAX_SIDE = 512
DISCLAIMER = "Research prototype. Not a medical device and not for clinical use."


# ----------------------------------------------------------------------------- schemas
class Health(BaseModel):
    status: str
    version: str
    uptime_s: float
    torch_available: bool
    checkpoint: Optional[str]
    inference_ready: bool
    disclaimer: str = DISCLAIMER


class AttentionQuery(BaseModel):
    h: int = Field(128, ge=8, le=MAX_SIDE)
    w: int = Field(128, ge=8, le=MAX_SIDE)
    d: int = Field(64, ge=8, le=MAX_SIDE)


class PredictRequest(BaseModel):
    case_id: str = Field(..., min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")


class PredictResponse(BaseModel):
    case_id: str
    run_id: Optional[str]
    shape: List[int]
    mosaic_columns: int
    label_png_base64: str
    suppressed: List[str]
    log_risk: float
    survival_probability: Dict[str, float]
    aleatoric_sd: float
    seconds: float
    disclaimer: str = DISCLAIMER


class ApiError(Exception):
    def __init__(self, status: int, detail: str, how_to_fix: str | None = None) -> None:
        self.status, self.detail, self.how_to_fix = status, detail, how_to_fix


def ok(model_or_data: Any, status: int = 200) -> JSONResponse:
    if isinstance(model_or_data, BaseModel):
        return JSONResponse(model_or_data.model_dump(), status_code=status)
    return JSONResponse(json.loads(to_json(model_or_data)), status_code=status)


# ----------------------------------------------------------------------------- inference
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def find_checkpoint() -> Optional[Path]:
    env = os.environ.get("QIHGNN_CHECKPOINT")
    if env:
        return Path(env) if Path(env).exists() else None
    latest = ROOT / "runs" / "LATEST"
    if latest.exists():
        run = ROOT / "runs" / latest.read_text().strip()
        for name in ("best.pt", "last.pt"):
            if (run / name).exists():
                return run / name
    return None


@lru_cache(maxsize=1)
def predictor():
    from qihgnn.serving import Predictor

    ck = find_checkpoint()
    if ck is None:
        raise ApiError(503, "No trained checkpoint was found.",
                       "Train one with `python -m qihgnn.train --config configs/smoke.json` (from ml/), then restart the API.")
    return Predictor(ck)


@lru_cache(maxsize=8)
def case_volume(case_id: str):
    from qihgnn.phantoms import all_phantoms

    for ph in all_phantoms():
        if ph.case_id == case_id:
            return ph
    raise ApiError(404, f"Unknown case '{case_id}'.", "GET /api/cases lists the available case ids.")


# ----------------------------------------------------------------------------- handlers
async def health(_: Request):
    ck = find_checkpoint()
    has_torch = torch_available()
    return ok(Health(status="ok", version=VERSION, uptime_s=round(time.time() - STARTED, 1), torch_available=has_torch,
                     checkpoint=str(ck.relative_to(ROOT)) if ck else None, inference_ready=bool(ck and has_torch)))


async def model_card(_: Request):
    return ok({
        "name": "QIHGNN-MSCA", "version": VERSION, "disclaimer": DISCLAIMER,
        "intended_use": "Research on partially-supervised multi-organ segmentation and joint prognosis.",
        "not_intended_for": ["Diagnosis or treatment decisions", "Use on patients", "Any organ marked unsupervised"],
        "labels": [{"name": l, "description": LABEL_DESCRIPTIONS[l]} for l in LABELS],
        "parameters": parameter_budget(), "config": ModelConfig().to_dict(),
        "limitations": [
            "No converged training run or baseline comparison is reported yet; segmentation and prognosis scores are 'not yet measured' until results/metrics.json exists.",
            "Each corpus supervises one organ, so organs that are adjacent in a single scan have not been evaluated together.",
            "Under the softmax head the exclusion loss is identically zero.",
            "Synthetic phantoms on this site illustrate the interface and are never used to report performance.",
        ],
    })


async def verification(_: Request):
    path = ROOT / "results" / "verification.json"
    if not path.exists():
        raise ApiError(404, "results/verification.json does not exist yet.", "POST /api/verification/run or run `python -m qihgnn.verify`.")
    return ok(json.loads(path.read_text()))


async def run_verification(_: Request):
    report = build_report()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "verification.json").write_text(to_json(report, indent=2))
    return ok(report)


async def attention(request: Request):
    try:
        q = AttentionQuery(**dict(request.query_params))
    except ValidationError as e:
        raise ApiError(422, e.errors()[0]["msg"], f"Each side must be an integer between 8 and {MAX_SIDE}.")
    return ok(attention_cost(shape=(q.h, q.w, q.d)) | {"shape": [q.h, q.w, q.d]})


async def mpo(_: Request):
    return ok({"rows": mpo_compression_table()})


async def corpus(_: Request):
    return ok(corpus_summary(ROOT))


async def cases(_: Request):
    path = ROOT / "web" / "public" / "data" / "cases.json"
    if not path.exists():
        raise ApiError(404, "No exported cases.", "Run `python -m qihgnn.export_web` from ml/.")
    return ok(json.loads(path.read_text()))


async def metrics(_: Request):
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        raise ApiError(404, "Not yet measured.", "Train, then run `python -m qihgnn.evaluate --checkpoint runs/<id>/best.pt`.")
    return ok(json.loads(path.read_text()))


async def predict(request: Request):
    try:
        body = PredictRequest(**(await request.json()))
    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        raise ApiError(422, "Body must be JSON like {\"case_id\": \"PH-BRAIN-01\"}.", str(e)[:200])
    if not torch_available():
        raise ApiError(503, "PyTorch is not installed on the server.", "pip install -r ml/requirements.txt")
    case = case_volume(body.case_id)
    pred = predictor()
    from PIL import Image

    from qihgnn.data.preprocess import to_grid
    from qihgnn.export_web import mosaic

    t0 = time.time()
    shape = pred.cfg.input_shape
    image = to_grid(case.image, shape, order=1)
    out = pred.predict(image, case.covariates)
    label = to_grid(out["label"], case.label.shape, order=0)
    m, cols = mosaic(label)
    buf = io.BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG", optimize=True)
    return ok(PredictResponse(
        case_id=body.case_id, run_id=pred.meta.get("run_id"), shape=list(label.shape), mosaic_columns=cols,
        label_png_base64=base64.b64encode(buf.getvalue()).decode(), suppressed=out["suppressed"],
        log_risk=out["log_risk"], survival_probability=out["survival_probability"], aleatoric_sd=out["aleatoric_sd"],
        seconds=round(time.time() - t0, 2),
    ))


async def api_error(_: Request, exc: ApiError):
    return JSONResponse({"detail": exc.detail, "how_to_fix": exc.how_to_fix}, status_code=exc.status)


routes = [
    Route("/api/health", health),
    Route("/api/model-card", model_card),
    Route("/api/verification", verification),
    Route("/api/verification/run", run_verification, methods=["POST"]),
    Route("/api/attention-cost", attention),
    Route("/api/mpo", mpo),
    Route("/api/corpus", corpus),
    Route("/api/cases", cases),
    Route("/api/metrics", metrics),
    Route("/api/predict", predict, methods=["POST"]),
]
dist = ROOT / "web" / "dist"
if dist.exists():
    routes.append(Mount("/", app=StaticFiles(directory=dist, html=True), name="web"))

app = Starlette(
    routes=routes,
    middleware=[Middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                           allow_methods=["GET", "POST"], allow_headers=["content-type"])],
    exception_handlers={ApiError: api_error},
)
