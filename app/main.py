"""
main.py
-------
FastAPI application entry point.

Startup sequence:
  1. FastAPI app is created.
  2. On first request (lazy) or at import time, model_service loads the
     trained model, shap_service builds TreeExplainer, lime_service builds
     LimeTabularExplainer.  All three are module-level singletons — they are
     created ONCE and reused for every subsequent request.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import API_TITLE, API_VERSION, API_DESC
from app.schemas import (
    PredictRequest,
    HealthResponse,
    RootResponse,
    PredictResponse,
    SHAPResponse,
    LIMEResponse,
    CombinedResponse,
)

# Import services — this triggers model/explainer loading at startup
import app.model_service as model_svc
import app.shap_service  as shap_svc
import app.lime_service  as lime_svc


# ── Lifespan (startup log) ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[main] {API_TITLE} v{API_VERSION} starting up.")
    print("[main] Model, SHAP explainer, and LIME explainer are loaded.")
    yield
    print("[main] Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESC,
    lifespan=lifespan,
)

# Allow browser dashboard (served from file://) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend dashboard at /dashboard ────────────────────────────────────
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        """Serve the XAI frontend dashboard."""
        return FileResponse(os.path.join(_frontend_dir, "index.html"))


# ── GET / ─────────────────────────────────────────────────────────────────────
@app.get("/", response_model=RootResponse, tags=["Info"])
def root():
    """Service information."""
    meta = model_svc.get_metadata()
    return RootResponse(
        service             = API_TITLE,
        model               = meta["model_info"]["algorithm"],
        dataset             = meta["model_info"]["dataset"],
        status              = "running",
        version             = API_VERSION,
        explanation_methods = ["SHAP", "LIME"],
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Kubernetes readiness / liveness probe endpoint."""
    return HealthResponse(status="healthy")


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    BASELINE endpoint — model inference only, no explanation.

    Use this to measure the raw model latency before adding XAI overhead.
    """
    try:
        result = model_svc.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        prediction        = result["prediction"],
        probability       = result["probability"],
        inference_time_ms = result["inference_time_ms"],
    )


# ── POST /explain/shap ────────────────────────────────────────────────────────
@app.post("/explain/shap", response_model=SHAPResponse, tags=["Explanation"])
def explain_shap(request: PredictRequest):
    """
    Model inference + SHAP explanation.

    Returns feature contributions sorted by absolute SHAP value (most
    impactful feature first).  Timings are measured separately so you can
    compute XAI overhead = total_time_ms - inference_time_ms.
    """
    try:
        # Inference timing (baseline portion)
        infer = model_svc.predict(request.features)

        # SHAP timing
        shap_result = shap_svc.explain(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return SHAPResponse(
        prediction          = shap_result["prediction"],
        probability         = shap_result["probability"],
        explanation_method  = "SHAP",
        features            = shap_result["feature_names"],
        contributions       = shap_result["shap_values"],
        base_value          = shap_result["base_value"],
        inference_time_ms   = infer["inference_time_ms"],
        explanation_time_ms = shap_result["explanation_time_ms"],
        total_time_ms       = shap_result["total_time_ms"],
    )


# ── POST /explain/lime ────────────────────────────────────────────────────────
@app.post("/explain/lime", response_model=LIMEResponse, tags=["Explanation"])
def explain_lime(request: PredictRequest):
    """
    Model inference + LIME explanation.

    Returns human-readable feature rules and their contribution weights.
    LIME is typically slower than SHAP — this difference is the overhead gap.
    """
    try:
        infer       = model_svc.predict(request.features)
        lime_result = lime_svc.explain(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return LIMEResponse(
        prediction          = lime_result["prediction"],
        probability         = lime_result["probability"],
        explanation_method  = "LIME",
        feature_rules       = lime_result["feature_rules"],
        contributions       = lime_result["lime_weights"],
        inference_time_ms   = infer["inference_time_ms"],
        explanation_time_ms = lime_result["explanation_time_ms"],
        total_time_ms       = lime_result["total_time_ms"],
    )


# ── POST /explain ─────────────────────────────────────────────────────────────
@app.post("/explain", response_model=CombinedResponse, tags=["Explanation"])
def explain_combined(request: PredictRequest):
    """
    Model inference + SHAP + LIME combined explanation.

    All timings are measured separately:
      inference_time_ms        — raw model prediction
      shap_explanation_time_ms — SHAP computation only
      lime_explanation_time_ms — LIME computation only
      total_time_ms            — wall clock for this entire request
      xai_overhead_ms          — total_time_ms - inference_time_ms
      xai_overhead_pct         — overhead as a percentage of baseline

    This endpoint is the main research measurement point.
    """
    t_request_start = time.perf_counter()

    try:
        infer       = model_svc.predict(request.features)
        shap_result = shap_svc.explain(request.features)
        lime_result = lime_svc.explain(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    t_request_end = time.perf_counter()

    total_ms    = round((t_request_end - t_request_start) * 1000, 4)
    baseline_ms = infer["inference_time_ms"]
    overhead_ms = round(total_ms - baseline_ms, 4)
    overhead_pct = round(
        ((total_ms - baseline_ms) / baseline_ms * 100) if baseline_ms > 0 else 0.0,
        2,
    )

    return CombinedResponse(
        prediction               = shap_result["prediction"],
        probability              = shap_result["probability"],
        shap_features            = shap_result["feature_names"],
        shap_contributions       = shap_result["shap_values"],
        shap_base_value          = shap_result["base_value"],
        lime_feature_rules       = lime_result["feature_rules"],
        lime_contributions       = lime_result["lime_weights"],
        inference_time_ms        = baseline_ms,
        shap_explanation_time_ms = shap_result["explanation_time_ms"],
        lime_explanation_time_ms = lime_result["explanation_time_ms"],
        total_time_ms            = total_ms,
        xai_overhead_ms          = overhead_ms,
        xai_overhead_pct         = overhead_pct,
    )
