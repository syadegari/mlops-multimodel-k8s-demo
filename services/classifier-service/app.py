#!/usr/bin/env python3
"""
FastAPI app for the MNIST RandomForest classifier.

- Loads a joblib model from MODEL_PATH (env var) or a default path.
- Exposes:
    GET /healthz          -> health check
    GET /metrics-simple   -> simple JSON metrics
    POST /predict         -> predict class + probabilities from feature vector
"""

import json
import os
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


ARTIFICIAL_CPU_SEC = float(os.getenv("ARTIFICIAL_CPU_SEC", "0.0"))

def simulate_cpu_load(seconds: float) -> None:
    """
    Keep cpu busy for approximately `seconds` wall-clock time.

    This is purely for demo purposes to make HPA behaviour visible on small hardware.
    """
    if seconds <= 0:
        return

    check_counter = 1
    end_time = time.perf_counter() + seconds
    x = 0.0

    while True:
        check_counter += 1
        if check_counter % 100 == 0:
            if time.perf_counter() > end_time:
                return
        x += 1
        x *= 1.000000001
        

def resolve_model_path() -> Path:
    """Resolve the model path from env var or default to models/mnist_rf_v1.joblib."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
    else:
        # Assume repo root is two levels up: services/classifier-service/app.py -> repo root
        here = Path(__file__).resolve()
        root_dir = here.parents[2]
        p = (root_dir / "models" / "mnist_rf_v1.joblib").resolve()

    if not p.is_file():
        raise FileNotFoundError(f"Model file not found at: {p}")
    return p


MODEL_PATH = resolve_model_path()
MODEL = joblib.load(MODEL_PATH)

# metrics 
TOTAL_REQUESTS = 0
TOTAL_ERRORS = 0
TOTAL_LATENCY_MS = 0.0


# FastAPI part
app = FastAPI(title="MNIST RandomForest Classifier Service")


@app.middleware("http")
async def logging_and_metrics_middleware(request: Request, call_next):
    """Log each request and track simple metrics."""
    global TOTAL_REQUESTS, TOTAL_ERRORS, TOTAL_LATENCY_MS

    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        TOTAL_ERRORS += 1
        # let fastapi handle traceback/error response
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        TOTAL_REQUESTS += 1
        TOTAL_LATENCY_MS += latency_ms

        log_record = {
            "service": "classifier",
            "path": request.url.path,
            "method": request.method,
            "status": status_code,
            "latency_ms": round(latency_ms, 2),
        }
        # Structured log as JSON line to stdout
        print(json.dumps(log_record), flush=True)

    return response


# Schemas
class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    label: int
    probabilities: List[float]


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    avg_latency_ms: float


# Endpoints
@app.get("/healthz")
def healthz():
    """Simple health check."""
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
    }


@app.get("/metrics-simple", response_model=MetricsResponse)
def metrics_simple():
    """Return very simple process-level metrics."""
    if TOTAL_REQUESTS > 0:
        avg = TOTAL_LATENCY_MS / TOTAL_REQUESTS
    else:
        avg = 0.0
    return MetricsResponse(
        total_requests=TOTAL_REQUESTS,
        total_errors=TOTAL_ERRORS,
        avg_latency_ms=round(avg, 2),
    )


# TODO: Extend API to accept a model_version field in the request and route to different saved 
# models for user-controlled versioning.
# This requires adding at least one more endpoint for querying the model versions.

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Predict MNIST digit from a flattened feature vector.

    Only check the length is nonzero and let the model complain otherwise.

    TODO: Add better checks 
    """
    features = np.array(payload.features, dtype=np.float32)

    if features.ndim != 1:
        raise HTTPException(status_code=400, detail="features must be a 1D list of floats")

    X = features.reshape(1, -1)  # (1, n_features)

    # Artificial CPU for demo purposes (HPA / kubectl top pods)
    simulate_cpu_load(ARTIFICIAL_CPU_SEC)
    
    try:
        # RandomForestClassifier supports predict_proba
        probas = MODEL.predict_proba(X)[0]
        label = int(np.argmax(probas))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictResponse(
        label=label,
        probabilities=[float(p) for p in probas],
    )


# For local dev: allow `python app.py` to run the server directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.classifier-service.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
