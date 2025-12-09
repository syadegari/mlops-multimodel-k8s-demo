#!/usr/bin/env python3
"""
FastAPI app for a small causal LM (erwanf/gpt2-mini).

Endpoints:
    GET /healthz          -> health check
    GET /metrics-simple   -> simple JSON metrics
    POST /generate        -> text generation from a prompt
"""

import json
import os
import time
from typing import List

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ARTIFICIAL_CPU_SEC = float(os.getenv("ARTIFICIAL_CPU_SEC", "0.0"))
MODEL_NAME = os.getenv("MODEL_NAME", "erwanf/gpt2-mini")
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "256"))

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
        

print(f"[lm-service] Loading model {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# make sure we have a pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# metrics
TOTAL_REQUESTS = 0
TOTAL_ERRORS = 0
TOTAL_LATENCY_MS = 0.0

# FastAPI part
app = FastAPI(title="GPT2-mini LM Service")


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
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        TOTAL_REQUESTS += 1
        TOTAL_LATENCY_MS += latency_ms

        log_record = {
            "service": "lm",
            "path": request.url.path,
            "method": request.method,
            "status": status_code,
            "latency_ms": round(latency_ms, 2),
        }
        print(json.dumps(log_record), flush=True)

    return response


# Schemas
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 40


class GenerateResponse(BaseModel):
    generated_text: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    avg_latency_ms: float


# Endpoints
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "max_total_tokens": MAX_TOTAL_TOKENS,
    }


@app.get("/metrics-simple", response_model=MetricsResponse)
def metrics_simple():
    if TOTAL_REQUESTS > 0:
        avg = TOTAL_LATENCY_MS / TOTAL_REQUESTS
    else:
        avg = 0.0
    return MetricsResponse(
        total_requests=TOTAL_REQUESTS,
        total_errors=TOTAL_ERRORS,
        avg_latency_ms=round(avg, 2),
    )

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Generate text from a prompt, enforcing a total token budget (prompt + new),
    and passing an explicit attention_mask to avoid HF warnings.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    # Artificial demo for demo (HPA / kubectl top pods)
    simulate_cpu_load(ARTIFICIAL_CPU_SEC)

    # Tokenize with tensors + attention_mask, no padding needed for single example
    enc = tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,              # truncate very long prompts
        max_length=MAX_TOTAL_TOKENS,  # never let the prompt exceed the total budget
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    input_len = input_ids.shape[-1]

    # Respect max_new_tokens and total token budget
    max_new = min(req.max_new_tokens, MAX_TOTAL_TOKENS)
    if input_len + max_new > MAX_TOTAL_TOKENS:
        max_new = max(MAX_TOTAL_TOKENS - input_len, 1)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.pad_token_id,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return GenerateResponse(generated_text=text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.lm-service.app:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
    )
