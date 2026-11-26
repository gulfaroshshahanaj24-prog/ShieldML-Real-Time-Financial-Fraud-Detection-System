import time, json
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.features import add_features, model_matrix, NUMERIC_BASE

app = FastAPI(title="ShieldML Fraud Detection API", version="1.0.0")
MODEL_PATH = Path("artifacts/model.joblib")
META_PATH = Path("artifacts/metadata.json")
bundle = None
metadata = {}
stats = {"prediction_count": 0, "latencies_ms": [], "fraud_count": 0}

class Transaction(BaseModel):
    amount: float = Field(gt=0)
    merchant: str = "unknown_merchant"
    timestamp: datetime
    device_id: str = "unknown_device"
    customer_id: str = "unknown_customer"

@app.on_event("startup")
def load_model():
    global bundle, metadata
    if MODEL_PATH.exists() and META_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        metadata = json.loads(META_PATH.read_text())

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.get("/metrics")
def metrics():
    lat = stats["latencies_ms"]
    return {
        "prediction_count": stats["prediction_count"],
        "average_latency_ms": float(np.mean(lat)) if lat else 0.0,
        "p95_latency_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        "fraud_rate_last_100": stats["fraud_count"] / max(1, min(100, len(lat)))
    }

@app.post("/predict")
def predict(tx: Transaction):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not found. Run `python -m src.train` first.")
    start = time.perf_counter()
    row = pd.DataFrame([tx.model_dump()])
    row["is_fraud"] = 0
    featured = add_features(row)
    X = model_matrix(featured)
    estimators = bundle["estimators"]
    probabilities = [model.predict_proba(X)[:, 1][0] for _, model in estimators]
    score = float(np.mean(probabilities))
    threshold = float(metadata.get("threshold", 0.5))
    label = int(score >= threshold)
    elapsed = (time.perf_counter() - start) * 1000
    stats["prediction_count"] += 1
    stats["latencies_ms"] = (stats["latencies_ms"] + [elapsed])[-100:]
    stats["fraud_count"] = (stats["fraud_count"] + label) if len(stats["latencies_ms"]) > 1 else label
    return {"fraud_probability": score, "threshold": threshold,
            "prediction": "fraud" if label else "legitimate",
            "latency_ms": elapsed,
            "note": "Educational model output; verify before taking financial action."}
