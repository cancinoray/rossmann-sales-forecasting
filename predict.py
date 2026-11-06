"""FastAPI service for the Rossmann daily-sales forecaster."""

from __future__ import annotations

import hashlib
from datetime import date as date_type
from pathlib import Path
from typing import Any, Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import build_inference_frame, load_json

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "model.joblib"

model = joblib.load(MODEL_PATH)
model_metadata = load_json(MODELS / "model_metadata.json")
feature_metadata = load_json(MODELS / "feature_metadata.json")
store_metadata = load_json(MODELS / "store_metadata.json")

MODEL_SHA256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
MIN_DATE = date_type(2013, 1, 1)
MAX_DATE = date_type(2015, 12, 31)

app = FastAPI(
    title="Rossmann daily sales forecast",
    version=model_metadata["model_version"],
    description="Predict daily sales for an open Rossmann store-day.",
)


class PredictionRequest(BaseModel):
    store: int = Field(ge=1, examples=[1])
    date: date_type = Field(examples=["2015-08-15"])
    promo: Literal[0, 1] = Field(default=0, examples=[1])
    state_holiday: Literal["0", "a", "b", "c"] = Field(default="0", examples=["0"])
    school_holiday: Literal[0, 1] = Field(default=0, examples=[0])


class PredictionResponse(BaseModel):
    store: int
    date: date_type
    predicted_sales: float
    model_version: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": model_metadata["model_version"],
        "model_type": model_metadata["model_type"],
        "trained_at": model_metadata["trained_at"],
        "model_sha256": MODEL_SHA256,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not (MIN_DATE <= request.date <= MAX_DATE):
        raise HTTPException(
            status_code=422,
            detail=f"date must be between {MIN_DATE} and {MAX_DATE}",
        )
    if str(request.store) not in store_metadata:
        raise HTTPException(status_code=404, detail=f"unknown store {request.store}")

    record = request.model_dump()
    record["date"] = request.date.isoformat()
    frame = build_inference_frame(
        [record],
        store_metadata=store_metadata,
        categories=feature_metadata["categories"],
    )
    prediction = float(model.predict(frame)[0])

    return PredictionResponse(
        store=request.store,
        date=request.date,
        predicted_sales=round(prediction, 2),
        model_version=model_metadata["model_version"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict:app", host="0.0.0.0", port=9696)
