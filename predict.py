"""FastAPI service for the Rossmann daily-sales forecaster."""

from __future__ import annotations

from datetime import date as date_type
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serving import DateOutOfRangeError, SalesForecaster, UnknownStoreError

forecaster = SalesForecaster()

app = FastAPI(
    title="Rossmann daily sales forecast",
    version=forecaster.model_version,
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
    return forecaster.health()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = forecaster.predict(
            store=request.store,
            when=request.date,
            promo=request.promo,
            state_holiday=request.state_holiday,
            school_holiday=request.school_holiday,
        )
    except UnknownStoreError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DateOutOfRangeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(
        store=request.store,
        date=request.date,
        predicted_sales=round(prediction, 2),
        model_version=forecaster.model_version,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("predict:app", host="0.0.0.0", port=9696)
