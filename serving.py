"""Model loading and prediction shared by the API and the Streamlit dashboard.

Both front ends (``predict.py`` and ``streamlit_app.py``) go through
:class:`SalesForecaster`, so a forecast shown in the dashboard is the same
number the API returns for the same store-day.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model import build_inference_frame, load_json

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MIN_DATE = date(2013, 1, 1)
MAX_DATE = date(2015, 12, 31)


class UnknownStoreError(LookupError):
    """The store ID is not one of the 1,115 stores the model was trained on."""


class DateOutOfRangeError(ValueError):
    """The date falls outside the window the model accepts."""


def as_date(value: date | datetime | pd.Timestamp | str) -> date:
    """Normalise the date-ish values the two front ends pass in."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


class SalesForecaster:
    """Load the committed artifact once and answer store-day questions."""

    def __init__(self, model_dir: str | Path = MODELS) -> None:
        self.model_dir = Path(model_dir)
        model_path = self.model_dir / "model.joblib"
        self.model = joblib.load(model_path)
        self.model_metadata: dict[str, Any] = load_json(
            self.model_dir / "model_metadata.json"
        )
        self.feature_metadata: dict[str, Any] = load_json(
            self.model_dir / "feature_metadata.json"
        )
        self.store_metadata: dict[str, Any] = load_json(
            self.model_dir / "store_metadata.json"
        )
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.store_ids = sorted(int(store) for store in self.store_metadata)
        self.categories: dict[str, list[Any]] = self.feature_metadata["categories"]

    @property
    def model_version(self) -> str:
        return str(self.model_metadata["model_version"])

    @property
    def model_type(self) -> str:
        return str(self.model_metadata["model_type"])

    def store_profile(self, store: int) -> dict[str, Any]:
        """Static attributes for one store (type, assortment, competition, Promo2)."""

        self._check_store(store)
        return self.store_metadata[str(int(store))]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": self.model_version,
            "model_type": self.model_type,
            "trained_at": self.model_metadata["trained_at"],
            "model_sha256": self.model_sha256,
        }

    def validate(self, store: int, when: date | datetime | str) -> date:
        """Check a store-day and return its normalised date."""

        when = as_date(when)
        self._check_store(store)
        if not MIN_DATE <= when <= MAX_DATE:
            raise DateOutOfRangeError(
                f"date must be between {MIN_DATE} and {MAX_DATE}"
            )
        return when

    def predict(
        self,
        store: int,
        when: date | datetime | str,
        promo: int = 0,
        state_holiday: str = "0",
        school_holiday: int = 0,
    ) -> float:
        """Forecast sales for one open store-day."""

        record = {
            "store": store,
            "date": when,
            "promo": promo,
            "state_holiday": state_holiday,
            "school_holiday": school_holiday,
        }
        return self.predict_many([record])[0]

    def predict_many(self, records: Sequence[dict[str, Any]]) -> list[float]:
        """Forecast a batch of store-days with one vectorised model call."""

        if not records:
            return []

        rows = [
            self._record(
                record["store"],
                record["date"],
                record.get("promo", 0),
                record.get("state_holiday", "0"),
                record.get("school_holiday", 0),
            )
            for record in records
        ]
        frame = build_inference_frame(
            rows,
            store_metadata=self.store_metadata,
            categories=self.categories,
        )
        predictions: np.ndarray = self.model.predict(frame)
        return [float(value) for value in predictions]

    def features(
        self,
        store: int,
        when: date | datetime | str,
        promo: int = 0,
        state_holiday: str = "0",
        school_holiday: int = 0,
    ) -> dict[str, Any]:
        """The exact model row for one store-day, for the dashboard's audit panel."""

        record = self._record(store, when, promo, state_holiday, school_holiday)
        frame = build_inference_frame(
            [record],
            store_metadata=self.store_metadata,
            categories=self.categories,
        )
        return {column: frame.iloc[0][column] for column in frame.columns}

    def _record(
        self,
        store: int,
        when: date | datetime | str,
        promo: int,
        state_holiday: str,
        school_holiday: int,
    ) -> dict[str, Any]:
        when = self.validate(store, when)
        return {
            "date": when.isoformat(),
            "store": int(store),
            "promo": int(promo),
            "state_holiday": str(state_holiday),
            "school_holiday": int(school_holiday),
        }

    def _check_store(self, store: int) -> None:
        try:
            known = str(int(store)) in self.store_metadata
        except (TypeError, ValueError):
            known = False
        if not known:
            raise UnknownStoreError(f"unknown store {store}")
