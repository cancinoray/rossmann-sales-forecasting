"""Feature engineering and inference helpers shared by training and serving.

The model is deliberately built from features that are known *before* the
store-day happens: calendar, promo, holiday, and static store metadata.  The
post-hoc columns in the raw data (`Customers`, the target `Sales`, and the
`Open` flag used for filtering) never enter the feature frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STORE_FIELDS = [
    "store_type",
    "assortment",
    "competition_distance",
    "competition_open_since_month",
    "competition_open_since_year",
    "promo2",
    "promo2_since_week",
    "promo2_since_year",
    "promo_interval",
]

CATEGORICAL_FEATURES = [
    "store",
    "state_holiday",
    "store_type",
    "assortment",
    "promo_interval",
]

NUMERIC_FEATURES = [
    "day_of_week",
    "promo",
    "school_holiday",
    "competition_distance",
    "competition_open_months",
    "promo2",
    "promo2_open_months",
    "year",
    "month",
    "day",
    "week_of_year",
    "day_of_year",
    "is_month_start",
    "is_month_end",
    "is_weekend",
    "days_to_christmas",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def days_to_christmas(dates: pd.Series) -> pd.Series:
    """Days until the next 25 December (0 on Christmas Day)."""

    christmas = pd.to_datetime(dates.dt.year.astype(str) + "-12-25")
    next_christmas = christmas.where(christmas >= dates, christmas + pd.DateOffset(years=1))
    return (next_christmas - dates).dt.days.astype("int16")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and store-tenure features.

    Expects the raw-ish columns from ``train.csv`` merged with ``store.csv``:
    ``date``, ``store``, ``promo``, ``state_holiday``, ``school_holiday`` and the
    store metadata fields.  Returns a copy; no target or post-hoc columns are
    touched.
    """

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["store"] = out["store"].astype(int)
    out["state_holiday"] = out["state_holiday"].astype(str)
    out["promo"] = out["promo"].astype(int)
    out["school_holiday"] = out["school_holiday"].astype(int)

    dates = out["date"]
    out["year"] = dates.dt.year.astype("int16")
    out["month"] = dates.dt.month.astype("int8")
    out["day"] = dates.dt.day.astype("int8")
    out["day_of_week"] = dates.dt.day_of_week.astype("int8")
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    out["day_of_year"] = dates.dt.day_of_year.astype("int16")
    out["is_month_start"] = dates.dt.is_month_start.astype("int8")
    out["is_month_end"] = dates.dt.is_month_end.astype("int8")
    out["is_weekend"] = (dates.dt.day_of_week >= 5).astype("int8")
    out["days_to_christmas"] = days_to_christmas(dates)

    competition_months = (
        (out["year"].astype("float64") - out["competition_open_since_year"]) * 12
        + (out["month"].astype("float64") - out["competition_open_since_month"])
    )
    out["competition_open_months"] = competition_months.clip(lower=0)

    week = dates.dt.isocalendar().week.astype("float64")
    promo2_weeks = (
        (out["year"].astype("float64") - out["promo2_since_year"]) * 52
        + (week - out["promo2_since_week"])
    )
    out["promo2_open_months"] = (promo2_weeks / 4.345).clip(lower=0)
    out["promo2"] = out["promo2"].fillna(0).astype("int8")

    out["promo_interval"] = out["promo_interval"].fillna("None").astype(str)
    out["store_type"] = out["store_type"].astype(str)
    out["assortment"] = out["assortment"].astype(str)

    return out


def attach_store(df: pd.DataFrame, store_metadata: dict[str, Any]) -> pd.DataFrame:
    """Attach static store metadata for inference records."""

    meta = pd.DataFrame.from_dict(store_metadata, orient="index")
    meta.index = meta.index.astype(int)
    meta = meta.reset_index().rename(columns={"index": "store"})
    meta["store"] = meta["store"].astype(int)

    out = df.drop(columns=[c for c in STORE_FIELDS if c in df.columns], errors="ignore")
    return out.merge(meta, on="store", how="left", validate="many_to_one")


def cast_categories(df: pd.DataFrame, categories: dict[str, list[Any]]) -> pd.DataFrame:
    """Cast categorical columns with the exact categories seen in training."""

    out = df.copy()
    for column, values in categories.items():
        out[column] = pd.Categorical(out[column], categories=values)
    return out


def build_inference_frame(
    records: list[dict[str, Any]],
    store_metadata: dict[str, Any],
    categories: dict[str, list[Any]],
) -> pd.DataFrame:
    """Turn API records into the exact feature frame the model was fitted on."""

    rows = [
        {
            "date": pd.Timestamp(record["date"]),
            "store": int(record["store"]),
            "promo": int(record["promo"]),
            "state_holiday": str(record["state_holiday"]),
            "school_holiday": int(record["school_holiday"]),
        }
        for record in records
    ]
    df = pd.DataFrame(rows)
    df = attach_store(df, store_metadata)
    df = enrich(df)
    df = cast_categories(df, categories)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURES]
