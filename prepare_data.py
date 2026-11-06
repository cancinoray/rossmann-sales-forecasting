"""Download (if needed), clean, and split the Rossmann dataset.

Outputs
-------
data/processed/rossmann.parquet
    One row per open store-day with the target, split label, and model features.
models/store_metadata.json
    Static store attributes used at inference time.
models/feature_metadata.json
    Feature order and the exact categorical levels seen in training.
models/data_metadata.json
    Checksums, row counts, and split boundaries for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, enrich

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TRAIN_URL = (
    "https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales/resolve/main/train.csv"
)
STORE_URL = (
    "https://huggingface.co/datasets/AiiN-aini/rossmann-store-sales/resolve/main/store.csv"
)
TRAIN_SHA256 = "f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc"
STORE_SHA256 = "f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344"

TRAIN_END = pd.Timestamp("2015-06-01")
VALIDATION_END = pd.Timestamp("2015-07-01")

RENAME = {
    "Store": "store",
    "DayOfWeek": "day_of_week",
    "Date": "date",
    "Sales": "sales",
    "Customers": "customers",
    "Open": "open",
    "Promo": "promo",
    "StateHoliday": "state_holiday",
    "SchoolHoliday": "school_holiday",
    "StoreType": "store_type",
    "Assortment": "assortment",
    "CompetitionDistance": "competition_distance",
    "CompetitionOpenSinceMonth": "competition_open_since_month",
    "CompetitionOpenSinceYear": "competition_open_since_year",
    "Promo2": "promo2",
    "Promo2SinceWeek": "promo2_since_week",
    "Promo2SinceYear": "promo2_since_year",
    "PromoInterval": "promo_interval",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "rossmann-sales-forecasting/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_raw() -> tuple[Path, Path]:
    train_path = RAW / "train.csv"
    store_path = RAW / "store.csv"
    download(TRAIN_URL, train_path)
    download(STORE_URL, store_path)

    for path, expected in ((train_path, TRAIN_SHA256), (store_path, STORE_SHA256)):
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(
                f"checksum mismatch for {path.name}: expected {expected}, got {actual}. "
                "Delete the file and re-run, or verify the source URL."
            )
    return train_path, store_path


def jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_store_metadata(store: pd.DataFrame) -> dict[str, dict[str, Any]]:
    store = store.rename(columns=RENAME)
    metadata: dict[str, dict[str, Any]] = {}
    for row in store.to_dict(orient="records"):
        key = str(int(row["store"]))
        metadata[key] = {
            column: jsonable(value) for column, value in row.items() if column != "store"
        }
    return metadata


def main() -> None:
    train_path, store_path = ensure_raw()

    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path)
    store_metadata = build_store_metadata(store)

    df = train.rename(columns=RENAME).merge(
        store.rename(columns=RENAME), on="store", how="left", validate="many_to_one"
    )

    # Keep only open days with positive sales.  Closed days are structurally
    # zero and would make the regression learn an `Open` shortcut.
    before = len(df)
    df = df[(df["open"] == 1) & (df["sales"] > 0)].copy()
    print(f"filtered closed/zero-sales rows: {before - len(df):,} of {before:,}")

    df["state_holiday"] = (
        df["state_holiday"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    df = enrich(df)

    df["split"] = np.select(
        [df["date"] < TRAIN_END, df["date"] < VALIDATION_END],
        ["train", "validation"],
        default="test",
    )

    categories = {
        column: [jsonable(value) for value in sorted(df[column].dropna().unique().tolist())]
        for column in CATEGORICAL_FEATURES
    }
    feature_metadata = {
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categories": categories,
    }

    MODELS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (MODELS / "store_metadata.json").write_text(
        json.dumps(store_metadata, indent=2), encoding="utf-8"
    )
    (MODELS / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, indent=2), encoding="utf-8"
    )

    output = df[["date", "sales", "split", *FEATURES]].copy()
    output.to_parquet(PROCESSED / "rossmann.parquet", index=False)

    counts = output["split"].value_counts().to_dict()
    data_metadata = {
        "source": {
            "train": TRAIN_URL,
            "store": STORE_URL,
            "original_competition": "https://www.kaggle.com/competitions/rossmann-store-sales",
        },
        "raw_sha256": {"train": TRAIN_SHA256, "store": STORE_SHA256},
        "processed_sha256": sha256(PROCESSED / "rossmann.parquet"),
        "rows": int(len(output)),
        "split_rows": {key: int(value) for key, value in counts.items()},
        "date_min": output["date"].min().isoformat(),
        "date_max": output["date"].max().isoformat(),
        "split_dates": {
            "train": f"< {TRAIN_END.date()}",
            "validation": f"{TRAIN_END.date()} to < {VALIDATION_END.date()}",
            "test": f">= {VALIDATION_END.date()}",
        },
        "filters": {"open": 1, "sales_gt": 0},
    }
    (MODELS / "data_metadata.json").write_text(
        json.dumps(data_metadata, indent=2), encoding="utf-8"
    )

    print(f"rows kept: {len(output):,}")
    print(f"split rows: {counts}")
    print(f"date range: {data_metadata['date_min']} -> {data_metadata['date_max']}")
    print(f"processed: {PROCESSED / 'rossmann.parquet'}")


if __name__ == "__main__":
    main()
