"""Historical context for the dashboard, read from the processed parquet.

The API only needs the artifact; the dashboard also shows what a store
normally sells so a reviewer can judge a forecast against something real.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / "rossmann.parquet"
HISTORY_COLUMNS = [
    "date",
    "store",
    "sales",
    "promo",
    "day_of_week",
    "state_holiday",
    "school_holiday",
    "split",
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_history(path: str | Path = PROCESSED) -> pd.DataFrame:
    """Read only the columns the dashboard uses."""

    return pd.read_parquet(path, columns=HISTORY_COLUMNS)


def store_frame(history: pd.DataFrame, store: int) -> pd.DataFrame:
    """All open store-days for one store."""

    frame = history[history["store"] == int(store)]
    if frame.empty:
        raise ValueError(f"no history for store {store}")
    return frame


def store_stats(frame: pd.DataFrame, window_days: int = 28) -> dict[str, float]:
    """Typical sales for a store, plus its most recent trading window."""

    last_date = frame["date"].max()
    recent = frame[frame["date"] > last_date - pd.Timedelta(days=window_days)]
    promo = frame.loc[frame["promo"] == 1, "sales"]
    no_promo = frame.loc[frame["promo"] == 0, "sales"]

    return {
        "median": float(frame["sales"].median()),
        "mean": float(frame["sales"].mean()),
        "p25": float(frame["sales"].quantile(0.25)),
        "p75": float(frame["sales"].quantile(0.75)),
        "recent_mean": float(recent["sales"].mean()),
        "recent_days": len(recent),
        "promo_mean": float(promo.mean()) if len(promo) else float("nan"),
        "no_promo_mean": float(no_promo.mean()) if len(no_promo) else float("nan"),
        "days": len(frame),
    }


def weekday_means(frame: pd.DataFrame) -> pd.Series:
    """Average sales by weekday, Monday first, for one store."""

    means = frame.groupby("day_of_week", observed=True)["sales"].mean()
    return pd.Series(
        [float(means.get(index, float("nan"))) for index in range(7)],
        index=DAY_NAMES,
        name="Mean sales",
    )


def monthly_means(history: pd.DataFrame) -> pd.DataFrame:
    """Average sales per calendar month across all stores."""

    monthly = (
        history.assign(month=history["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", observed=True)["sales"]
        .mean()
        .reset_index()
        .rename(columns={"sales": "Mean sales"})
    )
    return monthly


def promo_by_weekday(history: pd.DataFrame) -> pd.DataFrame:
    """Average sales on promo vs non-promo days, split by weekday."""

    grouped = (
        history.assign(weekday=history["day_of_week"].map(dict(enumerate(DAY_NAMES))))
        .pivot_table(
            index="weekday",
            columns="promo",
            values="sales",
            aggfunc="mean",
            observed=True,
        )
        .reindex(DAY_NAMES)
    )
    grouped = grouped.rename(columns={0: "No promo", 1: "Promo"})
    return grouped[["Promo", "No promo"]].astype(float)
