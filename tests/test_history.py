from __future__ import annotations

import pytest

from history import (
    DAY_NAMES,
    load_history,
    monthly_means,
    promo_by_weekday,
    store_frame,
    store_stats,
    weekday_means,
)


@pytest.fixture(scope="module")
def history():
    return load_history()


@pytest.fixture(scope="module")
def store_history(history):
    return store_frame(history, 1)


def test_store_frame_raises_for_an_unknown_store(history) -> None:
    with pytest.raises(ValueError):
        store_frame(history, 9999)


def test_store_stats_summarise_a_store(store_history) -> None:
    stats = store_stats(store_history)
    assert stats["median"] > 0
    assert 0 < stats["recent_days"] <= 28
    assert stats["p25"] < stats["median"] < stats["p75"]
    assert stats["days"] == len(store_history)


def test_weekday_means_cover_the_week(store_history) -> None:
    means = weekday_means(store_history)
    assert list(means.index) == DAY_NAMES
    # Rossmann stores are mostly closed on Sundays, so a NaN is expected.
    assert means.notna().sum() >= 5
    assert (means.dropna() > 0).all()


def test_monthly_means_span_the_dataset(history) -> None:
    monthly = monthly_means(history)
    assert len(monthly) == 31
    assert list(monthly.columns) == ["month", "Mean sales"]
    assert monthly["Mean sales"].notna().all()


def test_promo_by_weekday_has_both_conditions(history) -> None:
    lift = promo_by_weekday(history)
    assert list(lift.index) == DAY_NAMES
    assert list(lift.columns) == ["Promo", "No promo"]
    # Sundays are closed for most stores; the other weekdays must be present.
    assert lift.dropna().shape[0] >= 5
