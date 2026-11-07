from __future__ import annotations

import pytest

from serving import DateOutOfRangeError, SalesForecaster, UnknownStoreError


@pytest.fixture(scope="module")
def forecaster() -> SalesForecaster:
    return SalesForecaster()


def test_health_exposes_the_artifact_hash(forecaster: SalesForecaster) -> None:
    health = forecaster.health()
    assert health["status"] == "ok"
    assert health["model_type"] == "xgboost"
    assert len(health["model_sha256"]) == 64


def test_prediction_matches_the_documented_example(
    forecaster: SalesForecaster,
) -> None:
    # README §API. Update this assertion if the artifact is retrained.
    prediction = forecaster.predict(1, "2015-08-15", promo=1)
    assert prediction == pytest.approx(4829.68, abs=0.01)


def test_promo_raises_the_forecast(forecaster: SalesForecaster) -> None:
    promo = forecaster.predict(1, "2015-08-15", promo=1)
    no_promo = forecaster.predict(1, "2015-08-15", promo=0)
    assert promo > no_promo


def test_batch_matches_single_predictions(forecaster: SalesForecaster) -> None:
    records = [
        {"store": 1, "date": "2015-08-15", "promo": 1},
        {"store": 2, "date": "2015-08-16", "promo": 0},
        {"store": 3, "date": "2015-08-17", "promo": 1},
    ]
    batch = forecaster.predict_many(records)
    singles = [
        forecaster.predict(
            record["store"],
            record["date"],
            promo=record["promo"],
        )
        for record in records
    ]
    assert batch == pytest.approx(singles)


def test_empty_batch_returns_nothing(forecaster: SalesForecaster) -> None:
    assert forecaster.predict_many([]) == []


def test_features_are_the_training_features(forecaster: SalesForecaster) -> None:
    features = forecaster.features(1, "2015-08-15", promo=1)
    assert list(features) == forecaster.model_metadata["features"]["all"]
    assert len(features) == 21
    assert {"sales", "customers", "open"}.isdisjoint(features)


def test_unknown_store_raises(forecaster: SalesForecaster) -> None:
    with pytest.raises(UnknownStoreError):
        forecaster.predict(9999, "2015-08-15")


def test_date_outside_the_window_raises(forecaster: SalesForecaster) -> None:
    with pytest.raises(DateOutOfRangeError):
        forecaster.predict(1, "2016-01-01")


def test_store_profile_is_static_metadata(forecaster: SalesForecaster) -> None:
    profile = forecaster.store_profile(1)
    assert profile["store_type"] == "c"
    assert profile["assortment"] == "a"
