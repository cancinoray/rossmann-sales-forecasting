from __future__ import annotations

from fastapi.testclient import TestClient

from predict import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["model_sha256"]) == 64


def test_predict_returns_positive_sales() -> None:
    response = client.post(
        "/predict",
        json={
            "store": 1,
            "date": "2015-08-15",
            "promo": 1,
            "state_holiday": "0",
            "school_holiday": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["store"] == 1
    assert body["predicted_sales"] > 0
    assert body["model_version"]


def test_predict_is_deterministic() -> None:
    payload = {
        "store": 2,
        "date": "2015-08-16",
        "promo": 0,
        "state_holiday": "0",
        "school_holiday": 1,
    }
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first == second


def test_unknown_store_is_404() -> None:
    response = client.post("/predict", json={"store": 9999, "date": "2015-08-15"})
    assert response.status_code == 404


def test_date_outside_training_window_is_422() -> None:
    response = client.post(
        "/predict",
        json={"store": 1, "date": "2016-01-01"},
    )
    assert response.status_code == 422


def test_invalid_promo_is_422() -> None:
    response = client.post(
        "/predict",
        json={"store": 1, "date": "2015-08-15", "promo": 2},
    )
    assert response.status_code == 422


def test_invalid_state_holiday_is_422() -> None:
    response = client.post(
        "/predict",
        json={"store": 1, "date": "2015-08-15", "state_holiday": "x"},
    )
    assert response.status_code == 422
