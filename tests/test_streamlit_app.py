from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture()
def app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=180).run()


def test_app_renders_without_errors(app: AppTest) -> None:
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert [title.value for title in app.title] == ["Plan the next store-day"]


def test_forecast_tab_shows_the_price_tag(app: AppTest) -> None:
    tags = [
        block.value
        for block in app.markdown
        if '<div class="price-tag">' in block.value
    ]
    assert len(tags) == 1
    assert "€4,829.68" in tags[0]


def test_switching_store_recomputes_the_forecast() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=180).run()
    before = [metric.value for metric in app.metric]
    app.selectbox(key="store").select(2).run()
    assert len(app.exception) == 0
    assert [metric.value for metric in app.metric] != before


def test_week_ahead_has_a_receipt_and_csv(app: AppTest) -> None:
    receipts = [
        block.value
        for block in app.markdown
        if '<div class="receipt">' in block.value
    ]
    assert len(receipts) == 1
    assert "Total" in receipts[0]
    assert "Download forecast CSV" in [button.label for button in app.download_button]


def test_model_card_reports_the_test_holdout(app: AppTest) -> None:
    labels = [metric.label for metric in app.metric]
    assert "Test RMSE" in labels
    assert "Test R²" in labels
