"""Streamlit dashboard for the Rossmann daily-sales forecaster.

Run it with ``make ui`` (or ``uv run streamlit run streamlit_app.py``).  The
page reads the same committed artifact as the FastAPI service through
``serving.SalesForecaster``; predictions are identical to ``POST /predict``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from history import (
    DAY_NAMES,
    load_history,
    monthly_means,
    promo_by_weekday,
    store_frame,
    store_stats,
    weekday_means,
)
from model import STORE_FIELDS
from serving import MAX_DATE, MIN_DATE, SalesForecaster

DEFAULT_STORE = 1
DEFAULT_DATE = date(2015, 8, 15)
DEFAULT_WEEK_START = date(2015, 8, 1)
MAX_HORIZON = 28
STATE_HOLIDAYS = {
    "0": "None",
    "a": "Public holiday",
    "b": "Easter",
    "c": "Christmas",
}
MODEL_LABELS = {
    "baseline": "Store median",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}

st.set_page_config(
    page_title="Rossmann forecast desk",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --paper: #F7F4EC;
    --card: #FFFDF8;
    --ink: #101A20;
    --muted: #5A6B75;
    --line: #D9D2C3;
    --red: #C8102E;
    --blue: #1F4E66;
}
.stApp { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #EFEADF;
    border-right: 1px solid var(--line);
}
[data-testid="stAppDeployButton"] { display: none; }
h1, h2, h3, h4, h5 { letter-spacing: -0.02em; color: var(--ink); }
p, li { color: var(--ink); }
.eyebrow {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.7rem;
    color: var(--red);
    margin-bottom: 0.45rem;
}
.lede { color: var(--muted); max-width: 64ch; }
.rule { border-top: 2px solid var(--ink); margin: 0.2rem 0 1.4rem; }
.price-tag {
    position: relative;
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 6px solid var(--red);
    padding: 1.2rem 1.5rem 1.1rem;
    box-shadow: 0 2px 0 rgba(16, 26, 32, 0.06);
}
.price-tag::after {
    content: "";
    position: absolute;
    top: 0.85rem;
    right: 0.85rem;
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 50%;
    background: var(--paper);
    border: 1px solid var(--line);
}
.price-tag .label {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
}
.price-tag .value {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 3.1rem;
    line-height: 1.05;
    font-weight: 600;
    color: var(--ink);
    margin: 0.45rem 0 0.3rem;
}
.price-tag .note { color: var(--muted); font-size: 0.86rem; }
.receipt {
    background: var(--card);
    border: 1px solid var(--line);
    padding: 1rem 1.35rem 1.3rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    box-shadow: 0 2px 0 rgba(16, 26, 32, 0.06);
}
.receipt .head {
    display: flex;
    justify-content: space-between;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.7rem;
    color: var(--muted);
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.55rem;
    margin-bottom: 0.35rem;
}
.receipt .row {
    display: flex;
    justify-content: space-between;
    padding: 0.28rem 0;
    font-size: 0.86rem;
    border-bottom: 1px dotted var(--line);
}
.receipt .promo {
    color: var(--red);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 0.45rem;
}
.receipt .is-closed { color: var(--muted); }
.receipt .closed {
    color: var(--muted);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-left: 0.45rem;
}
.receipt .total {
    display: flex;
    justify-content: space-between;
    border-top: 2px solid var(--ink);
    margin-top: 0.55rem;
    padding-top: 0.7rem;
    font-weight: 700;
    font-size: 1rem;
}
.receipt .foot {
    margin-top: 0.75rem;
    color: var(--muted);
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    padding: 0.7rem 0.9rem;
}
[data-testid="stMetricValue"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.5rem;
    line-height: 1.25;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; }
[data-baseweb="tab"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.76rem;
}
[data-baseweb="tab-highlight"] { background-color: var(--red); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model…")
def get_forecaster() -> SalesForecaster:
    return SalesForecaster()


@st.cache_data(show_spinner="Loading store history…")
def get_history() -> pd.DataFrame:
    return load_history()


@st.cache_data(show_spinner=False)
def get_monthly(_history: pd.DataFrame) -> pd.DataFrame:
    return monthly_means(_history)


@st.cache_data(show_spinner=False)
def get_promo_by_weekday(_history: pd.DataFrame) -> pd.DataFrame:
    return promo_by_weekday(_history)


def money(value: float, decimals: int = 0) -> str:
    return f"€{value:,.{decimals}f}"


def signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}€{abs(value):,.0f}"


def day_label(when: date) -> str:
    return when.strftime("%a %d %b %Y")


try:
    forecaster = get_forecaster()
except FileNotFoundError:
    st.error(
        "No model artifact found at `models/model.joblib`. "
        "Run `make train` (or restore the committed artifact) and reload."
    )
    st.stop()

try:
    history: pd.DataFrame | None = get_history()
except FileNotFoundError:
    history = None
    st.warning("Historical parquet is missing; comparison tables and charts are hidden.")

with st.sidebar:
    st.markdown('<div class="eyebrow">Rossmann · forecast desk</div>', unsafe_allow_html=True)
    store = st.selectbox(
        "Store",
        forecaster.store_ids,
        index=forecaster.store_ids.index(DEFAULT_STORE),
        format_func=lambda value: f"Store {value}",
        key="store",
    )
    profile = forecaster.store_profile(store)

    distance = profile.get("competition_distance")
    competitor_since = "—"
    if profile.get("competition_open_since_month") and profile.get(
        "competition_open_since_year"
    ):
        competitor_since = (
            f"{int(profile['competition_open_since_month']):02d}/"
            f"{int(profile['competition_open_since_year'])}"
        )

    st.markdown("##### Store profile")
    st.dataframe(
        pd.DataFrame(
            {
                "Attribute": [
                    "Type",
                    "Assortment",
                    "Competition",
                    "Competitor since",
                    "Promo2",
                ],
                "Value": [
                    str(profile["store_type"]).upper(),
                    str(profile["assortment"]).upper(),
                    f"{distance:,.0f} m" if distance is not None else "—",
                    competitor_since,
                    "Yes" if profile.get("promo2") else "No",
                ],
            }
        ),
        hide_index=True,
        height=204,
    )

    st.divider()
    st.caption(f"Model {forecaster.model_version} · {forecaster.model_type.upper()}")
    st.caption(f"SHA-256 `{forecaster.model_sha256[:12]}…`")
    st.caption(
        "Forecasts cover open store-days only. Closed and zero-sales days were "
        "removed before training."
    )

st.markdown(
    f'<div class="eyebrow">Daily sales forecast · {len(forecaster.store_ids):,} stores</div>',
    unsafe_allow_html=True,
)
st.title("Plan the next store-day")
st.markdown(
    '<p class="lede">Pick a store and a date. You get a sales forecast from the '
    "tuned XGBoost model, how it compares with that store's usual trading, the "
    "week ahead as a receipt, and the evidence behind the model.</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

if history is not None:
    store_history = store_frame(history, store)
    stats = store_stats(store_history)
    weekday_reference = weekday_means(store_history)
else:
    store_history, stats, weekday_reference = None, None, None

forecast_tab, week_tab, model_tab, data_tab = st.tabs(
    ["Forecast", "Week ahead", "Model card", "Data"]
)

with forecast_tab:
    controls, result = st.columns([1, 1.05], gap="large")

    with controls:
        st.markdown("#### Store-day")
        forecast_date = st.date_input(
            "Date",
            value=DEFAULT_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            format="DD/MM/YYYY",
            key="forecast_date",
            help="The model was trained through July 2015; later dates use calendar features only.",
        )
        promo = st.radio(
            "Promotion",
            options=[0, 1],
            index=1,
            format_func=lambda value: "Promo day" if value else "No promo",
            horizontal=True,
            key="promo",
        )
        holiday = st.selectbox(
            "State holiday",
            options=list(STATE_HOLIDAYS),
            format_func=STATE_HOLIDAYS.get,
            key="holiday",
        )
        school = st.checkbox("School holiday", key="school")

    prediction = forecaster.predict(
        store,
        forecast_date,
        promo=promo,
        state_holiday=holiday,
        school_holiday=int(school),
    )
    promo_on = forecaster.predict(
        store, forecast_date, promo=1, state_holiday=holiday, school_holiday=int(school)
    )
    promo_off = forecaster.predict(
        store, forecast_date, promo=0, state_holiday=holiday, school_holiday=int(school)
    )

    context = [f"Store {store}", "promo" if promo else "no promo"]
    if holiday != "0":
        context.append(STATE_HOLIDAYS[holiday].lower())
    if school:
        context.append("school holiday")

    with result:
        st.markdown(
            f"""
            <div class="price-tag">
              <div class="label">Forecast · {day_label(forecast_date)}</div>
              <div class="value">{money(prediction, 2)}</div>
              <div class="note">{" · ".join(context)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if stats is not None and weekday_reference is not None:
            weekday_name = DAY_NAMES[forecast_date.weekday()]
            weekday_average = weekday_reference[weekday_name]
            reference_columns = st.columns(3)
            reference_columns[0].metric("Store median", money(stats["median"]))
            reference_columns[1].metric("Last 28 days", money(stats["recent_mean"]))
            reference_columns[2].metric(
                f"{weekday_name} average",
                money(weekday_average) if pd.notna(weekday_average) else "—",
                help=(
                    None
                    if pd.notna(weekday_average)
                    else f"Store {store} recorded no open {weekday_name} days."
                ),
            )
            change = (prediction - stats["recent_mean"]) / stats["recent_mean"] * 100
            st.caption(
                f"Forecast is **{change:+.1f}%** vs the store's last 28 trading days "
                f"({stats['recent_days']} open days)."
            )
            if pd.isna(weekday_average):
                st.info(
                    f"Store {store} has no open {weekday_name} days in the training "
                    "window, so this forecast is hypothetical."
                )
        else:
            st.caption("Historical comparison unavailable.")

        lift = promo_on - promo_off
        lift_pct = lift / promo_off * 100 if promo_off else 0.0
        st.markdown("##### Promotion effect")
        st.markdown(
            f"The same store-day forecasts **{money(promo_on, 2)}** with a promotion "
            f"and **{money(promo_off, 2)}** without one — a **{signed_money(lift)}** "
            f"difference ({lift_pct:+.1f}%)."
        )

        with st.expander("What the model sees"):
            features = forecaster.features(
                store,
                forecast_date,
                promo=promo,
                state_holiday=holiday,
                school_holiday=int(school),
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": list(features),
                        "Value": [str(value) for value in features.values()],
                    }
                ),
                hide_index=True,
                height=420,
            )
            st.caption(
                "21 features built from the date, promotion and holiday flags, and "
                "static store metadata. `Customers`, `Sales`, and `Open` never enter "
                "the feature frame."
            )

with week_tab:
    st.markdown("#### Week ahead")
    start_column, horizon_column, promo_column = st.columns([1, 1, 1.4])
    with start_column:
        week_start = st.date_input(
            "Start date",
            value=DEFAULT_WEEK_START,
            min_value=MIN_DATE,
            max_value=MAX_DATE - timedelta(days=MAX_HORIZON - 1),
            format="DD/MM/YYYY",
            key="week_start",
        )
    with horizon_column:
        horizon = st.slider("Days", 7, MAX_HORIZON, 14, key="horizon")
    with promo_column:
        promo_days = st.multiselect(
            "Promo weekdays",
            options=list(range(7)),
            default=[0, 1, 2, 3, 4],
            format_func=lambda day: DAY_NAMES[day],
            key="promo_days",
        )
    school_window = st.checkbox("School holiday for the whole window", key="week_school")

    days = [week_start + timedelta(days=offset) for offset in range(horizon)]
    promo_flags = [day.weekday() in promo_days for day in days]

    if weekday_reference is not None:
        open_weekdays = set(weekday_reference.dropna().index)
        open_days = [DAY_NAMES[day.weekday()] in open_weekdays for day in days]
    else:
        open_days = [True] * len(days)

    records = [
        {
            "store": store,
            "date": day,
            "promo": int(flag),
            "state_holiday": "0",
            "school_holiday": int(school_window),
        }
        for day, flag, is_open in zip(days, promo_flags, open_days)
        if is_open
    ]
    predicted = forecaster.predict_many(records)

    forecast_values: list[float | None] = []
    predicted_iter = iter(predicted)
    for is_open in open_days:
        forecast_values.append(next(predicted_iter) if is_open else None)

    open_values = [value for value in forecast_values if value is not None]
    total = float(sum(open_values))
    average = total / len(open_values) if open_values else 0.0
    peak = max(
        (index for index, value in enumerate(forecast_values) if value is not None),
        key=lambda index: forecast_values[index],
        default=0,
    )

    table = pd.DataFrame(
        {
            "Date": days,
            "Weekday": [DAY_NAMES[day.weekday()] for day in days],
            "Open": ["yes" if is_open else "no" for is_open in open_days],
            "Promo": [
                "yes" if flag and is_open else "no"
                for flag, is_open in zip(promo_flags, open_days)
            ],
            "Forecast (€)": [
                round(value, 2) if value is not None else None
                for value in forecast_values
            ],
        }
    )

    summary = st.columns(3)
    summary[0].metric("Horizon total", money(total))
    summary[1].metric("Open-day average", money(average))
    if open_values:
        peak_day = days[peak]
        summary[2].metric(
            f"Peak day · {DAY_NAMES[peak_day.weekday()]} {peak_day.strftime('%d %b')}",
            money(forecast_values[peak]),
        )
    else:
        summary[2].metric("Peak day", "—")

    closed_weekdays = sorted(
        {
            DAY_NAMES[day.weekday()]
            for day, is_open in zip(days, open_days)
            if not is_open
        },
        key=DAY_NAMES.index,
    )
    if closed_weekdays:
        st.caption(
            f"Closed days are excluded from the totals: store {store} has no open "
            f"{', '.join(closed_weekdays)} days in the training data."
        )

    def receipt_row(day: date, flag: bool, value: float | None) -> str:
        label = f"{DAY_NAMES[day.weekday()]} {day.strftime('%d %b')}"
        if value is None:
            return (
                f'<div class="row is-closed"><span>{label}'
                '<span class="closed">closed</span></span><span>—</span></div>'
            )
        badge = '<span class="promo">promo</span>' if flag else ""
        return (
            f'<div class="row"><span>{label}{badge}</span>'
            f"<span>{money(value, 2)}</span></div>"
        )

    receipt_column, chart_column = st.columns([0.9, 1.1], gap="large")
    with receipt_column:
        rows_html = "".join(
            receipt_row(day, flag, value)
            for day, flag, value in zip(days, promo_flags, forecast_values)
        )
        st.markdown(
            f"""
            <div class="receipt">
              <div class="head"><span>Store {store} · forecast</span><span>{horizon} days</span></div>
              {rows_html}
              <div class="total"><span>Total</span><span>{money(total)}</span></div>
              <div class="foot">{week_start:%d %b} – {days[-1]:%d %b %Y}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with chart_column:
        st.markdown("##### Daily forecast")
        st.bar_chart(
            table.dropna(subset=["Forecast (€)"]).set_index("Date")[["Forecast (€)"]],
            color="#C8102E",
            height=360,
            y_label="euros",
        )

    with st.expander("Table and CSV"):
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn(format="DD/MM/YYYY"),
                "Forecast (€)": st.column_config.NumberColumn(format="euro"),
            },
        )
        st.download_button(
            "Download forecast CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"rossmann-store-{store}-forecast-{week_start.isoformat()}.csv",
            mime="text/csv",
            key="download_week",
        )
        st.caption(
            "State holidays are not applied in this window; use the single-day tab "
            "to test one."
        )

with model_tab:
    metrics = forecaster.model_metadata["metrics"]
    test_metrics = metrics["test"]
    baseline_metrics = metrics["baseline"]["test"]
    reduction = 1 - test_metrics["rmse"] / baseline_metrics["rmse"]

    headline = st.columns(4)
    headline[0].metric("Test RMSE", money(test_metrics["rmse"], 2))
    headline[1].metric("Test MAE", money(test_metrics["mae"], 2))
    headline[2].metric("Test R²", f"{test_metrics['r2']:.3f}")
    headline[3].metric("RMSE vs baseline", f"−{reduction:.0%}")
    st.caption(
        f"July 2015 holdout, {forecaster.model_metadata['rows']['test']:,} open "
        "store-days. The test month was never used for selection or early stopping."
    )

    comparison_rows: list[dict[str, Any]] = []
    for name, model_metrics in metrics["all_models"].items():
        comparison_rows.append(
            {
                "Model": MODEL_LABELS.get(name, name),
                "Selected": "●" if name == forecaster.model_type else "",
                "Validation RMSE": model_metrics["validation"]["rmse"],
                "Validation MAE": model_metrics["validation"]["mae"],
                "Test RMSE": model_metrics["test"]["rmse"],
                "Test MAE": model_metrics["test"]["mae"],
                "Test R²": model_metrics["test"]["r2"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    st.markdown("##### Model comparison")
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "Validation RMSE": st.column_config.NumberColumn(format="euro"),
            "Validation MAE": st.column_config.NumberColumn(format="euro"),
            "Test RMSE": st.column_config.NumberColumn(format="euro"),
            "Test MAE": st.column_config.NumberColumn(format="euro"),
            "Test R²": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.bar_chart(
        comparison.set_index("Model")[["Test RMSE"]],
        color="#1F4E66",
        height=260,
        horizontal=True,
        y_label="euros",
    )
    st.caption(
        "Comparison rows are scored as fitted on train. The headline numbers are "
        "the served model — the XGBoost winner refit on train + validation — so its "
        f"test RMSE ({money(test_metrics['rmse'], 2)}) differs from the "
        f"{money(metrics['all_models']['xgboost']['test']['rmse'], 2)} measured "
        "before the refit."
    )

    tuning = pd.DataFrame(metrics["all_models"]["xgboost"]["tuning"]).rename(
        columns={
            "max_depth": "Max depth",
            "learning_rate": "Learning rate",
            "subsample": "Subsample",
            "colsample_bytree": "Column sample",
            "best_iteration": "Best iteration",
            "validation_rmse": "Validation RMSE",
        }
    )
    best_row = tuning.loc[tuning["Validation RMSE"].idxmin()]
    st.markdown("##### Tuning on the validation month")
    st.dataframe(
        tuning,
        hide_index=True,
        column_config={
            "Validation RMSE": st.column_config.NumberColumn(format="euro")
        },
    )
    st.caption(
        f"Selected: depth {int(best_row['Max depth'])}, learning rate "
        f"{best_row['Learning rate']}, {int(best_row['Best iteration'])} boosting "
        f"rounds — validation RMSE {money(best_row['Validation RMSE'], 2)}. The "
        "final model is refit on train + validation."
    )

    with st.expander("Selected parameters"):
        st.code(
            json.dumps(metrics["all_models"]["xgboost"]["params"], indent=2),
            language="json",
        )

    with st.expander("Features"):
        feature_columns = st.columns(3)
        groups = [
            ("Categorical", forecaster.model_metadata["features"]["categorical"]),
            ("Numeric", forecaster.model_metadata["features"]["numeric"]),
            ("Store metadata", list(STORE_FIELDS)),
        ]
        for column, (title, names) in zip(feature_columns, groups):
            column.markdown(f"**{title}**")
            column.markdown("\n".join(f"- `{name}`" for name in names))

    with st.expander("Artifact and reproducibility"):
        st.markdown(
            f"**Version** `{forecaster.model_version}` · **trained** "
            f"{forecaster.model_metadata['trained_at']}"
        )
        st.code(forecaster.model_sha256, language=None)
        rows = forecaster.model_metadata["rows"]
        st.markdown(
            f"**Rows** train {rows['train']:,} · validation {rows['validation']:,} · "
            f"test {rows['test']:,}"
        )
        st.markdown(
            "**Split** train < 2015-06-01 · validation June 2015 · test July 2015"
        )
        st.json(forecaster.model_metadata["library_versions"])

with data_tab:
    if history is None or store_history is None:
        st.info("Historical data is unavailable. Run `make data` to rebuild the parquet.")
    else:
        dataset = forecaster.model_metadata["data"]
        dataset_columns = st.columns(3)
        dataset_columns[0].metric("Modelled store-days", f"{dataset['rows']:,}")
        dataset_columns[1].metric("Stores", f"{len(forecaster.store_ids):,}")
        dataset_columns[2].metric("Period", "Jan 2013 – Jul 2015")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### Average sales by month")
            st.line_chart(
                get_monthly(history),
                x="month",
                y="Mean sales",
                color="#1F4E66",
                height=300,
                y_label="euros",
            )
        with right:
            st.markdown("##### Promo lift by weekday")
            st.bar_chart(
                get_promo_by_weekday(history).reset_index(names="Weekday"),
                x="Weekday",
                y=["Promo", "No promo"],
                color=["#C8102E", "#8A9AA5"],
                stack=False,
                sort=False,
                height=300,
                y_label="euros",
            )

        st.markdown(f"##### Store {store}: weekday rhythm")
        st.bar_chart(
            weekday_reference.rename("Mean sales")
            .to_frame()
            .reset_index(names="Weekday"),
            x="Weekday",
            y="Mean sales",
            color="#C8102E",
            sort=False,
            height=260,
            y_label="euros",
        )
        st.caption(
            "All charts use open store-days with positive sales. Closed days and "
            "zero-sales days (172,871 rows) are excluded from training and from "
            "these views."
        )

st.divider()
st.caption(
    f"Rossmann Store Sales · model {forecaster.model_version} · "
    f"artifact `{forecaster.model_sha256[:16]}…` · API on `make serve` → :9696"
)
