"""Train the Rossmann sales forecaster and save the artifact + metrics.

The split is temporal, not random:

    train       : 2013-01-01 .. 2015-05-31
    validation  : 2015-06-01 .. 2015-06-30
    test        : 2015-07-01 .. 2015-07-31

A store-median baseline, Ridge (linear), Random Forest, and XGBoost are
compared.  Ridge's alpha and XGBoost's tree/learning-rate settings are tuned on
the validation month.  The winner is refit on train + validation and evaluated
once on the held-out month.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from model import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_json

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed" / "rossmann.parquet"
MODELS = ROOT / "models"
RANDOM_STATE = 42

XGB_GRID = [
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 10, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.7},
]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def linear_pipeline(alpha: float = 1.0) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha, solver="lsqr"))])


def tree_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=14,
                    min_samples_leaf=5,
                    max_features=0.5,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def xgb_model(
    n_estimators: int = 1000,
    early_stopping: bool = True,
    **params: Any,
) -> XGBRegressor:
    defaults = {
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
    }
    defaults.update(params)
    return XGBRegressor(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50 if early_stopping else None,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **defaults,
    )


def store_median_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    medians = train.groupby("store", observed=True)["sales"].median()
    medians.index = medians.index.astype(int)
    fallback = float(train["sales"].median())
    return target["store"].astype(int).map(medians).fillna(fallback).to_numpy()


def tune_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_alpha, best_rmse = 1.0, float("inf")
    for alpha in (0.1, 1.0, 10.0):
        pipeline = linear_pipeline(alpha=alpha)
        pipeline.fit(X_train, y_train)
        rmse = float(
            np.sqrt(mean_squared_error(y_validation, pipeline.predict(X_validation)))
        )
        rows.append({"alpha": alpha, "validation_rmse": rmse})
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse
    return best_alpha, rows


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for params in XGB_GRID:
        model = xgb_model(**params)
        model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
        rmse = float(
            np.sqrt(mean_squared_error(y_validation, model.predict(X_validation)))
        )
        row = {
            **params,
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0) + 1,
            "validation_rmse": rmse,
        }
        rows.append(row)
        if best is None or rmse < best["validation_rmse"]:
            best = row
    assert best is not None
    return best, rows


def main() -> None:
    data_metadata = load_json(MODELS / "data_metadata.json")
    feature_metadata = load_json(MODELS / "feature_metadata.json")

    df = pd.read_parquet(PROCESSED)
    for column in CATEGORICAL_FEATURES:
        df[column] = pd.Categorical(
            df[column], categories=feature_metadata["categories"][column]
        )

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    X_train, y_train = train[FEATURES], train["sales"]
    X_validation, y_validation = validation[FEATURES], validation["sales"]
    X_test, y_test = test[FEATURES], test["sales"]

    results: dict[str, dict[str, Any]] = {}
    print(f"train {len(train):,} | validation {len(validation):,} | test {len(test):,}")

    # 1. Baseline: the store's training median.
    baseline_validation = store_median_baseline(train, validation)
    baseline_test = store_median_baseline(train, test)
    results["baseline"] = {
        "validation": regression_metrics(y_validation, baseline_validation),
        "test": regression_metrics(y_test, baseline_test),
    }
    print("baseline", results["baseline"]["validation"])

    # 2. Ridge, with alpha tuned on the validation month.
    started = time.perf_counter()
    best_alpha, ridge_tuning = tune_ridge(
        X_train, y_train, X_validation, y_validation
    )
    ridge = linear_pipeline(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    results["ridge"] = {
        "validation": regression_metrics(y_validation, ridge.predict(X_validation)),
        "test": regression_metrics(y_test, ridge.predict(X_test)),
        "alpha": best_alpha,
        "tuning": ridge_tuning,
    }
    print(f"ridge ({time.perf_counter() - started:.1f}s)", results["ridge"]["validation"])

    # 3. Random forest, a bagging benchmark.
    started = time.perf_counter()
    forest = tree_pipeline()
    forest.fit(X_train, y_train)
    results["random_forest"] = {
        "validation": regression_metrics(y_validation, forest.predict(X_validation)),
        "test": regression_metrics(y_test, forest.predict(X_test)),
    }
    print(
        f"random forest ({time.perf_counter() - started:.1f}s)",
        results["random_forest"]["validation"],
    )

    # 4. XGBoost, with the tree/learning-rate grid tuned on validation.
    started = time.perf_counter()
    best_xgb, xgb_tuning = tune_xgboost(
        X_train, y_train, X_validation, y_validation
    )
    best_iteration = int(best_xgb["best_iteration"])
    tuned_params = {
        key: best_xgb[key]
        for key in (
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        )
    }
    xgb = xgb_model(
        n_estimators=max(best_iteration, 100), early_stopping=False, **tuned_params
    )
    xgb.fit(X_train, y_train, verbose=False)
    results["xgboost"] = {
        "validation": regression_metrics(y_validation, xgb.predict(X_validation)),
        "test": regression_metrics(y_test, xgb.predict(X_test)),
        "params": tuned_params,
        "best_iteration": best_iteration,
        "tuning": xgb_tuning,
    }
    print(f"xgboost ({time.perf_counter() - started:.1f}s)", results["xgboost"]["validation"])

    candidates = ["ridge", "random_forest", "xgboost"]
    best_name = min(candidates, key=lambda name: results[name]["validation"]["rmse"])
    print(f"selected: {best_name}")

    # Refit the winner on train + validation, then touch the test month once.
    train_validation = pd.concat([train, validation], ignore_index=True)
    X_train_validation = train_validation[FEATURES]
    y_train_validation = train_validation["sales"]

    if best_name == "ridge":
        final_model: Any = linear_pipeline(alpha=best_alpha)
        final_model.fit(X_train_validation, y_train_validation)
    elif best_name == "random_forest":
        final_model = tree_pipeline()
        final_model.fit(X_train_validation, y_train_validation)
    else:
        final_model = xgb_model(
            n_estimators=max(best_iteration, 100),
            early_stopping=False,
            **tuned_params,
        )
        final_model.fit(X_train_validation, y_train_validation, verbose=False)

    final_test_predictions = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test, final_test_predictions)

    joblib.dump(final_model, MODELS / "model.joblib")

    model_metadata = {
        "model_version": "0.1.0",
        "model_type": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_on": "validation RMSE",
        "data": data_metadata,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "metrics": {
            "selected_validation": results[best_name]["validation"],
            "test": test_metrics,
            "baseline": results["baseline"],
            "all_models": results,
        },
        "features": {
            "all": FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "notes": (
            "Temporal split; closed and zero-sales days were filtered before training. "
            "The test month was not used for model selection or early stopping. "
            "Ridge alpha and XGBoost depth/learning-rate were tuned on the validation month."
        ),
    }
    (MODELS / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2), encoding="utf-8"
    )

    print("test", test_metrics)
    print(f"saved {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
