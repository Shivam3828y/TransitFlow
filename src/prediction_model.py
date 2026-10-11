from __future__ import annotations

from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MODELS_DIR
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    prepare_prediction_dataset,
)


def _make_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    result = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) >= 2 and np.unique(actual).size > 1
            else None
        ),
    }
    return result


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    n_estimators: int = 160,
    random_state: int = 42,
) -> dict:
    """Chronologically evaluate a lag baseline and forest, then fit for forecasting."""
    if n_estimators < 10 or n_estimators > 500:
        raise ValueError("Choose between 10 and 500 Random Forest trees.")

    x, y, engineered = prepare_prediction_dataset(frame)
    if len(x) > 100_000:
        raise ValueError(
            f"Feature preparation produced {len(x):,} rows. For model training, "
            "use the sidebar filters to reduce the dataset to 100,000 rows or fewer."
        )
    unique_days = np.array(sorted(engineered["timestamp"].dt.normalize().unique()))
    if len(x) < 50 or len(unique_days) < 14:
        raise ValueError(
            f"Only {len(x):,} rows with usable history across {len(unique_days):,} "
            "days. Reliable training needs at least 50 feature rows across 14 dates."
        )

    cutoff_index = int(len(unique_days) * 0.8)
    cutoff_index = min(max(cutoff_index, 1), len(unique_days) - 1)
    cutoff = pd.Timestamp(unique_days[cutoff_index])
    train_mask = engineered["timestamp"].dt.normalize() < cutoff
    test_mask = ~train_mask
    x_train, x_test = x.loc[train_mask], x.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    if len(x_train) < 20 or len(x_test) < 5:
        raise ValueError(
            "Chronological split leaves too little training or evaluation data."
        )

    evaluation_model = _make_pipeline(n_estimators, random_state)
    evaluation_model.fit(x_train, y_train)
    forest_prediction = np.clip(evaluation_model.predict(x_test), 0, None)
    baseline_prediction = np.clip(x_test["lag_1"].to_numpy(dtype=float), 0, None)
    actual = y_test.to_numpy(dtype=float)

    comparison = pd.DataFrame(
        {
            "timestamp": engineered.loc[test_mask, "timestamp"].to_numpy(),
            "actual": actual,
            "baseline_prediction": baseline_prediction,
            "forest_prediction": forest_prediction,
            "route_id": engineered.loc[test_mask, "route_id"].to_numpy(),
        }
    )
    final_model = _make_pipeline(n_estimators, random_state)
    final_model.fit(x, y)
    return {
        "model": final_model,
        "evaluation": {
            "baseline": _regression_metrics(actual, baseline_prediction),
            "random_forest": _regression_metrics(actual, forest_prediction),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "usable_rows": len(x),
            "date_count": len(unique_days),
            "split_date": cutoff.date().isoformat(),
            "target": "passenger_count",
            "feature_columns": FEATURE_COLUMNS.copy(),
            "temporal_split": True,
            "historical_feature_policy": (
                "Lag features are shifted within each route/stop/hour series; "
                "the current passenger count is never used as a feature."
            ),
        },
        "comparison": comparison,
        "model_name": "Random Forest Regressor",
    }


def forecast_route(
    frame: pd.DataFrame,
    model,
    route_id: str,
    *,
    periods: int = 14,
) -> pd.DataFrame:
    """Recursively forecast daily route demand using prior observed/predicted values."""
    if periods < 1 or periods > 60:
        raise ValueError("Forecast horizon must be between 1 and 60 days.")
    x, _, engineered = prepare_prediction_dataset(frame)
    if "route_id" not in frame.columns:
        raise ValueError("A route_id field is required for a route-specific forecast.")

    raw_route = frame["route_id"].astype("string").fillna("Unknown route")
    selected = frame.loc[raw_route.astype(str) == str(route_id)].copy()
    if selected.empty:
        raise ValueError(f"No historical observations were found for route {route_id}.")

    route_history = engineered[engineered["route_id"].astype(str) == str(route_id)]
    if route_history.empty:
        raise ValueError(f"Route {route_id} does not have enough lag history.")
    latest_date = pd.to_datetime(route_history["timestamp"]).max().normalize()
    recent = route_history[
        route_history["timestamp"].dt.normalize() >= latest_date - pd.Timedelta(days=6)
    ]
    if recent.empty:
        raise ValueError("No recent history is available for this route.")

    keys = ["route_id", "stop_id", "hour"]
    states: dict[tuple[str, str, int], deque] = {}
    latest_rows = route_history.sort_values("timestamp").groupby(
        keys, observed=True, sort=False
    )
    for key, series_frame in latest_rows:
        values = series_frame["target"].tail(7).astype(float).tolist()
        if values:
            states[(str(key[0]), str(key[1]), int(key[2]))] = deque(
                values, maxlen=7
            )
    if not states:
        raise ValueError("There are no route/stop/time series to forecast.")

    output: list[dict] = []
    for day_offset in range(1, periods + 1):
        future_date = latest_date + timedelta(days=day_offset)
        feature_rows: list[dict] = []
        state_keys = list(states.keys())
        for route, stop, hour in state_keys:
            history = states[(route, stop, hour)]
            feature_rows.append(
                {
                    "hour": hour,
                    "day_of_week": future_date.dayofweek,
                    "month": future_date.month,
                    "day_of_year": future_date.dayofyear,
                    "is_weekend": int(future_date.dayofweek >= 5),
                    "route_id": route,
                    "stop_id": stop,
                    "lag_1": float(history[-1]),
                    "rolling_7": float(np.mean(history)),
                }
            )
        features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        predictions = np.clip(model.predict(features), 0, None)
        for key, prediction in zip(state_keys, predictions):
            states[key].append(float(prediction))
            output.append(
                {
                    "date": future_date.date().isoformat(),
                    "route_id": str(route_id),
                    "stop_id": key[1],
                    "hour": key[2],
                    "predicted_passengers": float(prediction),
                }
            )

    result = pd.DataFrame(output)
    return (
        result.groupby(["date", "route_id"], as_index=False)["predicted_passengers"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )


def save_model(model, path: str | Path | None = None) -> Path:
    """Persist an explicitly requested trained model; no data is saved by default."""
    import joblib

    target = Path(path) if path is not None else MODELS_DIR / "transitiq_random_forest.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target
