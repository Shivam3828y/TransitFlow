from __future__ import annotations

import pandas as pd

from src.analytics import parse_service_hour

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "route_id",
    "stop_id",
    "lag_1",
    "rolling_7",
]
NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "rolling_7",
]
CATEGORICAL_FEATURES = ["route_id", "stop_id"]


def build_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lag features from records ordered within route/stop/hour."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if "passenger_count" not in frame.columns:
        raise ValueError("Prediction requires a passenger_count column.")

    data = frame.copy()
    if "service_datetime" in data.columns:
        timestamp = pd.to_datetime(data["service_datetime"], errors="coerce")
    elif "service_date" in data.columns:
        dates = pd.to_datetime(data["service_date"], errors="coerce")
        if "service_time" in data.columns:
            times = pd.to_datetime(
                data["service_time"].astype(str), format="mixed", errors="coerce"
            )
            date_text = dates.dt.strftime("%Y-%m-%d")
            hours = parse_service_hour(data["service_time"])
            minutes = times.dt.minute.fillna(0)
            time_text = (
                hours.fillna(-1).astype(int).astype(str).str.zfill(2)
                + ":"
                + minutes.astype(int).astype(str).str.zfill(2)
                + ":00"
            )
            timestamp = pd.to_datetime(date_text + " " + time_text, errors="coerce")
            timestamp = timestamp.mask(hours.isna())
        else:
            timestamp = dates
    else:
        raise ValueError(
            "Prediction requires service_date or service_datetime. "
            "Map a date column in Data Management."
        )

    data["timestamp"] = timestamp
    data["target"] = pd.to_numeric(data["passenger_count"], errors="coerce")
    data = data.dropna(subset=["timestamp", "target"])
    data = data[data["target"] >= 0].copy()
    if data.empty:
        raise ValueError("No valid dated, non-negative passenger observations remain.")

    data["route_id"] = (
        data["route_id"].astype("string").fillna("Unknown route")
        if "route_id" in data.columns
        else "All routes"
    )
    data["stop_id"] = (
        data["stop_id"].astype("string").fillna("Unknown stop")
        if "stop_id" in data.columns
        else "All stops"
    )
    data["hour"] = data["timestamp"].dt.hour.astype(int)

    # Collapse multiple records for the same route/stop/hour/day before lagging.
    series = ["route_id", "stop_id", "hour"]
    data["service_day"] = data["timestamp"].dt.normalize()
    daily = (
        data.groupby(series + ["service_day"], as_index=False, observed=True)["target"]
        .sum()
        .rename(columns={"service_day": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    grouped = daily.groupby(series, observed=True, sort=False)["target"]
    daily["lag_1"] = grouped.shift(1)
    daily["rolling_7"] = grouped.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    daily["day_of_week"] = daily["timestamp"].dt.dayofweek
    daily["month"] = daily["timestamp"].dt.month
    daily["day_of_year"] = daily["timestamp"].dt.dayofyear
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["route_id"] = daily["route_id"].astype(str)
    daily["stop_id"] = daily["stop_id"].astype(str)
    return daily.dropna(subset=["lag_1", "rolling_7"]).reset_index(drop=True)


def prepare_prediction_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    engineered = build_prediction_frame(frame)
    if engineered.empty:
        raise ValueError(
            "Not enough consecutive history to calculate lag features. "
            "Each route/stop/hour series needs at least four observations."
        )
    x = engineered[FEATURE_COLUMNS].copy()
    y = engineered["target"].astype(float).copy()
    return x, y, engineered
