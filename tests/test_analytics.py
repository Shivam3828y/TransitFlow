from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.analytics import (
    capacity_exceedance_summary,
    dataset_summary,
    filter_transport_data,
    peak_hour_analysis,
    route_aggregation,
)
from src.feature_engineering import prepare_prediction_dataset
from src.prediction_model import train_and_evaluate
from src.sample_data import generate_sample_data


def test_route_aggregation_and_summary() -> None:
    frame = pd.DataFrame(
        {
            "route_id": ["R1", "R1", "R2"],
            "stop_id": ["S1", "S2", "S3"],
            "passenger_count": [10, 20, 50],
        }
    )
    aggregated = route_aggregation(frame)
    assert aggregated.iloc[0]["route_id"] == "R2"
    assert aggregated.iloc[0]["total_passengers"] == 50
    assert dataset_summary(frame)["highest_demand_route"]["route_id"] == "R2"
    assert dataset_summary(frame)["total_passengers"] == 80


def test_date_and_route_filters() -> None:
    frame = pd.DataFrame(
        {
            "service_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "route_id": ["R1", "R1", "R2"],
            "passenger_count": [4, 5, 6],
        }
    )
    result = filter_transport_data(
        frame,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        routes=["R1"],
    )
    assert result["passenger_count"].tolist() == [5]


def test_peak_hour_and_capacity_indicators() -> None:
    frame = pd.DataFrame(
        {
            "service_time": ["08:00:00", "17:00:00", "17:30:00"],
            "passenger_count": [25, 50, 45],
            "vehicle_capacity": [40, 40, None],
        }
    )
    peak = peak_hour_analysis(frame)
    assert peak.sort_values("total_passengers").iloc[-1]["hour"] == 17
    indicator = capacity_exceedance_summary(frame)
    assert indicator["eligible_observations"] == 2
    assert indicator["exceedances"] == 1
    numeric_hour_peak = peak_hour_analysis(
        pd.DataFrame({"service_time": [8, 17], "passenger_count": [12, 40]})
    )
    assert numeric_hour_peak["hour"].tolist() == [8, 17]


def test_lag_features_exclude_current_target() -> None:
    frame = pd.DataFrame(
        {
            "service_date": [
                (date(2025, 1, 1) + timedelta(days=index)).isoformat()
                for index in range(10)
            ],
            "service_time": ["08:00:00"] * 10,
            "route_id": ["R1"] * 10,
            "stop_id": ["S1"] * 10,
            "passenger_count": list(range(1, 11)),
        }
    )
    features, target, engineered = prepare_prediction_dataset(frame)
    first = engineered.iloc[0]
    assert first["target"] == 4
    assert first["lag_1"] == 3
    assert first["rolling_7"] == 2
    assert "target" not in features.columns
    assert len(features) == len(target)


def test_prediction_rejects_missing_dates() -> None:
    frame = pd.DataFrame({"passenger_count": [1, 2, 3]})
    with pytest.raises(ValueError, match="service_date"):
        prepare_prediction_dataset(frame)


def test_real_model_evaluates_on_chronological_holdout() -> None:
    frame = generate_sample_data(days=22)
    result = train_and_evaluate(frame, n_estimators=10)
    evaluation = result["evaluation"]
    assert evaluation["temporal_split"] is True
    assert evaluation["train_rows"] > evaluation["test_rows"] > 0
    assert evaluation["split_date"] > "2025-01-01"
    assert evaluation["baseline"]["mae"] >= 0
    assert evaluation["random_forest"]["rmse"] >= 0
    assert {"lag_1", "rolling_7"}.issubset(evaluation["feature_columns"])
