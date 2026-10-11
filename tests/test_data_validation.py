from __future__ import annotations

import pandas as pd

from src.data_loader import apply_column_mapping, suggest_column_mapping
from src.data_validation import (
    duplicate_record_count,
    missing_value_summary,
    validate_transport_data,
)


def test_validation_reports_missing_passenger_count() -> None:
    report = validate_transport_data(pd.DataFrame({"route_id": ["R1"]}))
    assert not report["is_valid"]
    assert any("passenger_count" in message for message in report["errors"])


def test_validation_reports_missing_and_invalid_values_without_changing_data() -> None:
    frame = pd.DataFrame(
        {
            "service_date": ["2025-01-01", "not-a-date"],
            "passenger_count": [12, -2],
            "route_id": ["R1", "R1"],
            "stop_id": ["S1", None],
            "service_time": ["08:00:00", "invalid"],
            "record_id": ["a", "b"],
        }
    )
    original = frame.copy(deep=True)
    report = validate_transport_data(frame)

    assert report["is_valid"]
    assert report["missing_values"].set_index("column").loc["stop_id", "missing_count"] == 1
    assert report["invalid_counts"]["passenger_count: negative"] == 1
    assert report["invalid_counts"]["service_date: unparseable"] == 1
    assert report["invalid_counts"]["service_time: unparseable"] == 1
    pd.testing.assert_frame_equal(frame, original)


def test_duplicate_detection_prefers_record_id() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "a", "c"],
            "passenger_count": [1, 99, 3],
        }
    )
    assert duplicate_record_count(frame) == 1


def test_mapping_suggestions_and_invalid_mapping() -> None:
    columns = ["travel_date", "route", "ridership"]
    mapping = suggest_column_mapping(columns)
    assert mapping["service_date"] == "travel_date"
    assert mapping["route_id"] == "route"
    assert mapping["passenger_count"] == "ridership"

    frame = pd.DataFrame({"route": ["R1"], "ridership": [8]})
    mapped = apply_column_mapping(
        frame, {"route_id": "route", "passenger_count": "ridership"}
    )
    assert "route_id" in mapped and "passenger_count" in mapped
    try:
        apply_column_mapping(frame, {"route_id": "missing"})
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Invalid source columns should be rejected.")


def test_missing_summary_has_percentages() -> None:
    result = missing_value_summary(pd.DataFrame({"a": [1, None], "b": [2, 3]}))
    row = result.set_index("column").loc["a"]
    assert row["missing_count"] == 1
    assert row["missing_percent"] == 50
