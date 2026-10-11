from __future__ import annotations

import pandas as pd
import pytest

from src.data_cleaning import clean_transport_data


def test_removes_duplicate_records_and_invalid_passenger_counts() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "a", "b", "c"],
            "service_date": ["2025-01-01"] * 4,
            "passenger_count": [10, 10, -1, "not-a-number"],
        }
    )
    cleaned, summary = clean_transport_data(frame)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "passenger_count"] == 10
    assert summary["duplicates_detected"] == 1
    assert summary["duplicates_removed"] == 1
    assert summary["invalid_rows_detected"] == 2
    assert summary["invalid_rows_removed"] == 2


def test_missing_value_strategies_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "passenger_count": [10, None, 30],
            "vehicle_capacity": [40, 40, 40],
            "route_id": ["R1", None, "R1"],
        }
    )
    kept, keep_summary = clean_transport_data(
        frame, remove_duplicates=False, remove_invalid=False, missing_strategy="keep"
    )
    assert len(kept) == 3
    assert keep_summary["missing_values_after"] == 2

    dropped, drop_summary = clean_transport_data(
        frame, remove_duplicates=False, remove_invalid=False, missing_strategy="drop_rows"
    )
    assert len(dropped) == 2
    assert drop_summary["rows_removed_for_missing_values"] == 1

    filled, fill_summary = clean_transport_data(
        frame,
        remove_duplicates=False,
        remove_invalid=False,
        missing_strategy="median_numeric",
    )
    assert filled["passenger_count"].tolist() == [10, 20, 30]
    assert fill_summary["numeric_values_filled"] == 1
    assert filled["route_id"].isna().sum() == 1


def test_rejects_unknown_missing_strategy() -> None:
    with pytest.raises(ValueError, match="missing_strategy"):
        clean_transport_data(pd.DataFrame({"passenger_count": [1]}), missing_strategy="guess")


def test_cleaning_does_not_mutate_input() -> None:
    frame = pd.DataFrame({"passenger_count": ["4", "-1"]})
    original = frame.copy(deep=True)
    clean_transport_data(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_missing_passenger_counts_follow_the_selected_missing_strategy() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "passenger_count": [10, None, 30],
        }
    )
    kept, summary = clean_transport_data(
        frame,
        remove_duplicates=False,
        remove_invalid=True,
        missing_strategy="keep",
    )
    assert len(kept) == 3
    assert summary["invalid_rows_removed"] == 0
    filled, _ = clean_transport_data(
        frame,
        remove_duplicates=False,
        remove_invalid=True,
        missing_strategy="median_numeric",
    )
    assert filled["passenger_count"].tolist() == [10, 20, 30]
