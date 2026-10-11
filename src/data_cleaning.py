from __future__ import annotations

import pandas as pd

from src.data_validation import duplicate_record_count


def invalid_transport_rows(frame: pd.DataFrame) -> pd.Series:
    invalid = pd.Series(False, index=frame.index)
    if "passenger_count" in frame.columns:
        passengers = pd.to_numeric(frame["passenger_count"], errors="coerce")
        present = frame["passenger_count"].notna()
        invalid |= (present & passengers.isna()) | (passengers < 0)
    for column in ("vehicle_capacity", "journey_duration", "distance_km"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            present = frame[column].notna()
            invalid |= present & values.isna()
            if column == "vehicle_capacity":
                invalid |= values.notna() & (values <= 0)
            else:
                invalid |= values.notna() & (values < 0)
    for column in ("service_date", "service_datetime"):
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            invalid |= frame[column].notna() & parsed.isna()
            break
    if "service_time" in frame.columns:
        from src.analytics import parse_service_hour

        hours = parse_service_hour(frame["service_time"])
        invalid |= frame["service_time"].notna() & hours.isna()
    return invalid


def clean_transport_data(
    frame: pd.DataFrame,
    *,
    remove_duplicates: bool = True,
    remove_invalid: bool = True,
    missing_strategy: str = "keep",
) -> tuple[pd.DataFrame, dict]:
    """Clean a copy of a transport dataset and return an auditable change summary."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if missing_strategy not in {"keep", "drop_rows", "median_numeric"}:
        raise ValueError(
            "missing_strategy must be keep, drop_rows, or median_numeric."
        )

    result = frame.copy()
    invalid_mask = invalid_transport_rows(result)
    summary = {
        "input_rows": len(result),
        "duplicates_detected": duplicate_record_count(result),
        "duplicates_removed": 0,
        "invalid_rows_detected": int(invalid_mask.sum()),
        "invalid_rows_removed": 0,
        "missing_values_before": int(result.isna().sum().sum()),
        "rows_removed_for_missing_values": 0,
        "numeric_values_filled": 0,
        "output_rows": 0,
        "missing_strategy": missing_strategy,
    }

    if "passenger_count" in result.columns:
        result["passenger_count"] = pd.to_numeric(
            result["passenger_count"], errors="coerce"
        )
    for column in ("vehicle_capacity", "journey_duration", "distance_km"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if remove_invalid:
        current_invalid = invalid_mask.reindex(result.index, fill_value=False)
        summary["invalid_rows_removed"] = int(current_invalid.sum())
        result = result.loc[~current_invalid].copy()

    if remove_duplicates:
        before = len(result)
        subset = ["record_id"] if "record_id" in result.columns else None
        result = result.drop_duplicates(subset=subset, keep="first")
        summary["duplicates_removed"] = int(before - len(result))

    if missing_strategy == "drop_rows":
        before = len(result)
        result = result.dropna()
        summary["rows_removed_for_missing_values"] = int(before - len(result))
    elif missing_strategy == "median_numeric":
        for column in result.select_dtypes(include="number").columns:
            median = result[column].median()
            if pd.notna(median):
                missing_count = int(result[column].isna().sum())
                result[column] = result[column].fillna(median)
                summary["numeric_values_filled"] += missing_count

    summary["output_rows"] = len(result)
    summary["missing_values_after"] = int(result.isna().sum().sum())
    return result.reset_index(drop=True), summary
