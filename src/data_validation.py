from __future__ import annotations

import pandas as pd

from config import NUMERIC_FIELDS
from src.analytics import parse_service_hour


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.isna().sum()
    result = pd.DataFrame(
        {
            "column": counts.index.astype(str),
            "missing_count": counts.to_numpy(dtype=int),
        }
    )
    result["missing_percent"] = (
        result["missing_count"].div(max(len(frame), 1)).mul(100).round(2)
    )
    return result.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)


def duplicate_record_count(frame: pd.DataFrame) -> int:
    if "record_id" in frame.columns:
        return int(frame.duplicated(subset=["record_id"]).sum())
    return int(frame.duplicated().sum())


def validate_transport_data(frame: pd.DataFrame) -> dict:
    """Inspect a dataset and report issues without modifying any input values."""
    errors: list[str] = []
    warnings: list[str] = []
    invalid_counts: dict[str, int] = {}
    if not isinstance(frame, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": ["Input must be a Pandas DataFrame."],
            "warnings": [],
            "invalid_counts": {},
            "missing_values": pd.DataFrame(),
            "duplicate_count": 0,
            "row_count": 0,
        }

    if frame.empty:
        errors.append("The dataset has no rows.")
    if "passenger_count" not in frame.columns:
        errors.append("Map a column to passenger_count before analyzing demand.")

    missing = missing_value_summary(frame)
    duplicate_count = duplicate_record_count(frame)
    if duplicate_count:
        warnings.append(f"{duplicate_count:,} duplicate record(s) were detected.")
    if "service_date" not in frame.columns and "service_datetime" not in frame.columns:
        warnings.append(
            "No service date was mapped. Time trends and chronological forecasting "
            "will be unavailable."
        )
    if "route_id" not in frame.columns:
        warnings.append("No route ID was mapped. Route comparisons will be unavailable.")

    for column in NUMERIC_FIELDS.intersection(frame.columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        present = frame[column].notna()
        invalid_numeric = present & values.isna()
        invalid_counts[f"{column}: not numeric"] = int(invalid_numeric.sum())
        if column == "passenger_count":
            invalid_counts["passenger_count: negative"] = int((values < 0).sum())
            invalid_counts["passenger_count: fractional"] = int(
                ((values.notna()) & (values % 1 != 0)).sum()
            )
        elif column == "vehicle_capacity":
            invalid_counts["vehicle_capacity: non-positive"] = int((values <= 0).sum())
        else:
            invalid_counts[f"{column}: negative"] = int((values < 0).sum())

    for column in ("service_date", "service_datetime"):
        if column in frame.columns:
            values = pd.to_datetime(frame[column], errors="coerce")
            invalid_counts[f"{column}: unparseable"] = int(
                (frame[column].notna() & values.isna()).sum()
            )
            break
    if "service_time" in frame.columns:
        hours = parse_service_hour(frame["service_time"])
        invalid_counts["service_time: unparseable"] = int(
            (frame["service_time"].notna() & hours.isna()).sum()
        )

    invalid_counts = {key: value for key, value in invalid_counts.items() if value}
    if invalid_counts:
        warnings.append(
            f"{sum(invalid_counts.values()):,} invalid value(s) were detected. "
            "They have not been changed."
        )

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "invalid_counts": invalid_counts,
        "missing_values": missing,
        "duplicate_count": duplicate_count,
        "row_count": len(frame),
    }
