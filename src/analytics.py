from __future__ import annotations

import pandas as pd


def filter_transport_data(
    frame: pd.DataFrame,
    *,
    start_date=None,
    end_date=None,
    routes: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    date_column = (
        "service_datetime"
        if "service_datetime" in result.columns
        else "service_date"
        if "service_date" in result.columns
        else None
    )
    if date_column and (start_date is not None or end_date is not None):
        parsed = pd.to_datetime(result[date_column], errors="coerce")
        mask = parsed.notna()
        if start_date is not None:
            mask &= parsed.dt.date >= pd.Timestamp(start_date).date()
        if end_date is not None:
            mask &= parsed.dt.date <= pd.Timestamp(end_date).date()
        result = result.loc[mask]
    if routes and "route_id" in result.columns:
        result = result[result["route_id"].astype(str).isin({str(r) for r in routes})]
    return result.copy()


def dataset_summary(frame: pd.DataFrame) -> dict:
    passengers = (
        pd.to_numeric(frame["passenger_count"], errors="coerce")
        if "passenger_count" in frame.columns
        else pd.Series(dtype="float64")
    )
    routes = (
        int(frame["route_id"].nunique(dropna=True))
        if "route_id" in frame.columns
        else None
    )
    stops = (
        int(frame["stop_id"].nunique(dropna=True))
        if "stop_id" in frame.columns
        else None
    )
    route_totals = route_aggregation(frame)
    highest_route = None
    if not route_totals.empty:
        row = route_totals.iloc[0]
        highest_route = {
            "route_id": str(row["route_id"]),
            "total_passengers": int(row["total_passengers"]),
        }
    return {
        "observations": len(frame),
        "total_passengers": (
            float(passengers.sum()) if passengers.notna().any() else None
        ),
        "routes": routes,
        "stops": stops,
        "average_passengers_per_observation": (
            float(passengers.mean()) if passengers.notna().any() else None
        ),
        "highest_demand_route": highest_route,
    }


def route_aggregation(frame: pd.DataFrame, *, metric: str = "sum") -> pd.DataFrame:
    if "route_id" not in frame.columns or "passenger_count" not in frame.columns:
        return pd.DataFrame(columns=["route_id", "total_passengers", "avg_passengers"])
    data = frame[["route_id", "passenger_count"]].copy()
    data["passenger_count"] = pd.to_numeric(data["passenger_count"], errors="coerce")
    data = data.dropna(subset=["route_id", "passenger_count"])
    grouped = data.groupby("route_id", observed=True)["passenger_count"]
    result = grouped.agg(total_passengers="sum", avg_passengers="mean").reset_index()
    result["total_passengers"] = result["total_passengers"].round(2)
    result["avg_passengers"] = result["avg_passengers"].round(2)
    sort_column = "avg_passengers" if metric == "mean" else "total_passengers"
    return result.sort_values(sort_column, ascending=False).reset_index(drop=True)


def stop_aggregation(frame: pd.DataFrame) -> pd.DataFrame:
    if "stop_id" not in frame.columns or "passenger_count" not in frame.columns:
        return pd.DataFrame(columns=["stop_id", "total_passengers"])
    values = pd.to_numeric(frame["passenger_count"], errors="coerce")
    data = pd.DataFrame({"stop_id": frame["stop_id"], "passengers": values}).dropna()
    return (
        data.groupby("stop_id", observed=True)["passengers"]
        .agg(total_passengers="sum", avg_passengers="mean")
        .reset_index()
        .sort_values("total_passengers", ascending=False)
        .reset_index(drop=True)
    )


def demand_over_time(frame: pd.DataFrame, *, frequency: str = "D") -> pd.DataFrame:
    date_column = (
        "service_datetime"
        if "service_datetime" in frame.columns
        else "service_date"
        if "service_date" in frame.columns
        else None
    )
    if not date_column or "passenger_count" not in frame.columns:
        return pd.DataFrame(columns=["date", "passengers"])
    data = frame[[date_column, "passenger_count"]].copy()
    data["date"] = pd.to_datetime(data[date_column], errors="coerce")
    data["passengers"] = pd.to_numeric(data["passenger_count"], errors="coerce")
    data = data.dropna(subset=["date", "passengers"])
    if data.empty:
        return pd.DataFrame(columns=["date", "passengers"])
    data["date"] = data["date"].dt.to_period(frequency).dt.to_timestamp()
    return (
        data.groupby("date", as_index=False)["passengers"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )


def parse_service_hour(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values.astype(str), format="mixed", errors="coerce")
    result = parsed.dt.hour.astype("float64")
    numeric = pd.to_numeric(values, errors="coerce")
    whole_hours = numeric.notna() & numeric.between(0, 23) & (numeric % 1 == 0)
    result.loc[whole_hours] = numeric.loc[whole_hours]
    result.loc[numeric.notna() & ~whole_hours] = float("nan")
    return result


def _service_hour(values: pd.Series) -> pd.Series:
    return parse_service_hour(values)


def peak_hour_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    if "service_time" not in frame.columns or "passenger_count" not in frame.columns:
        return pd.DataFrame(columns=["hour", "total_passengers", "avg_passengers"])
    hours = _service_hour(frame["service_time"])
    counts = pd.to_numeric(frame["passenger_count"], errors="coerce")
    data = pd.DataFrame({"hour": hours, "passengers": counts}).dropna()
    if data.empty:
        return pd.DataFrame(columns=["hour", "total_passengers", "avg_passengers"])
    result = (
        data.groupby("hour", as_index=False)["passengers"]
        .agg(total_passengers="sum", avg_passengers="mean")
        .sort_values("hour")
    )
    result["hour"] = result["hour"].astype(int)
    return result.reset_index(drop=True)


def weekday_weekend_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    date_column = (
        "service_datetime"
        if "service_datetime" in frame.columns
        else "service_date"
        if "service_date" in frame.columns
        else None
    )
    if not date_column or "passenger_count" not in frame.columns:
        return pd.DataFrame(columns=["day_type", "avg_passengers", "total_passengers"])
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    counts = pd.to_numeric(frame["passenger_count"], errors="coerce")
    data = pd.DataFrame({"weekday": dates.dt.dayofweek, "passengers": counts}).dropna()
    data["day_type"] = data["weekday"].apply(
        lambda value: "Weekend" if int(value) >= 5 else "Weekday"
    )
    return (
        data.groupby("day_type", as_index=False)["passengers"]
        .agg(avg_passengers="mean", total_passengers="sum")
        .reset_index(drop=True)
    )


def capacity_exceedance_summary(frame: pd.DataFrame) -> dict:
    if not {"passenger_count", "vehicle_capacity"}.issubset(frame.columns):
        return {
            "available": False,
            "eligible_observations": 0,
            "exceedances": 0,
            "rate_percent": None,
        }
    passengers = pd.to_numeric(frame["passenger_count"], errors="coerce")
    capacity = pd.to_numeric(frame["vehicle_capacity"], errors="coerce")
    valid = passengers.notna() & capacity.notna() & (capacity > 0)
    eligible = int(valid.sum())
    exceedances = int((passengers[valid] > capacity[valid]).sum())
    return {
        "available": eligible > 0,
        "eligible_observations": eligible,
        "exceedances": exceedances,
        "rate_percent": round(exceedances / eligible * 100, 2) if eligible else None,
    }


def route_demand_changes(frame: pd.DataFrame) -> pd.DataFrame:
    date_column = (
        "service_datetime"
        if "service_datetime" in frame.columns
        else "service_date"
        if "service_date" in frame.columns
        else None
    )
    if not date_column or "route_id" not in frame.columns:
        return pd.DataFrame(columns=["date", "route_id", "passengers"])
    data = frame[[date_column, "route_id", "passenger_count"]].copy()
    data["date"] = pd.to_datetime(data[date_column], errors="coerce").dt.date
    data["passengers"] = pd.to_numeric(data["passenger_count"], errors="coerce")
    data = data.dropna(subset=["date", "route_id", "passengers"])
    return (
        data.groupby(["date", "route_id"], as_index=False, observed=True)["passengers"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
