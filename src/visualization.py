from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics import (
    demand_over_time,
    parse_service_hour,
    peak_hour_analysis,
    route_aggregation,
    route_demand_changes,
    stop_aggregation,
    weekday_weekend_analysis,
)
from src.data_validation import missing_value_summary


def trend_figure(frame: pd.DataFrame):
    data = demand_over_time(frame)
    if data.empty:
        return None
    fig = px.line(
        data,
        x="date",
        y="passengers",
        markers=True,
        title="Recorded passenger demand over time",
        labels={"date": "Service date", "passengers": "Passengers"},
    )
    fig.update_layout(hovermode="x unified")
    return fig


def route_figure(frame: pd.DataFrame, *, top_n: int = 20):
    data = route_aggregation(frame).head(top_n)
    if data.empty:
        return None
    fig = px.bar(
        data.sort_values("total_passengers"),
        x="total_passengers",
        y="route_id",
        orientation="h",
        title="Recorded passenger demand by route",
        labels={"route_id": "Route", "total_passengers": "Passengers"},
        hover_data={"avg_passengers": ":.1f"},
    )
    return fig


def stop_figure(frame: pd.DataFrame, *, top_n: int = 20):
    data = stop_aggregation(frame).head(top_n)
    if data.empty:
        return None
    return px.bar(
        data.sort_values("total_passengers"),
        x="total_passengers",
        y="stop_id",
        orientation="h",
        title="Recorded passenger demand by stop",
        labels={"stop_id": "Stop", "total_passengers": "Passengers"},
    )


def peak_hour_figure(frame: pd.DataFrame):
    data = peak_hour_analysis(frame)
    if data.empty:
        return None
    return px.bar(
        data,
        x="hour",
        y="total_passengers",
        title="Demand by service hour",
        labels={"hour": "Hour of day", "total_passengers": "Passengers"},
        hover_data={"avg_passengers": ":.1f"},
    )


def weekday_weekend_figure(frame: pd.DataFrame):
    data = weekday_weekend_analysis(frame)
    if data.empty:
        return None
    return px.bar(
        data,
        x="day_type",
        y="avg_passengers",
        title="Average passengers per observation: weekdays vs weekends",
        labels={"day_type": "Day type", "avg_passengers": "Average passengers"},
    )


def weekday_hour_heatmap(frame: pd.DataFrame):
    if not {"service_time", "passenger_count"}.issubset(frame.columns):
        return None
    date_column = (
        "service_datetime"
        if "service_datetime" in frame.columns
        else "service_date"
        if "service_date" in frame.columns
        else None
    )
    if not date_column:
        return None
    data = frame[[date_column, "service_time", "passenger_count"]].copy()
    dates = pd.to_datetime(data[date_column], errors="coerce")
    hours = parse_service_hour(data["service_time"])
    data["weekday"] = dates.dt.day_name()
    data["hour"] = hours
    data["passengers"] = pd.to_numeric(data["passenger_count"], errors="coerce")
    data = data.dropna(subset=["weekday", "hour", "passengers"])
    if data.empty:
        return None
    pivot = data.pivot_table(
        index="weekday", columns="hour", values="passengers", aggfunc="mean"
    )
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([day for day in order if day in pivot.index])
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[int(hour) for hour in pivot.columns],
            y=pivot.index.tolist(),
            colorbar={"title": "Avg. passengers"},
            hovertemplate="%{y}, %{x}:00<br>%{z:.1f} passengers<extra></extra>",
        )
    )
    fig.update_layout(
        title="Average demand by weekday and service hour",
        xaxis_title="Hour of day",
        yaxis_title="Weekday",
    )
    return fig


def missing_values_figure(frame: pd.DataFrame):
    data = missing_value_summary(frame)
    data = data[data["missing_count"] > 0]
    if data.empty:
        return None
    return px.bar(
        data.sort_values("missing_count"),
        x="missing_count",
        y="column",
        orientation="h",
        title="Missing values by column",
        labels={"column": "Column", "missing_count": "Missing values"},
        hover_data={"missing_percent": ":.2f"},
    )


def route_comparison_figure(frame: pd.DataFrame, *, top_n: int = 8):
    data = route_demand_changes(frame)
    if data.empty:
        return None
    totals = data.groupby("route_id")["passengers"].sum().nlargest(top_n)
    data = data[data["route_id"].isin(totals.index)]
    return px.line(
        data,
        x="date",
        y="passengers",
        color="route_id",
        title="Daily passenger demand: route comparison",
        labels={"date": "Service date", "passengers": "Passengers", "route_id": "Route"},
    )


def prediction_comparison_figure(data: pd.DataFrame):
    if data.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["actual"],
            mode="lines+markers",
            name="Actual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["baseline_prediction"],
            mode="lines",
            name="Lag-1 baseline",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["forest_prediction"],
            mode="lines",
            name="Random Forest",
        )
    )
    fig.update_layout(
        title="Actual vs predicted passenger counts (chronological holdout)",
        xaxis_title="Service date",
        yaxis_title="Passengers per route/stop/hour/day",
        hovermode="x unified",
    )
    return fig


def forecast_figure(data: pd.DataFrame):
    if data.empty:
        return None
    return px.line(
        data,
        x="date",
        y="predicted_passengers",
        markers=True,
        title=f"Forecast daily demand for route {data['route_id'].iloc[0]}",
        labels={"date": "Forecast date", "predicted_passengers": "Predicted passengers"},
    )
