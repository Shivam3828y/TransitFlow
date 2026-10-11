from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    capacity_exceedance_summary,
    peak_hour_analysis,
    route_aggregation,
    stop_aggregation,
)
from src.ui import format_number, get_filtered_dataset, render_header
from src.visualization import (
    peak_hour_figure,
    route_comparison_figure,
    route_figure,
    stop_figure,
)

data = get_filtered_dataset()
render_header(
    "Route Analytics",
    "Compare route and stop passenger counts over time and review potential "
    "vehicle-capacity exceedance indicators.",
)

route_totals = route_aggregation(data)
stop_totals = stop_aggregation(data)
if route_totals.empty:
    st.warning("Route analytics need mapped route_id and passenger_count fields.")
else:
    totals = st.columns(3)
    totals[0].metric("Routes in selection", f"{route_totals['route_id'].nunique():,}")
    highest = route_totals.iloc[0]
    lowest = route_totals.sort_values("total_passengers").iloc[0]
    totals[1].metric(
        "Highest recorded demand",
        str(highest["route_id"]),
        f"{format_number(highest['total_passengers'])} passengers",
    )
    totals[2].metric(
        "Lowest recorded demand",
        str(lowest["route_id"]),
        f"{format_number(lowest['total_passengers'])} passengers",
    )

    left, right = st.columns(2)
    left.plotly_chart(route_figure(data), use_container_width=True)
    if stop_totals.empty:
        right.info("Stop demand is unavailable because stop_id is not mapped.")
    else:
        right.plotly_chart(stop_figure(data), use_container_width=True)

    st.markdown("#### Demand trends by route")
    route_trend = route_comparison_figure(data)
    if route_trend is None:
        st.info("Route trends need service dates and multiple route observations.")
    else:
        st.plotly_chart(route_trend, use_container_width=True)

    st.markdown("#### Route summary")
    st.dataframe(route_totals, hide_index=True, use_container_width=True)

hourly = peak_hour_analysis(data)
st.markdown("#### Peak demand periods")
hour_chart = peak_hour_figure(data)
if hour_chart is None:
    st.info("Peak period analysis needs a mapped service_time field.")
else:
    st.plotly_chart(hour_chart, use_container_width=True)
    peak = hourly.sort_values("total_passengers", ascending=False).head(3)
    st.dataframe(peak, hide_index=True, use_container_width=True)

capacity = capacity_exceedance_summary(data)
st.markdown("#### Potential capacity-exceedance indicators")
if not capacity["available"]:
    st.info(
        "This indicator needs valid passenger_count and vehicle_capacity values."
    )
else:
    cols = st.columns(3)
    cols[0].metric("Comparable observations", f"{capacity['eligible_observations']:,}")
    cols[1].metric(
        "Potential capacity exceedances",
        f"{capacity['exceedances']:,}",
    )
    cols[2].metric("Share of comparable observations", f"{capacity['rate_percent']:.2f}%")
    st.warning(
        "These are potential capacity-exceedance indicators only. A recorded count "
        "above vehicle_capacity does not prove actual overcrowding."
    )
    passengers = pd.to_numeric(data["passenger_count"], errors="coerce")
    vehicle_capacity = pd.to_numeric(data["vehicle_capacity"], errors="coerce")
    flagged = data.loc[
        passengers.notna()
        & vehicle_capacity.notna()
        & (vehicle_capacity > 0)
        & (passengers > vehicle_capacity)
    ].copy()
    if not flagged.empty:
        st.dataframe(flagged.head(200), hide_index=True, use_container_width=True)
        if len(flagged) > 200:
            st.caption(f"Showing 200 of {len(flagged):,} flagged observations.")

latitude_field = next((name for name in ("latitude", "lat") if name in data.columns), None)
longitude_field = next((name for name in ("longitude", "lon", "lng") if name in data.columns), None)
if latitude_field and longitude_field:
    coordinates = data.copy()
    coordinates[latitude_field] = pd.to_numeric(
        coordinates[latitude_field], errors="coerce"
    )
    coordinates[longitude_field] = pd.to_numeric(
        coordinates[longitude_field], errors="coerce"
    )
    coordinates = coordinates.dropna(subset=[latitude_field, longitude_field])
    if not coordinates.empty:
        st.markdown("#### Stops with uploaded coordinates")
        st.caption("Only coordinates supplied in the selected dataset are plotted.")
        map_frame = (
            coordinates.groupby(
                ["stop_id", latitude_field, longitude_field], dropna=False
            )["passenger_count"]
            .agg(
                average_passengers=lambda values: pd.to_numeric(
                    values, errors="coerce"
                ).mean()
            )
            .reset_index()
        )
        map_fig = px.scatter_mapbox(
            map_frame,
            lat=latitude_field,
            lon=longitude_field,
            color="average_passengers",
            hover_name="stop_id",
            mapbox_style="open-street-map",
            title="Recorded stop locations and average passengers",
        )
        st.plotly_chart(map_fig, use_container_width=True)
