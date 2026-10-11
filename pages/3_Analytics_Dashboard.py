from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import (
    capacity_exceedance_summary,
    dataset_summary,
    route_aggregation,
)
from src.ui import (
    format_number,
    get_filtered_dataset,
    has_dataset,
    render_header,
    show_dataset_quality,
)
from src.visualization import (
    missing_values_figure,
    peak_hour_figure,
    route_comparison_figure,
    route_figure,
    stop_figure,
    trend_figure,
    weekday_hour_heatmap,
    weekday_weekend_figure,
)

render_header(
    "Analytics Dashboard",
    "Analyze passenger demand, routes, stops, peak periods, data quality, "
    "and potential vehicle-capacity exceedances.",
)

# Do not calculate analytics before a dataset is uploaded.
if not has_dataset():
    st.info(
        "No dataset uploaded. Open Data Management, upload your CSV or "
        "Parquet file, and select 'Validate and use this dataset' "
        "to activate the dashboard."
    )
    st.stop()

data = get_filtered_dataset()

if data is None or data.empty:
    st.warning(
        "No observations match the current filters. Adjust the sidebar "
        "filters or reset them to view your uploaded data."
    )
    st.stop()

quality = show_dataset_quality(data)
summary = dataset_summary(data)

st.markdown("### Passenger Overview")

metrics = st.columns(4)

metrics[0].metric(
    "Passenger observations",
    format_number(summary["observations"]),
)

metrics[1].metric(
    "Recorded passengers",
    format_number(summary["total_passengers"])
    if summary["total_passengers"] is not None
    else "N/A",
)

metrics[2].metric(
    "Routes",
    format_number(summary["routes"])
    if summary["routes"] is not None
    else "N/A",
)

metrics[3].metric(
    "Stops",
    format_number(summary["stops"])
    if summary["stops"] is not None
    else "N/A",
)

if quality["errors"]:
    st.warning(
        "Some analyses are unavailable until the required dataset fields "
        "are mapped correctly."
    )

st.markdown("### Passenger Demand Trends")

left, right = st.columns(2)

trend = trend_figure(data)

if trend is None:
    left.info(
        "Demand trends require valid service dates and passenger counts."
    )
else:
    left.plotly_chart(trend, use_container_width=True)

route_chart = route_figure(data)

if route_chart is None:
    right.info(
        "Route analysis requires route IDs and passenger counts."
    )
else:
    right.plotly_chart(route_chart, use_container_width=True)

st.markdown("### Route and Stop Analysis")

left, right = st.columns(2)

stop_chart = stop_figure(data)

if stop_chart is None:
    left.info(
        "Stop analysis requires stop IDs and passenger counts."
    )
else:
    left.plotly_chart(stop_chart, use_container_width=True)

route_totals = route_aggregation(data)

if route_totals.empty:
    right.info(
        "Route summary is unavailable for the selected dataset."
    )
else:
    right.markdown("#### Route Summary")
    right.dataframe(
        route_totals,
        hide_index=True,
        use_container_width=True,
    )

if not route_totals.empty:
    highest = route_totals.iloc[0]
    lowest = route_totals.sort_values("total_passengers").iloc[0]

    cols = st.columns(2)

    cols[0].metric(
        "Highest-demand route",
        str(highest["route_id"]),
        f"{format_number(highest['total_passengers'])} passengers",
    )

    cols[1].metric(
        "Lowest-demand route",
        str(lowest["route_id"]),
        f"{format_number(lowest['total_passengers'])} passengers",
    )

st.markdown("### Travel Patterns")

left, right = st.columns(2)

peak_chart = peak_hour_figure(data)

if peak_chart is None:
    left.info(
        "Peak-hour analysis requires valid service times."
    )
else:
    left.plotly_chart(peak_chart, use_container_width=True)

weekday_chart = weekday_weekend_figure(data)

if weekday_chart is None:
    right.info(
        "Weekday/weekend analysis requires valid service dates."
    )
else:
    right.plotly_chart(weekday_chart, use_container_width=True)

heatmap = weekday_hour_heatmap(data)

if heatmap is None:
    st.info(
        "The weekday-hour heatmap requires service dates and service times."
    )
else:
    st.plotly_chart(heatmap, use_container_width=True)

st.markdown("### Route Demand Over Time")

comparison = route_comparison_figure(data)

if comparison is None:
    st.info(
        "Route trends require route IDs, service dates, and passenger counts."
    )
else:
    st.plotly_chart(comparison, use_container_width=True)

st.markdown("### Vehicle-Capacity Indicators")

capacity = capacity_exceedance_summary(data)

if not capacity["available"]:
    st.info(
        "Capacity indicators require valid passenger_count and "
        "vehicle_capacity values."
    )
else:
    cols = st.columns(3)

    cols[0].metric(
        "Comparable observations",
        f"{capacity['eligible_observations']:,}",
    )

    cols[1].metric(
        "Potential capacity exceedances",
        f"{capacity['exceedances']:,}",
    )

    cols[2].metric(
        "Exceedance share",
        f"{capacity['rate_percent']:.2f}%",
    )

    st.caption(
        "These are potential indicators only. A recorded passenger count "
        "above vehicle capacity does not independently prove actual overcrowding."
    )

    passengers = pd.to_numeric(
        data["passenger_count"],
        errors="coerce",
    )

    vehicle_capacity = pd.to_numeric(
        data["vehicle_capacity"],
        errors="coerce",
    )

    flagged = data.loc[
        passengers.notna()
        & vehicle_capacity.notna()
        & (vehicle_capacity > 0)
        & (passengers > vehicle_capacity)
    ].copy()

    if not flagged.empty:
        st.dataframe(
            flagged.head(200),
            hide_index=True,
            use_container_width=True,
        )

        if len(flagged) > 200:
            st.caption(
                f"Showing 200 of {len(flagged):,} flagged observations."
            )

st.markdown("### Data Quality")

missing_chart = missing_values_figure(data)

if missing_chart is None:
    st.success(
        "No missing values were found in the selected observations."
    )
else:
    st.plotly_chart(missing_chart, use_container_width=True)

latitude_field = next(
    (
        name
        for name in ("latitude", "lat")
        if name in data.columns
    ),
    None,
)

longitude_field = next(
    (
        name
        for name in ("longitude", "lon", "lng")
        if name in data.columns
    ),
    None,
)

if (
    latitude_field
    and longitude_field
    and "stop_id" in data.columns
):
    coordinates = data.copy()

    coordinates[latitude_field] = pd.to_numeric(
        coordinates[latitude_field],
        errors="coerce",
    )

    coordinates[longitude_field] = pd.to_numeric(
        coordinates[longitude_field],
        errors="coerce",
    )

    if "passenger_count" in coordinates.columns:
        coordinates["passenger_count"] = pd.to_numeric(
            coordinates["passenger_count"],
            errors="coerce",
        )
    else:
        coordinates["passenger_count"] = float("nan")

    coordinates = coordinates.dropna(
        subset=[latitude_field, longitude_field]
    )

    if not coordinates.empty:
        st.markdown("### Stop Locations")

        map_frame = (
            coordinates.groupby(
                ["stop_id", latitude_field, longitude_field],
                dropna=False,
            )["passenger_count"]
            .mean()
            .reset_index(name="average_passengers")
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

st.caption(
    "All statistics and charts use the uploaded dataset and active sidebar "
    "filters. No sample statistics are inserted."
)