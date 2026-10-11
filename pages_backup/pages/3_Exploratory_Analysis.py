from __future__ import annotations

import streamlit as st

from src.analytics import dataset_summary
from src.ui import (
    format_number,
    get_filtered_dataset,
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

data = get_filtered_dataset()
render_header(
    "Exploratory Analysis",
    "Explore time patterns, route and stop demand, peak periods, weekdays, and "
    "data completeness using the active dataset filters.",
)

quality = show_dataset_quality(data)
summary = dataset_summary(data)
metrics = st.columns(3)
metrics[0].metric("Passenger observations", format_number(summary["observations"]))
metrics[1].metric("Recorded passengers", format_number(summary["total_passengers"]))
metrics[2].metric(
    "Average passengers / observation",
    format_number(summary["average_passengers_per_observation"], 1),
)

if quality["errors"]:
    st.warning("Passenger-count analysis is unavailable until required fields are mapped.")
else:
    row1 = st.columns(2)
    figures = [
        (row1[0], trend_figure(data), "Demand over time needs usable service dates."),
        (row1[1], route_figure(data), "Route comparison needs route IDs."),
    ]
    for column, figure, empty_message in figures:
        if figure is None:
            column.info(empty_message)
        else:
            column.plotly_chart(figure, use_container_width=True)

    row2 = st.columns(2)
    figures = [
        (row2[0], stop_figure(data), "Stop comparison needs stop IDs."),
        (row2[1], peak_hour_figure(data), "Peak-hour analysis needs service times."),
    ]
    for column, figure, empty_message in figures:
        if figure is None:
            column.info(empty_message)
        else:
            column.plotly_chart(figure, use_container_width=True)

    row3 = st.columns(2)
    figures = [
        (
            row3[0],
            weekday_weekend_figure(data),
            "Weekday/weekend comparison needs service dates.",
        ),
        (
            row3[1],
            weekday_hour_heatmap(data),
            "Heatmap needs both service dates and service times.",
        ),
    ]
    for column, figure, empty_message in figures:
        if figure is None:
            column.info(empty_message)
        else:
            column.plotly_chart(figure, use_container_width=True)

    st.markdown("#### Route demand comparison over time")
    comparison = route_comparison_figure(data)
    if comparison is None:
        st.info("Route comparison over time needs route IDs and service dates.")
    else:
        st.plotly_chart(comparison, use_container_width=True)

st.markdown("#### Missing-value distribution")
missing_fig = missing_values_figure(data)
if missing_fig is None:
    st.success("No missing values were found in the selected observations.")
else:
    st.plotly_chart(missing_fig, use_container_width=True)

st.caption(
    "Charts use aggregated values from the active dataset. No sample or placeholder "
    "statistics are inserted."
)
