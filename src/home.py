from __future__ import annotations

import streamlit as st

from src.analytics import dataset_summary, route_aggregation
from src.ui import (
    format_number,
    get_filtered_dataset,
    has_dataset,
    render_header,
    show_dataset_quality,
)
from src.visualization import (
    peak_hour_figure,
    route_figure,
    trend_figure,
    weekday_hour_heatmap,
)

# Stop before calculating analytics if no dataset has been uploaded.
if not has_dataset():
    render_header(
        "TransitIQ",
        "Public transport analytics and route performance.",
    )

    st.info(
        "No dataset uploaded yet. Open Data Management from the sidebar "
        "and upload your CSV or Parquet file to begin."
    )

    st.markdown("### What you can analyze after uploading")
    st.markdown(
        """
        - **Passenger demand:** Analyze passenger observations and trends.
        - **Route performance:** Compare passenger counts across routes.
        - **Peak hours:** Identify periods with higher passenger demand.
        - **Data quality:** Review missing values and data issues.
        - **Route ranking:** Compare routes using your uploaded data.
        """
    )

    st.stop()


data = get_filtered_dataset()

if data is None or data.empty:
    st.warning(
        "No records match the current filters. Open Data Management "
        "to verify your dataset or reset the sidebar filters."
    )
    st.stop()


source_name = st.session_state.get("dataset_name", "Uploaded dataset")

render_header(
    "TransitIQ",
    (
        "Demand patterns, route performance, and data quality · "
        f"{source_name}"
    ),
)

quality = show_dataset_quality(data)
summary = dataset_summary(data)

route = summary["highest_demand_route"]

highest_route = (
    f"{route['route_id']} · "
    f"{format_number(route['total_passengers'])} passengers"
    if route
    else "—"
)

metrics = [
    ("Passenger observations", format_number(summary["observations"])),
    ("Recorded passengers", format_number(summary["total_passengers"])),
    ("Routes", format_number(summary["routes"])),
    ("Stops", format_number(summary["stops"])),
    (
        "Average / observation",
        format_number(summary["average_passengers_per_observation"], 1),
    ),
    ("Highest-demand route", highest_route),
]

columns = st.columns(3)

for index, (label, value) in enumerate(metrics):
    columns[index % 3].metric(label, value)


if quality["errors"]:
    st.warning(
        "Some analyses are unavailable until the required fields "
        "are mapped correctly."
    )

else:
    left, right = st.columns(2)

    trend = trend_figure(data)
    routes = route_figure(data)

    if trend is not None:
        left.plotly_chart(trend, use_container_width=True)
    else:
        left.info(
            "Demand-over-time analysis needs usable service dates."
        )

    if routes is not None:
        right.plotly_chart(routes, use_container_width=True)
    else:
        right.info(
            "Route comparison needs route IDs and passenger counts."
        )

    left, right = st.columns(2)

    peak = peak_hour_figure(data)
    heatmap = weekday_hour_heatmap(data)

    if peak is not None:
        left.plotly_chart(peak, use_container_width=True)
    else:
        left.info(
            "Peak-hour analysis needs service times."
        )

    if heatmap is not None:
        right.plotly_chart(heatmap, use_container_width=True)
    else:
        right.info(
            "The weekday/hour heatmap needs service dates and times."
        )


st.markdown("#### Quick route ranking")

route_table = route_aggregation(data).head(10)

if route_table.empty:
    st.info(
        "Route ranking is unavailable for this dataset."
    )
else:
    st.dataframe(
        route_table,
        hide_index=True,
        use_container_width=True,
    )


st.caption(
    "All figures are calculated from the uploaded dataset "
    "and the active sidebar filters."
)