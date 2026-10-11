from __future__ import annotations

import pandas as pd
import streamlit as st

from config import APP_TITLE
from src.analytics import filter_transport_data
from src.ui import ensure_dataset, get_full_dataset

st.set_page_config(
    page_title=f"{APP_TITLE} | Transport Analytics",
    page_icon=":material/directions_bus:",
    layout="wide",
)

ensure_dataset()
full_data = get_full_dataset()
source_name = st.session_state.get("dataset_name", "Dataset")

pages = [
    st.Page("src/home.py", title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(
        "pages/1_Data_Management.py",
        title="Data Management",
        icon=":material/database:",
    ),
    st.Page(
        "pages/2_Big_Data_Processing.py",
        title="Big Data Processing",
        icon=":material/schema:",
    ),
    st.Page(
        "pages/3_Exploratory_Analysis.py",
        title="Exploratory Analysis",
        icon=":material/monitoring:",
    ),
    st.Page(
        "pages/4_Demand_Prediction.py",
        title="Demand Prediction",
        icon=":material/query_stats:",
    ),
    st.Page(
        "pages/5_Route_Analytics.py",
        title="Route Analytics",
        icon=":material/route:",
    ),
    st.Page("pages/6_Reports.py", title="Reports", icon=":material/description:"),
]

page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown("### TransitIQ")
    st.caption("Public transport analytics")
    st.divider()
    st.markdown("#### Dataset filters")
    date_column = (
        "service_datetime"
        if "service_datetime" in full_data.columns
        else "service_date"
        if "service_date" in full_data.columns
        else None
    )
    start_date = end_date = None
    current_version = st.session_state.get("dataset_version", 1)
    if date_column:
        parsed = pd.to_datetime(full_data[date_column], errors="coerce").dropna()
        if not parsed.empty:
            minimum, maximum = parsed.min().date(), parsed.max().date()
            if st.session_state.get("_filter_dataset_version") != current_version:
                st.session_state["global_date_range"] = (minimum, maximum)
                st.session_state["global_route_filter"] = []
                st.session_state["_filter_dataset_version"] = current_version
            st.session_state["_filter_default_dates"] = (minimum, maximum)
            selected_dates = st.date_input(
                "Service date range",
                value=st.session_state.get("global_date_range", (minimum, maximum)),
                min_value=minimum,
                max_value=maximum,
                key="global_date_range",
            )
            if isinstance(selected_dates, (tuple, list)) and selected_dates:
                start_date = selected_dates[0]
                end_date = selected_dates[-1] if len(selected_dates) > 1 else None
    else:
        st.caption("Date filters are unavailable: no date field is mapped.")
        if st.session_state.get("_filter_dataset_version") != current_version:
            st.session_state["global_route_filter"] = []
            st.session_state.pop("global_date_range", None)
            st.session_state["_filter_dataset_version"] = current_version

    selected_routes: list[str] = []
    if "route_id" in full_data.columns:
        route_values = sorted(
            full_data["route_id"].dropna().astype(str).unique().tolist()
        )
        if len(route_values) > 500:
            st.warning("Showing the first 500 route values in the filter.")
            route_values = route_values[:500]
        selected_routes = st.multiselect(
            "Routes",
            options=route_values,
            key="global_route_filter",
            help="Leave empty to include every route.",
        )
    else:
        st.caption("Route filters are unavailable: no route field is mapped.")

    st.session_state["filter_start_date"] = start_date
    st.session_state["filter_end_date"] = end_date
    st.session_state["selected_routes"] = selected_routes
    filtered = filter_transport_data(
        full_data,
        start_date=start_date,
        end_date=end_date,
        routes=selected_routes,
    )
    st.session_state["filtered_df"] = filtered
    st.caption(
        f"{len(filtered):,} of {len(full_data):,} observations · {source_name}"
    )
    def _reset_filter_widgets() -> None:
        if "_filter_default_dates" in st.session_state:
            st.session_state["global_date_range"] = st.session_state[
                "_filter_default_dates"
            ]
        st.session_state["global_route_filter"] = []

    if st.button(
        "Reset filters",
        use_container_width=True,
        on_click=_reset_filter_widgets,
    ):
        st.rerun()

    st.divider()
    st.caption("Preview tables are limited to keep large uploads responsive.")

page.run()
