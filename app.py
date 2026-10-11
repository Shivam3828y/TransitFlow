from __future__ import annotations

import pandas as pd
import streamlit as st

from config import APP_TITLE
from src.analytics import filter_transport_data
from src.ui import ensure_dataset, get_full_dataset, has_dataset


st.set_page_config(
    page_title=f"{APP_TITLE} | Transport Analytics",
    page_icon=":material/directions_bus:",
    layout="wide",
)

# Initialize an empty dataset state.
# No sample data is loaded automatically.
ensure_dataset()


# ---------------------------------------------------------
# PAGE NAVIGATION
# ---------------------------------------------------------

pages = [
    st.Page(
        "pages/1_Data_Management.py",
        title="Data Management",
        icon=":material/database:",
        default=True,
    ),
    st.Page(
        "src/home.py",
        title="Overview",
        icon=":material/space_dashboard:",
    ),
    st.Page(
        "pages/2_Big_Data_Processing.py",
        title="Big Data Processing",
        icon=":material/schema:",
    ),
    st.Page(
        "pages/3_Analytics_Dashboard.py",
        title="Analytics Dashboard",
        icon=":material/monitoring:",
    ),
]

page = st.navigation(pages, position="sidebar")


# ---------------------------------------------------------
# FILTER CALLBACKS
# ---------------------------------------------------------

def reset_filter_widgets() -> None:
    """
    Reset filters using a button callback.

    Streamlit executes this callback before rerunning the app,
    so widget state can be changed without triggering
    StreamlitWidgetAlreadyInstantiatedError.
    """
    default_dates = st.session_state.get("_filter_default_dates")

    if default_dates is not None:
        st.session_state["global_date_range"] = tuple(default_dates)

    st.session_state["global_route_filter"] = []


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### TransitIQ")
    st.caption("Public transport analytics")
    st.divider()

    if has_dataset():
        full_data = get_full_dataset()

        source_name = st.session_state.get(
            "dataset_name",
            "Uploaded dataset",
        )

        if full_data is not None and not full_data.empty:
            st.markdown("#### Dataset filters")

            # Identify the available date column.
            date_column = (
                "service_datetime"
                if "service_datetime" in full_data.columns
                else "service_date"
                if "service_date" in full_data.columns
                else None
            )

            start_date = None
            end_date = None

            current_version = st.session_state.get(
                "dataset_version",
                0,
            )

            # -------------------------------------------------
            # DATE FILTER
            # -------------------------------------------------

            if date_column:
                parsed = pd.to_datetime(
                    full_data[date_column],
                    errors="coerce",
                ).dropna()

                if not parsed.empty:
                    minimum = parsed.min().date()
                    maximum = parsed.max().date()

                    default_dates = (minimum, maximum)

                    # Initialize filter state for a new dataset.
                    if (
                        st.session_state.get("_filter_dataset_version")
                        != current_version
                    ):
                        # Clear widget state for the previous dataset.
                        st.session_state["global_date_range"] = (
                            minimum,
                            maximum,
                        )
                        st.session_state["global_route_filter"] = []

                        st.session_state["_filter_dataset_version"] = (
                            current_version
                        )

                    # Store the current dataset's default dates.
                    st.session_state["_filter_default_dates"] = (
                        default_dates
                    )

                    # Ensure the existing selection remains within
                    # the uploaded dataset's valid date range.
                    current_dates = st.session_state.get(
                        "global_date_range",
                        default_dates,
                    )

                    if (
                        not isinstance(current_dates, (tuple, list))
                        or len(current_dates) != 2
                        or current_dates[0] < minimum
                        or current_dates[1] > maximum
                        or current_dates[0] > current_dates[1]
                    ):
                        st.session_state["global_date_range"] = (
                            default_dates
                        )

                    selected_dates = st.date_input(
                        "Service date range",
                        min_value=minimum,
                        max_value=maximum,
                        key="global_date_range",
                    )

                    if (
                        isinstance(selected_dates, (tuple, list))
                        and len(selected_dates) > 0
                    ):
                        start_date = selected_dates[0]

                        end_date = (
                            selected_dates[1]
                            if len(selected_dates) > 1
                            else None
                        )

                else:
                    st.caption(
                        "No valid dates are available in this dataset."
                    )

            else:
                st.caption(
                    "Date filters are unavailable: "
                    "no date field was found."
                )

            # -------------------------------------------------
            # ROUTE FILTER
            # -------------------------------------------------

            selected_routes: list[str] = []

            if "route_id" in full_data.columns:
                route_values = sorted(
                    full_data["route_id"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(route_values) > 500:
                    st.warning(
                        "Showing the first 500 route values in the filter."
                    )
                    route_values = route_values[:500]

                # Remove selections that are no longer available.
                current_routes = st.session_state.get(
                    "global_route_filter",
                    [],
                )

                valid_routes = [
                    route
                    for route in current_routes
                    if route in route_values
                ]

                if valid_routes != current_routes:
                    st.session_state["global_route_filter"] = valid_routes

                selected_routes = st.multiselect(
                    "Routes",
                    options=route_values,
                    key="global_route_filter",
                    help="Leave empty to include every route.",
                )

            else:
                st.caption(
                    "Route filters are unavailable: "
                    "no route field was found."
                )

            # -------------------------------------------------
            # APPLY FILTERS
            # -------------------------------------------------

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
                f"{len(filtered):,} of {len(full_data):,} observations"
            )
            st.caption(f"Dataset: {source_name}")

            # -------------------------------------------------
            # RESET FILTER BUTTON
            # -------------------------------------------------

            st.button(
                "Reset filters",
                use_container_width=True,
                on_click=reset_filter_widgets,
            )

        else:
            st.warning(
                "The uploaded dataset is empty. "
                "Upload a valid dataset to continue."
            )

    else:
        st.info(
            "No dataset uploaded. Open Data Management to upload "
            "a CSV or Parquet file."
        )

        # Clear filters left over from a previously loaded dataset.
        st.session_state["filter_start_date"] = None
        st.session_state["filter_end_date"] = None
        st.session_state["selected_routes"] = []
        st.session_state["filtered_df"] = None

    st.divider()

    st.caption(
        "Preview tables are limited to keep large uploads responsive."
    )


# ---------------------------------------------------------
# RENDER SELECTED PAGE
# ---------------------------------------------------------

page.run()