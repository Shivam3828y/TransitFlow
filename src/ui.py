from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from config import MAX_PREVIEW_ROWS
from src.analytics import filter_transport_data
from src.data_loader import load_transport_file


@st.cache_data(show_spinner=False, max_entries=3)
def cached_load_transport_file(
    file_bytes: bytes,
    filename: str,
) -> pd.DataFrame:
    """Load and cache a user-uploaded transport dataset."""
    return load_transport_file(file_bytes, filename)


def ensure_dataset() -> None:
    """
    Initialize an empty dataset state.

    No sample or default dataset is loaded automatically.
    """
    if "transport_df" not in st.session_state:
        st.session_state["transport_df"] = None
        st.session_state["dataset_name"] = None
        st.session_state["dataset_is_synthetic"] = False
        st.session_state["dataset_version"] = 0
        st.session_state["filtered_df"] = None


def has_dataset() -> bool:
    """Return True only when a dataset has been uploaded."""
    ensure_dataset()

    frame = st.session_state.get("transport_df")

    return isinstance(frame, pd.DataFrame) and not frame.empty


def get_full_dataset() -> pd.DataFrame | None:
    """Return the uploaded dataset, or None if no dataset is available."""
    ensure_dataset()

    return st.session_state.get("transport_df")


def get_filtered_dataset() -> pd.DataFrame | None:
    """Return the filtered dataset, or the uploaded dataset."""
    ensure_dataset()

    if not has_dataset():
        return None

    filtered = st.session_state.get("filtered_df")

    if isinstance(filtered, pd.DataFrame):
        return filtered

    return st.session_state["transport_df"]


def dataset_signature(frame: pd.DataFrame) -> tuple:
    """Create a signature for the currently selected dataset."""
    date_column = (
        "service_datetime"
        if "service_datetime" in frame.columns
        else "service_date"
        if "service_date" in frame.columns
        else None
    )

    date_bounds = ("", "")

    if date_column:
        parsed = pd.to_datetime(
            frame[date_column],
            errors="coerce",
        ).dropna()

        if not parsed.empty:
            date_bounds = (
                str(parsed.min()),
                str(parsed.max()),
            )

    return (
        st.session_state.get("dataset_version", 0),
        st.session_state.get("dataset_name", "Dataset"),
        len(frame),
        date_bounds,
        tuple(st.session_state.get("selected_routes", [])),
    )


def render_header(
    title: str,
    caption: str | None = None,
) -> None:
    """Render a page heading and optional caption."""
    st.title(title)

    if caption:
        st.caption(caption)

    if not has_dataset():
        st.info(
            "No dataset uploaded yet. Open Data Management and upload "
            "your transport dataset to view data, analytics, and charts."
        )


def format_number(value, digits: int = 0) -> str:
    """Format a number for display."""
    if value is None or pd.isna(value):
        return "—"

    return f"{value:,.{digits}f}"


def dataframe_download_bytes(
    frame: pd.DataFrame,
    file_format: str,
) -> bytes:
    """Convert a DataFrame into downloadable CSV or Parquet bytes."""
    if file_format.lower() == "csv":
        return frame.to_csv(index=False).encode("utf-8")

    if file_format.lower() in {"parquet", "pq"}:
        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()

    raise ValueError("Unsupported download format. Use CSV or Parquet.")


def filtered_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the selected date and route filters."""
    return filter_transport_data(
        frame,
        start_date=st.session_state.get("filter_start_date"),
        end_date=st.session_state.get("filter_end_date"),
        routes=st.session_state.get("selected_routes"),
    )


def show_dataset_quality(frame: pd.DataFrame) -> dict:
    """Validate the uploaded dataset and display quality issues."""
    from src.data_validation import validate_transport_data

    report = validate_transport_data(frame)

    if report["errors"]:
        for message in report["errors"]:
            st.error(message)

    if report["warnings"]:
        for message in report["warnings"]:
            st.warning(message)

    if report["invalid_counts"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "issue": issue,
                        "affected_rows": count,
                    }
                    for issue, count in report["invalid_counts"].items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    return report


def show_dataset_preview(
    frame: pd.DataFrame,
    row_limit: int = MAX_PREVIEW_ROWS,
) -> None:
    """Display a preview of the uploaded dataset."""
    st.dataframe(
        frame.head(row_limit),
        hide_index=True,
        use_container_width=True,
    )

    if len(frame) > row_limit:
        st.caption(
            f"Showing {row_limit:,} of {len(frame):,} rows."
        )