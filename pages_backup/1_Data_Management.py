from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from config import CANONICAL_FIELDS, FIELD_LABELS, MAX_PREVIEW_ROWS, SAMPLE_DATA_PATH
from src.data_loader import (
    apply_column_mapping,
    suggest_column_mapping,
)
from src.data_validation import validate_transport_data
from src.ui import (
    cached_load_transport_file,
    dataframe_download_bytes,
    get_full_dataset,
    render_header,
    show_dataset_preview,
)

render_header(
    "Data Management",
    "Upload a CSV or Parquet file, map its columns, and review quality findings "
    "before analyzing it.",
)

uploaded = st.file_uploader(
    "Upload a transport dataset",
    type=["csv", "parquet", "pq"],
    help="Uploads are parsed as data only; uploaded files are never executed.",
)

if uploaded is not None:
    content = uploaded.getvalue()
    file_size = len(content)
    st.caption(
        f"{uploaded.name} · {file_size / (1024 * 1024):.2f} MB · "
        "CSV or Parquet"
    )
    try:
        raw = cached_load_transport_file(content, uploaded.name)
    except Exception as exc:
        st.error(str(exc))
        raw = None
    if raw is not None:
        st.write(f"Uploaded file dimensions: **{len(raw):,} rows × {len(raw.columns):,} columns**")
        guesses = suggest_column_mapping(list(raw.columns))
        source_options = ["— Ignore this field —", *[str(c) for c in raw.columns]]
        with st.form("column_mapping_form"):
            st.markdown("#### Map uploaded columns")
            st.caption(
                "TransitIQ suggests matches by column name. Review them before "
                "applying; mapping does not clean or alter values."
            )
            selected_mapping = {}
            columns = st.columns(3)
            for index, field in enumerate(CANONICAL_FIELDS):
                suggestion = guesses.get(field)
                selected_index = (
                    source_options.index(suggestion)
                    if suggestion in source_options
                    else 0
                )
                selected_mapping[field] = columns[index % 3].selectbox(
                    FIELD_LABELS[field],
                    options=source_options,
                    index=selected_index,
                    key=f"mapping_{uploaded.name}_{field}",
                )
            apply_mapping = st.form_submit_button(
                "Validate and use this dataset",
                type="primary",
                use_container_width=True,
            )
        if apply_mapping:
            mapping = {
                field: (
                    source
                    if source != "— Ignore this field —"
                    else None
                )
                for field, source in selected_mapping.items()
            }
            try:
                mapped = apply_column_mapping(raw, mapping)
                quality = validate_transport_data(mapped)
                if quality["errors"]:
                    for message in quality["errors"]:
                        st.error(message)
                else:
                    st.session_state["transport_df"] = mapped
                    st.session_state["dataset_name"] = f"Uploaded: {uploaded.name}"
                    st.session_state["dataset_is_synthetic"] = False
                    st.session_state["dataset_file_size"] = file_size
                    st.session_state["dataset_version"] = (
                        st.session_state.get("dataset_version", 0) + 1
                    )
                    for key in (
                        "model_result",
                        "model_signature",
                        "processing_result",
                        "processing_summary",
                        "forecast_result",
                        "report_payload",
                    ):
                        st.session_state.pop(key, None)
                    st.session_state.pop("global_date_range", None)
                    st.success(
                        f"Dataset loaded: {len(mapped):,} observations. "
                        "No cleaning was applied."
                    )
                    for message in quality["warnings"]:
                        st.warning(message)
            except Exception as exc:
                st.error(f"Could not apply the selected column mapping: {exc}")

st.divider()
left, right = st.columns([3, 1])
left.markdown("#### Active dataset")
if right.button("Load synthetic sample", use_container_width=True):
    try:
        sample = pd.read_csv(SAMPLE_DATA_PATH, low_memory=False)
        st.session_state["transport_df"] = sample
        st.session_state["dataset_name"] = "Synthetic sample transport data"
        st.session_state["dataset_is_synthetic"] = True
        st.session_state["dataset_file_size"] = os.path.getsize(SAMPLE_DATA_PATH)
        st.session_state["dataset_version"] = st.session_state.get(
            "dataset_version", 0
        ) + 1
        for key in (
            "model_result",
            "model_signature",
            "processing_result",
            "processing_summary",
            "forecast_result",
            "report_payload",
        ):
            st.session_state.pop(key, None)
        st.rerun()
    except Exception as exc:
        st.error(f"Could not load the sample dataset: {exc}")

data = get_full_dataset()
size = st.session_state.get("dataset_file_size")
if size is None and Path(SAMPLE_DATA_PATH).exists():
    size = os.path.getsize(SAMPLE_DATA_PATH)
st.caption(
    f"**{st.session_state.get('dataset_name', 'Dataset')}** · "
    f"{len(data):,} rows × {len(data.columns):,} columns · "
    f"{len(data.columns)} fields · "
    f"{(size / (1024 * 1024)):.2f} MB"
    if size is not None
    else f"**{st.session_state.get('dataset_name', 'Dataset')}** · "
    f"{len(data):,} rows × {len(data.columns):,} columns"
)

quality = validate_transport_data(data)
if quality["errors"]:
    for message in quality["errors"]:
        st.error(message)
if quality["warnings"]:
    for message in quality["warnings"]:
        st.warning(message)

quality_metrics = st.columns(4)
quality_metrics[0].metric("Rows", f"{len(data):,}")
quality_metrics[1].metric("Columns", f"{len(data.columns):,}")
quality_metrics[2].metric("Duplicate records", f"{quality['duplicate_count']:,}")
quality_metrics[3].metric(
    "Missing values",
    f"{int(data.isna().sum().sum()):,}",
)

with st.expander("Column names and data types", expanded=True):
    schema = pd.DataFrame(
        {
            "column": [str(column) for column in data.columns],
            "data_type": [str(dtype) for dtype in data.dtypes],
            "non_missing": [int(data[column].notna().sum()) for column in data.columns],
            "missing": [int(data[column].isna().sum()) for column in data.columns],
        }
    )
    st.dataframe(schema, hide_index=True, use_container_width=True)

with st.expander("Missing values and invalid values"):
    st.dataframe(quality["missing_values"], hide_index=True, use_container_width=True)
    if quality["invalid_counts"]:
        st.dataframe(
            pd.DataFrame(
                [
                    {"issue": issue, "affected_rows": count}
                    for issue, count in quality["invalid_counts"].items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("No invalid numeric values or unparseable dates were detected.")

st.markdown("#### Dataset preview")
show_dataset_preview(data, MAX_PREVIEW_ROWS)

st.markdown("#### Download active dataset")
download_csv = dataframe_download_bytes(data, "csv")
st.download_button(
    "Download active dataset as CSV",
    data=download_csv,
    file_name="transitiq_active_dataset.csv",
    mime="text/csv",
)
try:
    download_parquet = dataframe_download_bytes(data, "parquet")
    st.download_button(
        "Download active dataset as Parquet",
        data=download_parquet,
        file_name="transitiq_active_dataset.parquet",
        mime="application/vnd.apache.parquet",
    )
except Exception as exc:
    st.info(f"Parquet export is unavailable for this dataset: {exc}")
