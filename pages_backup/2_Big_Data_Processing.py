from __future__ import annotations

import streamlit as st

from config import MAX_SPARK_FALLBACK_ROWS
from src.spark_processor import (
    process_with_pandas_fallback,
    process_with_spark,
    spark_parquet_zip_bytes,
)
from src.ui import dataset_signature, get_filtered_dataset, render_header

data = get_filtered_dataset()
render_header(
    "Big Data Processing",
    "Run deduplication, invalid-value handling, optional missing-value treatment, "
    "route/date filtering, and passenger aggregation.",
)

st.info(
    "The Spark engine runs in local[*] mode on this machine. It is genuine Spark "
    "DataFrame processing, but it is not a distributed cluster. If Java or Spark "
    "cannot start, smaller datasets can use a clearly labeled Pandas fallback."
)
st.caption(
    f"Current input: {len(data):,} observations after applying the sidebar filters."
)

with st.form("spark_processing_form"):
    remove_duplicates = st.checkbox("Remove duplicate records", value=True)
    remove_invalid = st.checkbox(
        "Remove invalid passenger counts and invalid numeric/date values",
        value=True,
    )
    missing_label = st.selectbox(
        "Missing-value strategy",
        options=[
            "Keep missing values",
            "Drop rows with any missing values",
            "Fill numeric fields with their median",
        ],
        help="Cleaning is opt-in and its effects are reported below.",
    )
    run_processing = st.form_submit_button(
        "Process with PySpark",
        type="primary",
        use_container_width=True,
    )

strategy = {
    "Keep missing values": "keep",
    "Drop rows with any missing values": "drop_rows",
    "Fill numeric fields with their median": "median_numeric",
}[missing_label]

if run_processing:
    with st.spinner("Processing the selected dataset..."):
        try:
            result = process_with_spark(
                data,
                remove_duplicates=remove_duplicates,
                remove_invalid=remove_invalid,
                missing_strategy=strategy,
            )
            st.session_state["processing_result"] = result
            st.session_state["processing_summary"] = {
                key: value
                for key, value in result.items()
                if key not in {
                    "cleaned_spark_df",
                    "aggregation_spark_df",
                    "aggregation_preview",
                    "preview",
                    "cleaned_pandas_df",
                }
            }
            st.session_state["processing_signature"] = dataset_signature(data)
            st.session_state.pop("parquet_download_bytes", None)
        except Exception as spark_error:
            if len(data) > MAX_SPARK_FALLBACK_ROWS:
                st.error(
                    "Spark did not start, and this dataset is too large for the "
                    f"Pandas fallback ({len(data):,} rows; limit "
                    f"{MAX_SPARK_FALLBACK_ROWS:,}). Spark reported: {spark_error}"
                )
            else:
                try:
                    result = process_with_pandas_fallback(
                        data,
                        remove_duplicates=remove_duplicates,
                        remove_invalid=remove_invalid,
                        missing_strategy=strategy,
                    )
                    result["spark_start_error"] = str(spark_error)
                    st.session_state["processing_result"] = result
                    st.session_state["processing_summary"] = result["cleaning_summary"]
                    st.session_state["processing_signature"] = dataset_signature(data)
                    st.session_state.pop("parquet_download_bytes", None)
                    st.warning(
                        f"Spark could not start, so Pandas processed this small "
                        f"dataset. Reason: {spark_error}"
                    )
                except Exception as fallback_error:
                    st.error(
                        f"Both processing engines failed. Spark: {spark_error} "
                        f"· Pandas: {fallback_error}"
                    )

result = st.session_state.get("processing_result")
if result:
    st.divider()
    st.subheader("Processing result")
    st.caption(
        f"Engine: **{result['engine']}** · {result['execution_mode']}"
    )
    stats = st.columns(4)
    stats[0].metric("Input rows", f"{result['input_rows']:,}")
    stats[1].metric("Processed rows", f"{result['output_rows']:,}")
    stats[2].metric(
        "Duplicate records removed",
        f"{result['duplicates_detected_and_removed']:,}",
    )
    stats[3].metric(
        "Processing time",
        f"{result['processing_seconds']:.3f} s",
    )
    stats = st.columns(4)
    stats[0].metric(
        "Missing values detected",
        f"{result['missing_values_detected']:,}",
    )
    stats[1].metric(
        "Invalid rows detected",
        f"{result['invalid_rows_detected']:,}",
    )
    stats[2].metric(
        "Invalid rows removed",
        f"{result['invalid_rows_removed']:,}",
    )
    stats[3].metric(
        "Rows removed for missing values",
        f"{result['rows_removed_for_missing_values']:,}",
    )

    if result["engine"] == "PySpark local[*]":
        with st.expander("Spark schema", expanded=True):
            st.code(result["cleaned_spark_df"].schema.simpleString(), language="text")
        st.markdown("#### Route, stop, and date aggregation")
        st.dataframe(
            result["aggregation_preview"],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.markdown("#### Processed-data preview")
        st.dataframe(result["preview"], hide_index=True, use_container_width=True)
        st.markdown("#### Route aggregation")
        st.dataframe(
            result["aggregation_preview"],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Download processed Parquet data")
    st.caption(
        "Spark exports are downloaded as a ZIP containing the Parquet part file(s); "
        "extract the ZIP and read them together as a Parquet dataset."
        if result["engine"] == "PySpark local[*]"
        else "The Pandas fallback exports a single Parquet file."
    )
    if st.button("Prepare Parquet download"):
        with st.spinner("Writing processed data to Parquet..."):
            try:
                if result["engine"] == "PySpark local[*]":
                    payload = spark_parquet_zip_bytes(result["cleaned_spark_df"])
                    name = "transitiq_processed_parquet.zip"
                    mime = "application/zip"
                else:
                    payload = result["cleaned_pandas_df"].to_parquet(index=False)
                    name = "transitiq_processed.parquet"
                    mime = "application/vnd.apache.parquet"
                st.session_state["parquet_download_bytes"] = payload
                st.session_state["parquet_download_name"] = name
                st.session_state["parquet_download_mime"] = mime
            except Exception as exc:
                st.error(f"Parquet export failed: {exc}")
    if st.session_state.get("parquet_download_bytes"):
        st.download_button(
            "Download processed Parquet",
            data=st.session_state["parquet_download_bytes"],
            file_name=st.session_state["parquet_download_name"],
            mime=st.session_state["parquet_download_mime"],
        )
