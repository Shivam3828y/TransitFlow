from __future__ import annotations

import logging
import traceback

import streamlit as st

from config import MAX_SPARK_FALLBACK_ROWS
from src.spark_processor import (
    process_with_pandas_fallback,
    process_with_spark,
    spark_parquet_zip_bytes,
)
from src.ui import (
    dataset_signature,
    get_filtered_dataset,
    has_dataset,
    render_header,
)


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

render_header(
    "Big Data Processing",
    "Process your uploaded transport dataset using PySpark "
    "with a clearly labeled Pandas fallback.",
)


# ---------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------

if not has_dataset():
    st.info(
        "No dataset uploaded. Open Data Management, upload a CSV "
        "or Parquet file, and validate it before starting processing."
    )
    st.stop()


data = get_filtered_dataset()

if data is None or data.empty:
    st.warning(
        "The selected filters returned no records. Adjust the "
        "sidebar filters before running processing."
    )
    st.stop()


current_signature = dataset_signature(data)


# ---------------------------------------------------------
# CLEAR STALE RESULTS
# ---------------------------------------------------------

if (
    st.session_state.get("processing_signature")
    != current_signature
):
    st.session_state.pop("processing_result", None)
    st.session_state.pop("processing_summary", None)
    st.session_state.pop("processing_signature", None)
    st.session_state.pop("parquet_download_bytes", None)
    st.session_state.pop("parquet_download_name", None)
    st.session_state.pop("parquet_download_mime", None)


st.caption(
    f"Current input: {len(data):,} observations after applying "
    "the sidebar filters."
)


# ---------------------------------------------------------
# PROCESSING OPTIONS
# ---------------------------------------------------------

with st.form("spark_processing_form"):

    remove_duplicates = st.checkbox(
        "Remove duplicate records",
        value=True,
    )

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
        width="stretch",
    )


strategy = {
    "Keep missing values": "keep",
    "Drop rows with any missing values": "drop_rows",
    "Fill numeric fields with their median": "median_numeric",
}[missing_label]


# ---------------------------------------------------------
# RUN PYSPARK PROCESSING
# ---------------------------------------------------------

if run_processing:

    # Remove stale results before starting a new operation.
    st.session_state.pop("processing_result", None)
    st.session_state.pop("processing_summary", None)
    st.session_state.pop("processing_signature", None)
    st.session_state.pop("parquet_download_bytes", None)
    st.session_state.pop("parquet_download_name", None)
    st.session_state.pop("parquet_download_mime", None)

    spark_error_text = None
    spark_traceback_text = None

    with st.spinner(
        "Processing the uploaded dataset with PySpark..."
    ):
        try:
            result = process_with_spark(
                data,
                remove_duplicates=remove_duplicates,
                remove_invalid=remove_invalid,
                missing_strategy=strategy,
            )

            # Validate the expected result structure.
            required_keys = [
                "engine",
                "execution_mode",
                "input_rows",
                "output_rows",
                "duplicates_detected_and_removed",
                "processing_seconds",
                "missing_values_detected",
                "invalid_rows_detected",
                "invalid_rows_removed",
                "rows_removed_for_missing_values",
                "aggregation_preview",
            ]

            missing_keys = [
                key for key in required_keys
                if key not in result
            ]

            if missing_keys:
                raise KeyError(
                    "PySpark returned an incomplete result. "
                    f"Missing keys: {missing_keys}"
                )

            # Save the complete result for display and export.
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

            st.session_state["processing_signature"] = (
                current_signature
            )

            st.session_state["processing_engine_used"] = (
                "PySpark"
            )

            st.success(
                "PySpark processing completed successfully."
            )

        except Exception as exc:

            # Capture the real exception instead of claiming that
            # Spark startup necessarily failed.
            spark_error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            spark_traceback_text = traceback.format_exc()

            logger.exception(
                "PySpark processing failed."
            )

            if len(data) > MAX_SPARK_FALLBACK_ROWS:

                st.error(
                    "PySpark processing failed. Pandas fallback was "
                    "not attempted because the dataset exceeds the "
                    "configured fallback limit."
                )

                st.write(
                    f"Dataset rows: {len(data):,}"
                )

                st.write(
                    "Pandas fallback limit: "
                    f"{MAX_SPARK_FALLBACK_ROWS:,}"
                )

                st.code(
                    spark_error_text,
                    language="text",
                )

                with st.expander(
                    "View complete PySpark traceback"
                ):
                    st.code(
                        spark_traceback_text,
                        language="text",
                    )

            else:

                st.warning(
                    "PySpark processing failed. Attempting the "
                    "Pandas fallback for this dataset."
                )

                st.error(
                    f"PySpark error: {spark_error_text}"
                )

                with st.spinner(
                    "Processing the dataset with Pandas fallback..."
                ):
                    try:
                        result = process_with_pandas_fallback(
                            data,
                            remove_duplicates=remove_duplicates,
                            remove_invalid=remove_invalid,
                            missing_strategy=strategy,
                        )

                        result["spark_error"] = spark_error_text
                        result["spark_traceback"] = (
                            spark_traceback_text
                        )

                        # Check the minimum fields required by the UI.
                        required_fallback_keys = [
                            "engine",
                            "execution_mode",
                            "input_rows",
                            "output_rows",
                            "duplicates_detected_and_removed",
                            "processing_seconds",
                            "missing_values_detected",
                            "invalid_rows_detected",
                            "invalid_rows_removed",
                            "rows_removed_for_missing_values",
                            "aggregation_preview",
                            "preview",
                            "cleaned_pandas_df",
                        ]

                        missing_fallback_keys = [
                            key
                            for key in required_fallback_keys
                            if key not in result
                        ]

                        if missing_fallback_keys:
                            raise KeyError(
                                "Pandas fallback returned an "
                                "incomplete result. Missing keys: "
                                f"{missing_fallback_keys}"
                            )

                        st.session_state["processing_result"] = (
                            result
                        )

                        st.session_state["processing_summary"] = (
                            result.get("cleaning_summary", {})
                        )

                        st.session_state[
                            "processing_signature"
                        ] = current_signature

                        st.session_state[
                            "processing_engine_used"
                        ] = "Pandas fallback"

                        st.warning(
                            "Processing completed using Pandas. "
                            "Review the PySpark error above to "
                            "identify why Spark processing failed."
                        )

                        with st.expander(
                            "View complete PySpark traceback"
                        ):
                            st.code(
                                spark_traceback_text,
                                language="text",
                            )

                    except Exception as fallback_error:

                        fallback_traceback = traceback.format_exc()

                        logger.exception(
                            "Pandas fallback processing failed."
                        )

                        st.error(
                            "Both PySpark and Pandas processing failed."
                        )

                        st.markdown("**PySpark error**")

                        st.code(
                            spark_error_text,
                            language="text",
                        )

                        st.markdown("**Pandas fallback error**")

                        st.code(
                            f"{type(fallback_error).__name__}: "
                            f"{fallback_error}",
                            language="text",
                        )

                        with st.expander(
                            "View complete PySpark traceback"
                        ):
                            st.code(
                                spark_traceback_text,
                                language="text",
                            )

                        with st.expander(
                            "View complete Pandas traceback"
                        ):
                            st.code(
                                fallback_traceback,
                                language="text",
                            )


# ---------------------------------------------------------
# DISPLAY PROCESSING RESULTS
# ---------------------------------------------------------

result = st.session_state.get("processing_result")

if (
    result is not None
    and st.session_state.get("processing_signature")
    == current_signature
):

    st.divider()
    st.subheader("Processing result")

    st.caption(
        f"Engine: **{result['engine']}** · "
        f"{result['execution_mode']}"
    )

    # Row and processing metrics.
    stats = st.columns(4)

    stats[0].metric(
        "Input rows",
        f"{result['input_rows']:,}",
    )

    stats[1].metric(
        "Processed rows",
        f"{result['output_rows']:,}",
    )

    stats[2].metric(
        "Duplicate records removed",
        f"{result['duplicates_detected_and_removed']:,}",
    )

    stats[3].metric(
        "Processing time",
        f"{result['processing_seconds']:.3f} s",
    )

    # Data-quality metrics.
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

    # Spark results.
    if result["engine"] == "PySpark local[*]":

        with st.expander(
            "Spark schema",
            expanded=True,
        ):
            st.code(
                result["cleaned_spark_df"].schema.simpleString(),
                language="text",
            )

        st.markdown(
            "#### Route, stop, and date aggregation"
        )

        st.dataframe(
            result["aggregation_preview"],
            hide_index=True,
            width="stretch",
        )

    # Pandas fallback results.
    else:

        st.markdown(
            "#### Processed-data preview"
        )

        st.dataframe(
            result["preview"],
            hide_index=True,
            width="stretch",
        )

        st.markdown(
            "#### Route aggregation"
        )

        st.dataframe(
            result["aggregation_preview"],
            hide_index=True,
            width="stretch",
        )


    # -----------------------------------------------------
    # PARQUET EXPORT
    # -----------------------------------------------------

    st.markdown(
        "#### Download processed Parquet data"
    )

    if result["engine"] == "PySpark local[*]":

        st.caption(
            "Spark exports are provided as a ZIP containing "
            "the Parquet part file(s). Extract the ZIP and read "
            "the files together as a Parquet dataset."
        )

    else:

        st.caption(
            "The Pandas fallback exports a single Parquet file."
        )

    if st.button(
        "Prepare Parquet download",
        key="prepare_parquet_download",
    ):

        with st.spinner(
            "Writing processed data to Parquet..."
        ):

            try:

                if result["engine"] == "PySpark local[*]":

                    payload = spark_parquet_zip_bytes(
                        result["cleaned_spark_df"]
                    )

                    name = "transitiq_processed_parquet.zip"
                    mime = "application/zip"

                else:

                    import io

                    buffer = io.BytesIO()

                    result["cleaned_pandas_df"].to_parquet(
                        buffer,
                        index=False,
                    )

                    payload = buffer.getvalue()

                    name = "transitiq_processed.parquet"

                    mime = (
                        "application/vnd.apache.parquet"
                    )

                st.session_state[
                    "parquet_download_bytes"
                ] = payload

                st.session_state[
                    "parquet_download_name"
                ] = name

                st.session_state[
                    "parquet_download_mime"
                ] = mime

                st.success(
                    "Parquet export is ready."
                )

            except Exception as exc:

                logger.exception(
                    "Parquet export failed."
                )

                st.error(
                    f"Parquet export failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                with st.expander(
                    "View export traceback"
                ):
                    st.code(
                        traceback.format_exc(),
                        language="text",
                    )

    if st.session_state.get(
        "parquet_download_bytes"
    ) is not None:

        st.download_button(
            "Download processed Parquet",
            data=st.session_state[
                "parquet_download_bytes"
            ],
            file_name=st.session_state[
                "parquet_download_name"
            ],
            mime=st.session_state[
                "parquet_download_mime"
            ],
            key="download_processed_parquet",
        )