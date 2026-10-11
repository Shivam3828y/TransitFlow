from __future__ import annotations

import streamlit as st

from src.report_generator import (
    build_report_data,
    report_csv_zip_bytes,
    report_html_bytes,
    report_json_bytes,
)
from src.ui import dataset_signature, get_filtered_dataset, render_header

data = get_filtered_dataset()
signature = dataset_signature(data)
render_header(
    "Reports",
    "Generate downloadable HTML, JSON, and CSV reports from the current dataset "
    "and active sidebar filters.",
)
st.caption(
    f"Report scope: {len(data):,} selected observations · "
    f"{st.session_state.get('dataset_name', 'Dataset')}"
)

if st.button("Generate report for current selection", type="primary"):
    processing = (
        st.session_state.get("processing_summary")
        if st.session_state.get("processing_signature") == signature
        else None
    )
    model_result = (
        st.session_state.get("model_result")
        if st.session_state.get("model_signature") == signature
        else None
    )
    evaluation = model_result.get("evaluation") if model_result else None
    with st.spinner("Building report tables and charts..."):
        payload = build_report_data(
            data,
            source_name=st.session_state.get("dataset_name", "Dataset"),
            cleaning_summary=processing,
            model_evaluation=evaluation,
        )
        st.session_state["report_payload"] = payload
        st.session_state["report_signature"] = signature

report = st.session_state.get("report_payload")
if report and st.session_state.get("report_signature") == signature:
    st.success("Report data is ready and matches the current selection.")
    with st.expander("Report preview", expanded=True):
        st.json(report, expanded=False)
        st.markdown("**Route analytics**")
        st.dataframe(
            report["route_analytics"],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Peak-hour analysis**")
        st.dataframe(
            report["peak_hour_analysis"],
            hide_index=True,
            use_container_width=True,
        )

    columns = st.columns(3)
    columns[0].download_button(
        "Download interactive HTML report",
        data=report_html_bytes(report, data),
        file_name="transitiq_analytics_report.html",
        mime="text/html",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download JSON report",
        data=report_json_bytes(report),
        file_name="transitiq_analytics_report.json",
        mime="application/json",
        use_container_width=True,
    )
    columns[2].download_button(
        "Download CSV report bundle",
        data=report_csv_zip_bytes(report),
        file_name="transitiq_analytics_report_csv.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.caption(
        "The HTML report includes interactive demand and route charts when the "
        "corresponding fields are available. The CSV download is a ZIP of overview, "
        "quality, route, peak-hour, and model-metric tables."
    )
elif report:
    st.info(
        "The selected dataset or filters changed. Generate a new report so exported "
        "results match the current selection."
    )
else:
    st.info(
        "Generate a report to include the selected dataset's overview, data-quality "
        "findings, available cleaning/model results, route analytics, peak-hour "
        "analysis, charts, and stated limitations."
    )
