from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.io as pio

from src.analytics import (
    capacity_exceedance_summary,
    dataset_summary,
    peak_hour_analysis,
    route_aggregation,
)
from src.data_validation import validate_transport_data
from src.visualization import route_figure, trend_figure, weekday_hour_heatmap


def build_report_data(
    frame: pd.DataFrame,
    *,
    source_name: str = "Dataset",
    cleaning_summary: dict | None = None,
    model_evaluation: dict | None = None,
) -> dict:
    validation = validate_transport_data(frame)
    invalid_rows = validation["invalid_counts"]
    return {
        "report_title": "TransitIQ transport analytics report",
        "source_name": source_name,
        "dataset_overview": dataset_summary(frame),
        "data_quality": {
            "rows": validation["row_count"],
            "duplicate_records": validation["duplicate_count"],
            "missing_values": validation["missing_values"].to_dict(orient="records"),
            "invalid_values": invalid_rows,
            "warnings": validation["warnings"],
            "errors": validation["errors"],
        },
        "cleaning_summary": cleaning_summary or {},
        "route_analytics": route_aggregation(frame).to_dict(orient="records"),
        "peak_hour_analysis": peak_hour_analysis(frame).to_dict(orient="records"),
        "capacity_exceedance_indicators": capacity_exceedance_summary(frame),
        "model_evaluation": model_evaluation or {},
        "limitations": [
            "The report describes the selected dataset and filters only.",
            "Capacity exceedances are indicators for review, not proof of overcrowding.",
            "Forecast quality depends on historical coverage and data quality.",
            "Synthetic demo data, when selected, is not an official transit authority dataset.",
        ],
    }


def report_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2, default=str, allow_nan=False).encode("utf-8")


def report_csv_zip_bytes(report: dict) -> bytes:
    buffer = BytesIO()
    overview = pd.DataFrame([report["dataset_overview"]])
    routes = pd.DataFrame(report["route_analytics"])
    peaks = pd.DataFrame(report["peak_hour_analysis"])
    quality = pd.DataFrame(report["data_quality"]["missing_values"])
    metrics = pd.json_normalize(report.get("model_evaluation", {}), sep="_")
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for name, frame in (
            ("dataset_overview.csv", overview),
            ("data_quality_missing_values.csv", quality),
            ("route_analytics.csv", routes),
            ("peak_hour_analysis.csv", peaks),
            ("model_evaluation.csv", metrics),
        ):
            archive.writestr(name, frame.to_csv(index=False))
        archive.writestr(
            "report_summary.json",
            json.dumps(report, indent=2, default=str, allow_nan=False),
        )
    return buffer.getvalue()


def report_html_bytes(
    report: dict,
    frame: pd.DataFrame,
) -> bytes:
    def table(records, empty_message: str) -> str:
        data = pd.DataFrame(records)
        return (
            data.to_html(index=False, border=0, classes="data-table")
            if not data.empty
            else f"<p>{empty_message}</p>"
        )

    figures = [
        ("Passenger demand over time", trend_figure(frame)),
        ("Demand by route", route_figure(frame)),
        ("Demand by weekday and hour", weekday_hour_heatmap(frame)),
    ]
    chart_html = []
    for title, figure in figures:
        if figure is not None:
            chart_html.append(
                f"<section><h2>{title}</h2>"
                f"{pio.to_html(figure, full_html=False, include_plotlyjs='cdn')}"
                "</section>"
            )

    metrics = report.get("model_evaluation", {})
    overview = report["dataset_overview"]
    sections = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>TransitIQ analytics report</title>"
        "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;"
        "margin:32px auto;padding:0 20px;color:#16232e}h1{color:#0e5b52}"
        "h2{margin-top:36px;color:#164e63}.data-table{border-collapse:collapse;"
        "width:100%}.data-table th,.data-table td{padding:9px;border-bottom:1px solid #ddd;"
        "text-align:left}.note{padding:12px;background:#edf5f4;border-radius:8px}"
        "</style></head><body>",
        "<h1>TransitIQ transport analytics report</h1>",
        f"<p>Dataset: {report['source_name']}</p>",
        "<p class='note'>This report uses only the selected dataset and filters. "
        "Potential capacity exceedances are indicators for review, not proof of overcrowding.</p>",
        "<h2>Dataset overview</h2>",
        table([overview], "No overview values are available."),
        "<h2>Data quality: missing values</h2>",
        table(report["data_quality"]["missing_values"], "No columns were found."),
        "<h2>Route analytics</h2>",
        table(report["route_analytics"], "Route fields are unavailable."),
        "<h2>Peak-hour analysis</h2>",
        table(report["peak_hour_analysis"], "Service-time data is unavailable."),
        "<h2>Model evaluation</h2>",
        table([metrics] if metrics else [], "No model evaluation is available."),
        "".join(chart_html),
        "<h2>Limitations</h2><ul>"
        + "".join(f"<li>{item}</li>" for item in report["limitations"])
        + "</ul></body></html>",
    ]
    return "\n".join(sections).encode("utf-8")
