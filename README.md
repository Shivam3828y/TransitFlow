# TransitIQ — Big Data-Driven Public Transport Analytics

TransitIQ is a modular Streamlit application for inspecting public transport
passenger data, identifying demand patterns, comparing routes, processing data
with PySpark, and evaluating short-term demand forecasts.

The included dataset is reproducible synthetic demonstration data. It is not
observed ridership from Mumbai BEST, Mumbai Metro, or any other transport
authority.

## Abstract and problem statement

Transport planning depends on knowing when and where passenger demand occurs.
Operational datasets are often inconsistent, incomplete, or difficult to
compare across routes. A chart-only dashboard can summarize records but does
not provide an auditable cleaning workflow, scalable data processing, or
measured predictions.

TransitIQ provides one workflow for uploading, mapping, validating, cleaning,
aggregating, exploring, forecasting, and reporting on passenger observations.
It compares a lag-based baseline with a Random Forest using time-ordered
evaluation rather than random train/test shuffling.

## Objectives

- Accept CSV and Parquet passenger datasets with configurable column mapping.
- Report missing values, duplicates, malformed values, and schema details
  without silently changing uploaded data.
- Apply optional, auditable cleaning and genuine local PySpark DataFrame
  transformations.
- Explore demand by date, route, stop, service hour, and weekday/weekend.
- Compare chronological baseline and Random Forest forecasts with MAE, RMSE,
  and R-squared where suitable.
- Identify potential capacity-exceedance observations when capacity is supplied.
- Export processed Parquet data and dataset-specific HTML, JSON, and CSV reports.

## Existing and proposed system

| Existing approach | TransitIQ |
| --- | --- |
| Static charts without dataset context | Interactive charts calculated from the active dataset and filters |
| Manual column renaming before analysis | In-app mapping to canonical transport fields |
| Unexplained automatic cleaning | Validation first, explicit cleaning choices, and row-change summaries |
| Random model split that can leak future patterns | Chronological holdout and lag features shifted within each series |
| Unqualified “overcrowding” flags | Potential capacity-exceedance indicators with explicit limitations |

## Architecture

```text
Streamlit pages
    ├── Data management and column mapping
    ├── PySpark processing / small-file Pandas fallback
    ├── Exploratory analysis and Plotly charts
    ├── Chronological demand model and route forecast
    ├── Route and capacity analytics
    └── HTML, JSON, and CSV report exports
             │
             ├── Pandas / NumPy data validation and feature engineering
             ├── PySpark DataFrames in local[*] mode
             ├── Scikit-learn baseline and Random Forest
             └── CSV and Parquet files; optional saved joblib model
```

## Technology

- Python 3.11, Streamlit
- PySpark, Pandas, NumPy
- Plotly
- Scikit-learn and joblib
- CSV and Parquet (PyArrow)
- Pytest

## Modules

- `app.py` — multipage entry point and shared date/route filters.
- `pages/1_Data_Management.py` — upload, map, validate, inspect, and download data.
- `pages/2_Big_Data_Processing.py` — optional cleaning, Spark/fallback execution,
  aggregation, and Parquet export.
- `pages/3_Exploratory_Analysis.py` — interactive demand and data-quality charts.
- `pages/4_Demand_Prediction.py` — chronological baseline/forest evaluation and
  recursive route forecast.
- `pages/5_Route_Analytics.py` — route ranking, trend, peak periods, and capacity
  indicators.
- `pages/6_Reports.py` — selected-dataset HTML, JSON, and CSV-bundle downloads.
- `src/data_loader.py` — CSV/Parquet parsing and column mapping.
- `src/data_validation.py` — missing, duplicate, and invalid-value findings.
- `src/data_cleaning.py` — opt-in cleaning with a change summary.
- `src/spark_processor.py` — Spark transformations, aggregation, and explicit
  Pandas fallback.
- `src/analytics.py`, `src/visualization.py` — analysis functions and Plotly figures.
- `src/feature_engineering.py`, `src/prediction_model.py` — shifted time-series
  features, evaluation, forecasting, and optional model persistence.
- `src/report_generator.py` — dataset-specific report data and exports.
- `src/sample_data.py` — deterministic synthetic-data generator.
- `tests/` — validation, cleaning, analytics, feature, and model tests.

## Data-processing workflow

1. Upload CSV or Parquet, or use the included synthetic dataset.
2. Review the suggested mapping from uploaded fields to TransitIQ names.
3. Inspect dimensions, dtypes, missing values, duplicates, and invalid values.
4. Select optional duplicate, invalid-row, and missing-value handling.
5. Process with genuine Spark DataFrames in `local[*]`, inspect the schema,
   aggregate demand, and download Parquet output.
6. If Spark cannot start, datasets up to 100,000 rows may use the clearly labeled
   single-machine Pandas fallback. Larger files report the Spark error instead of
   collecting them into Pandas.

The Spark session is local to one machine. TransitIQ does **not** claim to use a
distributed Spark cluster.

## Machine-learning workflow

The target is `passenger_count`. Features include hour, day of week, month,
day of year, route, stop, the previous observation, and a shifted seven-record
rolling average. For each route/stop/hour series, lag features are shifted so
the current target cannot leak into its own predictors.

TransitIQ trains and evaluates using an 80/20 chronological holdout. It compares
the Random Forest with a lag-1 baseline and reports MAE, RMSE, and R-squared
when the holdout target supports it. A route forecast recursively feeds earlier
predictions into the later horizon. This is a short-term extrapolation, not a
guaranteed schedule or a substitute for live service data.

The app will not train on fewer than 50 usable feature rows across 14 dates.
For large uploads, use sidebar filters to keep model training at or below
100,000 feature rows. Trained models are saved only after the user selects
**Save model to the project**.

## Installation and running on Windows

Install Python 3.11 and a Java 17+ JDK (needed for PySpark). From PowerShell:

```powershell
cd transitiq
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8000
```

Open the local Streamlit address printed by the command.

## Running in VS Code terminal (Linux/macOS)

```bash
cd transitiq
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8000
```

Run tests from the same activated environment:

```bash
python -m pytest tests -q
```

On Replit, the Python 3.11 environment and dependencies are configured for the
project. The app workflow runs:

```bash
cd transitiq
../.pythonlibs/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8000
```

## Sample dataset

`data/sample_transport_data.csv` is generated with a fixed random seed and
contains 18,000 synthetic observations across 90 dates, eight route IDs, five
stops per route, and five service time windows. Demand varies by route, stop,
weekday/weekend, time of day, and occasional event multipliers.

Fields:

| Field | Meaning |
| --- | --- |
| `record_id` | Unique synthetic observation identifier |
| `service_date` | ISO service date |
| `service_time` | Local service-time label |
| `route_id` | Synthetic route identifier |
| `stop_id` | Synthetic stop identifier |
| `passenger_count` | Synthetic passenger observation count |
| `vehicle_capacity` | Example vehicle capacity for indicator analysis |
| `journey_duration` | Synthetic journey duration in minutes |
| `distance_km` | Synthetic distance in kilometres |

These sample records are for demonstrating the application only and must not be
presented as official ridership data.

## Limitations

- Spark uses `local[*]`; a remote or multi-node cluster is not configured.
- The built-in data is synthetic and cannot validate real-world forecasts.
- Forecasts need adequate dated passenger history, and recursive errors can grow
  with the forecast horizon.
- Capacity-exceedance flags are not proof of actual overcrowding.
- Geographic maps appear only when uploaded latitude and longitude are available;
  TransitIQ never invents coordinates.
- Large datasets may need filters for responsive charting and model training.
- Data upload and optional model persistence use the current application
  environment; a multi-user production storage layer is not configured.

## Future enhancements

- Connect to an approved live transit feed and maintain a provenance trail.
- Add holidays, service disruptions, weather, and scheduled service as
  forecast features.
- Add a configurable remote Spark cluster and durable object storage.
- Add model versioning and route-level forecast monitoring.
- Add role-based access, deployment-specific data retention, and audit controls.
