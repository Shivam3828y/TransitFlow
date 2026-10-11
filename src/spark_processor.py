from __future__ import annotations

import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pandas as pd

from config import NUMERIC_FIELDS
from src.analytics import filter_transport_data, route_aggregation, stop_aggregation
from src.data_cleaning import clean_transport_data, invalid_transport_rows

_SPARK_SESSION = None


def get_spark_session():
    """Create one local Spark session; local[*] is not a distributed cluster."""
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        return _SPARK_SESSION
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    try:
        from pyspark.sql import SparkSession

        _SPARK_SESSION = (
            SparkSession.builder.master("local[*]")
            .appName("TransitIQ")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )
        _SPARK_SESSION.sparkContext.setLogLevel("ERROR")
        return _SPARK_SESSION
    except Exception as exc:
        raise RuntimeError(
            "Local Spark could not start. Check that Java 17 or newer is available. "
            f"Spark reported: {exc}"
        ) from exc


def _spark_input(frame: pd.DataFrame, spark):
    from pyspark.sql import functions as F

    safe = frame.copy()
    for column in NUMERIC_FIELDS.intersection(safe.columns):
        safe[column] = pd.to_numeric(safe[column], errors="coerce")
    safe = safe.astype(object).where(pd.notna(safe), None)
    result = spark.createDataFrame(safe)
    for column in NUMERIC_FIELDS.intersection(result.columns):
        result = result.withColumn(column, F.col(column).cast("double"))
    return result


def _spark_aggregation(frame):
    from pyspark.sql import functions as F

    if "passenger_count" not in frame.columns:
        return frame.limit(0)
    groups = [name for name in ("route_id", "stop_id") if name in frame.columns]
    if "service_date" in frame.columns:
        frame = frame.withColumn("service_date", F.to_date("service_date"))
        groups.append("service_date")
    elif "service_datetime" in frame.columns:
        frame = frame.withColumn("service_date", F.to_date("service_datetime"))
        groups.append("service_date")
    if not groups:
        groups = []
    return frame.groupBy(*groups).agg(
        F.count("*").alias("observations"),
        F.sum("passenger_count").alias("total_passengers"),
        F.avg("passenger_count").alias("avg_passengers"),
    )


def process_with_spark(
    frame: pd.DataFrame,
    *,
    remove_duplicates: bool = True,
    remove_invalid: bool = True,
    missing_strategy: str = "keep",
    start_date=None,
    end_date=None,
    routes: list[str] | None = None,
) -> dict:
    """Clean and aggregate through real Spark DataFrames in local execution mode."""
    from pyspark.sql import functions as F

    started = time.perf_counter()
    spark = get_spark_session()
    invalid_marker = "__transitiq_invalid_value"
    while invalid_marker in frame.columns:
        invalid_marker += "_"
    invalid_mask = invalid_transport_rows(frame)
    invalid_values_detected = int(invalid_mask.sum())
    safe_frame = frame.copy()
    safe_frame[invalid_marker] = invalid_mask.to_numpy()
    source = _spark_input(safe_frame, spark)
    input_rows = source.count()
    missing_values = sum(
        source.select(
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias("missing")
        ).first()["missing"]
        or 0
        for column in source.columns
    )
    duplicates = 0
    if remove_duplicates:
        subset = ["record_id"] if "record_id" in source.columns else source.columns
        before = source.count()
        source = source.dropDuplicates(subset)
        duplicates = before - source.count()

    invalid_rows_removed = 0
    if remove_invalid:
        invalid = F.col(invalid_marker)
        if "passenger_count" in source.columns:
            invalid = invalid | F.coalesce(
                F.col("passenger_count") < 0, F.lit(False)
            )
        for column in ("vehicle_capacity", "journey_duration", "distance_km"):
            if column in source.columns:
                numeric = F.col(column).cast("double")
                if column == "vehicle_capacity":
                    invalid = invalid | (numeric.isNotNull() & (numeric <= 0))
                else:
                    invalid = invalid | (numeric.isNotNull() & (numeric < 0))
        for column in ("service_date", "service_datetime"):
            if column in source.columns:
                invalid = invalid | (
                    F.col(column).isNotNull() & F.to_date(F.col(column)).isNull()
                )
                break
        invalid_rows_removed = source.filter(invalid).count()
        source = source.filter(~invalid)

    if missing_strategy == "drop_rows":
        before = source.count()
        source = source.dropna(how="any")
        missing_rows_removed = before - source.count()
    elif missing_strategy == "median_numeric":
        missing_rows_removed = 0
        for column in NUMERIC_FIELDS.intersection(source.columns):
            median_values = source.approxQuantile(column, [0.5], 0.01)
            if median_values:
                source = source.fillna({column: float(median_values[0])})
    elif missing_strategy == "keep":
        missing_rows_removed = 0
    else:
        raise ValueError("Unsupported missing-value strategy.")

    if start_date is not None and "service_date" in source.columns:
        source = source.filter(F.to_date("service_date") >= F.lit(str(start_date)))
    if end_date is not None and "service_date" in source.columns:
        source = source.filter(F.to_date("service_date") <= F.lit(str(end_date)))
    if routes and "route_id" in source.columns:
        source = source.filter(F.col("route_id").cast("string").isin(routes))
    source = source.drop(invalid_marker)

    output_rows = source.count()
    aggregate = _spark_aggregation(source)
    aggregation_preview = aggregate.limit(500).toPandas()
    return {
        "engine": "PySpark local[*]",
        "execution_mode": "Single-machine local Spark; not a distributed cluster.",
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "duplicates_detected_and_removed": int(duplicates),
        "missing_values_detected": int(missing_values),
        "invalid_rows_detected": invalid_values_detected,
        "invalid_rows_removed": int(invalid_rows_removed),
        "rows_removed_for_missing_values": int(missing_rows_removed),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "cleaned_spark_df": source,
        "aggregation_spark_df": aggregate,
        "aggregation_preview": aggregation_preview,
        "preview": source.limit(250).toPandas(),
    }


def process_with_pandas_fallback(
    frame: pd.DataFrame,
    *,
    remove_duplicates: bool = True,
    remove_invalid: bool = True,
    missing_strategy: str = "keep",
    start_date=None,
    end_date=None,
    routes: list[str] | None = None,
) -> dict:
    """Explicit small-file fallback when local Spark cannot start."""
    started = time.perf_counter()
    cleaned, summary = clean_transport_data(
        frame,
        remove_duplicates=remove_duplicates,
        remove_invalid=remove_invalid,
        missing_strategy=missing_strategy,
    )
    cleaned = filter_transport_data(
        cleaned, start_date=start_date, end_date=end_date, routes=routes
    )
    if "route_id" in cleaned.columns:
        aggregation = route_aggregation(cleaned)
    elif "stop_id" in cleaned.columns:
        aggregation = stop_aggregation(cleaned)
    else:
        aggregation = pd.DataFrame(
            {
                "observations": [len(cleaned)],
                "total_passengers": [
                    pd.to_numeric(
                        cleaned.get("passenger_count", pd.Series(dtype=float)),
                        errors="coerce",
                    ).sum()
                ],
            }
        )
    return {
        "engine": "Pandas fallback",
        "execution_mode": (
            "Single-machine Pandas processing. Spark did not start; this is not "
            "distributed processing."
        ),
        "input_rows": summary["input_rows"],
        "output_rows": len(cleaned),
        "duplicates_detected_and_removed": summary["duplicates_removed"],
        "missing_values_detected": summary["missing_values_before"],
        "invalid_rows_detected": summary["invalid_rows_detected"],
        "invalid_rows_removed": summary["invalid_rows_removed"],
        "rows_removed_for_missing_values": summary["rows_removed_for_missing_values"],
        "processing_seconds": round(time.perf_counter() - started, 3),
        "cleaned_pandas_df": cleaned,
        "aggregation_preview": aggregation.head(500),
        "preview": cleaned.head(250),
        "cleaning_summary": summary,
    }


def spark_parquet_zip_bytes(spark_frame) -> bytes:
    """Package a Spark Parquet dataset (one or more part files) for download."""
    with tempfile.TemporaryDirectory(prefix="transitiq_parquet_") as temp_dir:
        output_dir = Path(temp_dir) / "processed"
        spark_frame.write.mode("overwrite").parquet(str(output_dir))
        buffer = BytesIO()
        with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
            archive.writestr(
                "README.txt",
                "TransitIQ processed Parquet dataset. Extract this ZIP and read the "
                "included part-*.parquet files as one Parquet dataset.\n",
            )
            for path in output_dir.iterdir():
                if path.is_file() and (path.name.startswith("part-") or path.name == "_SUCCESS"):
                    archive.write(path, arcname=path.name)
        return buffer.getvalue()
