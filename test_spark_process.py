import os
import traceback
from pathlib import Path

import pandas as pd

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = (
    r"D:\TransitIQ_Clean\.venv311\Scripts\python.exe"
)
os.environ["PYSPARK_DRIVER_PYTHON"] = (
    r"D:\TransitIQ_Clean\.venv311\Scripts\python.exe"
)
os.environ["JAVA_HOME"] = (
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
)
os.environ["PATH"] = (
    os.path.join(os.environ["JAVA_HOME"], "bin")
    + os.pathsep
    + os.environ["PATH"]
)

from src.spark_processor import process_with_spark

csv_path = (
    Path(__file__).resolve().parent
    / "data"
    / "sample_transport_data.csv"
)

try:
    print("STEP 1: Loading dataset...")
    data = pd.read_csv(csv_path)
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nSTEP 2: Running complete Spark pipeline...")

    result = process_with_spark(
        data,
        remove_duplicates=True,
        remove_invalid=True,
        missing_strategy="keep",
    )

    print("\nSUCCESS: Complete Spark processing finished.")
    print("Engine:", result["engine"])
    print("Input rows:", result["input_rows"])
    print("Output rows:", result["output_rows"])
    print("Processing seconds:", result["processing_seconds"])

    print("\nAggregation preview:")
    print(result["aggregation_preview"].head())

    print("\nDIAGNOSTIC RESULT: PASSED")

except Exception:
    print("\nDIAGNOSTIC RESULT: FAILED")
    traceback.print_exc()