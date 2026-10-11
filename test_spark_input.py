import os
import traceback
from pathlib import Path

import pandas as pd

from src.spark_processor import get_spark_session, _spark_input

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

print("TransitIQ Spark Diagnostic")
print("-" * 40)

# Find CSV files in the project directory.
project_dir = Path(__file__).resolve().parent

csv_files = [
    path
    for path in project_dir.rglob("*.csv")
    if ".venv" not in str(path)
    and ".venv311" not in str(path)
    and "site-packages" not in str(path)
]

if not csv_files:
    print("No CSV files found inside the project directory.")
    print("Please provide the path to your uploaded CSV file.")
    raise SystemExit(1)

print("\nAvailable CSV files:")

for index, path in enumerate(csv_files):
    print(f"{index}: {path.relative_to(project_dir)}")

try:
    choice = int(input("\nEnter the number of your transport dataset: "))
    csv_path = csv_files[choice]

    print("\nSTEP 1: Reading CSV...")
    data = pd.read_csv(csv_path)

    print("File:", csv_path.name)
    print("Dataset shape:", data.shape)
    print("Columns:", data.columns.tolist())

    print("\nSTEP 2: Starting Spark...")
    spark = get_spark_session()

    print("Spark version:", spark.version)

    print("\nSTEP 3: Converting Pandas DataFrame to Spark...")
    spark_df = _spark_input(data, spark)

    print("\nSUCCESS: Conversion completed.")
    spark_df.printSchema()

    print("\nSTEP 4: Testing Spark execution...")
    print("Row count:", spark_df.count())
    spark_df.show(5, truncate=False)

    print("\nDIAGNOSTIC RESULT: PASSED")

except Exception:
    print("\nDIAGNOSTIC RESULT: FAILED")
    traceback.print_exc()

finally:
    print("\nDiagnostic finished.")