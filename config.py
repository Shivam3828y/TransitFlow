from pathlib import Path

APP_TITLE = "TransitIQ"
APP_DESCRIPTION = "Public transport demand analytics and forecasting"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_DATA_PATH = DATA_DIR / "sample_transport_data.csv"

MAX_PREVIEW_ROWS = 250
MAX_CHART_ROWS = 100_000
MAX_SPARK_FALLBACK_ROWS = 100_000
MAX_PANDAS_COLLECT_ROWS = 200_000

CANONICAL_FIELDS = [
    "record_id",
    "service_date",
    "service_time",
    "service_datetime",
    "route_id",
    "stop_id",
    "passenger_count",
    "vehicle_capacity",
    "journey_duration",
    "distance_km",
    "latitude",
    "longitude",
]

FIELD_LABELS = {
    "record_id": "Record ID",
    "service_date": "Service date",
    "service_time": "Service time",
    "service_datetime": "Service date and time",
    "route_id": "Route ID",
    "stop_id": "Stop ID",
    "passenger_count": "Passenger count",
    "vehicle_capacity": "Vehicle capacity",
    "journey_duration": "Journey duration",
    "distance_km": "Distance (km)",
    "latitude": "Latitude (optional)",
    "longitude": "Longitude (optional)",
}

FIELD_ALIASES = {
    "record_id": {"record_id", "recordid", "id", "observation_id", "trip_id"},
    "service_date": {"service_date", "date", "journey_date", "travel_date", "day"},
    "service_time": {"service_time", "time", "departure_time", "hour"},
    "service_datetime": {
        "service_datetime",
        "datetime",
        "timestamp",
        "service_timestamp",
        "datetime_local",
    },
    "route_id": {"route_id", "route", "line", "line_id", "route_number"},
    "stop_id": {"stop_id", "stop", "station", "station_id", "stop_name"},
    "passenger_count": {
        "passenger_count",
        "passengers",
        "ridership",
        "boardings",
        "demand",
        "count",
    },
    "vehicle_capacity": {"vehicle_capacity", "capacity", "seats", "vehicle_seats"},
    "journey_duration": {
        "journey_duration",
        "duration",
        "duration_minutes",
        "travel_minutes",
    },
    "distance_km": {"distance_km", "distance", "distance_in_km", "km"},
    "latitude": {"latitude", "lat"},
    "longitude": {"longitude", "lon", "lng"},
}

NUMERIC_FIELDS = {
    "passenger_count",
    "vehicle_capacity",
    "journey_duration",
    "distance_km",
}
