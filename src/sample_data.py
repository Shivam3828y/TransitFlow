from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

ROUTES = {
    "R01": {"demand": 1.00, "capacity": 62, "length_km": 18.4},
    "R02": {"demand": 0.84, "capacity": 55, "length_km": 14.1},
    "R03": {"demand": 1.21, "capacity": 72, "length_km": 22.8},
    "R04": {"demand": 0.72, "capacity": 52, "length_km": 11.6},
    "R05": {"demand": 1.08, "capacity": 68, "length_km": 20.2},
    "R06": {"demand": 0.91, "capacity": 58, "length_km": 16.7},
    "R07": {"demand": 1.34, "capacity": 78, "length_km": 26.3},
    "R08": {"demand": 0.78, "capacity": 55, "length_km": 13.9},
}
SERVICE_TIMES = ["05:30:00", "08:00:00", "12:30:00", "17:30:00", "21:00:00"]
TIME_FACTORS = [0.48, 1.72, 0.94, 1.92, 0.52]
STOPS_PER_ROUTE = 5


def generate_sample_data(
    *,
    start_date: date = date(2025, 1, 1),
    days: int = 90,
    seed: int = 241,
) -> pd.DataFrame:
    """Create reproducible synthetic service observations with daily/hourly patterns."""
    if days < 8:
        raise ValueError("Use at least eight days to demonstrate lag features.")
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    record_number = 1
    for day_offset in range(days):
        service_day = start_date + timedelta(days=day_offset)
        weekend_factor = 0.76 if service_day.weekday() >= 5 else 1.0
        weekly_wave = 1 + 0.07 * np.sin(day_offset * 2 * np.pi / 7)
        trend = 1 + day_offset * 0.001
        for route_index, (route_id, route) in enumerate(ROUTES.items()):
            route_weekend = 1.12 if service_day.weekday() >= 5 and route_index % 3 == 0 else 1.0
            for stop_index in range(STOPS_PER_ROUTE):
                stop_factor = [0.80, 1.02, 1.24, 0.94, 0.69][stop_index]
                distance = route["length_km"] * (stop_index + 1) / (
                    STOPS_PER_ROUTE + 1
                )
                for time_index, service_time in enumerate(SERVICE_TIMES):
                    event_factor = 1.0
                    if service_day.day % 17 == 0 and time_index in {1, 3}:
                        event_factor = 1.28
                    expected = (
                        20
                        * route["demand"]
                        * stop_factor
                        * TIME_FACTORS[time_index]
                        * weekend_factor
                        * route_weekend
                        * weekly_wave
                        * trend
                        * event_factor
                    )
                    passengers = int(rng.poisson(max(expected, 0.5)))
                    duration = max(
                        5.0,
                        (7.5 + distance * 1.35 + time_index * 1.2)
                        * rng.normal(1.0, 0.09),
                    )
                    records.append(
                        {
                            "record_id": f"SYN-{record_number:07d}",
                            "service_date": service_day.isoformat(),
                            "service_time": service_time,
                            "route_id": route_id,
                            "stop_id": f"{route_id}-S{stop_index + 1:02d}",
                            "passenger_count": passengers,
                            "vehicle_capacity": route["capacity"],
                            "journey_duration": round(float(duration), 1),
                            "distance_km": round(float(distance), 2),
                        }
                    )
                    record_number += 1
    return pd.DataFrame.from_records(records)
