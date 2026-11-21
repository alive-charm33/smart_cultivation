"""
Fetches real climate data for a given lat/lon from NASA POWER
(https://power.larc.nasa.gov/), a free, no-API-key-required climate
dataset published by NASA specifically for agricultural applications
(their "AG" community parameter set).

This is a "trusted" source in the sense that matters for a project
report: it's a government space agency dataset, publicly documented,
used in published agricultural research worldwide — not a random
blog or scraped site.

We use the CLIMATOLOGY endpoint, which gives long-term monthly/annual
averages for a location (not today's weather) — this is the right
choice for a suitability check, since we care about "is this a place
where tomatoes generally grow well", not "what's the weather today".
"""

from typing import Dict

import requests

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# T2M = temperature at 2m (°C), RH2M = relative humidity at 2m (%),
# PRECTOTCORR = bias-corrected precipitation (mm/day)
PARAMETERS = ["T2M", "RH2M", "PRECTOTCORR"]

# Typical tomato growing-season length used to convert daily average
# rainfall into a season total (matches the "rainfall_mm" feature used
# by the suitability model, which represents a growing-season total).
GROWING_SEASON_DAYS = 90


def fetch_climatology(lat: float, lon: float) -> Dict:
    """Fetch long-term average temperature, humidity, and rainfall for a
    coordinate from NASA POWER.

    Raises requests.HTTPError if the API call fails (e.g. no network).
    """
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    props = data["properties"]["parameter"]

    # Each parameter comes back as {"JAN": v, "FEB": v, ..., "ANN": v}
    # "ANN" is the annual average — use that as the representative value.
    avg_temp_c = props["T2M"]["ANN"]
    avg_humidity_pct = props["RH2M"]["ANN"]
    avg_daily_rainfall_mm = props["PRECTOTCORR"]["ANN"]

    return {
        "temperature_c": round(avg_temp_c, 2),
        "humidity_pct": round(avg_humidity_pct, 2),
        # convert daily average rainfall into a growing-season total,
        # since that's the unit the suitability model expects
        "rainfall_mm": round(avg_daily_rainfall_mm * GROWING_SEASON_DAYS, 2),
    }


if __name__ == "__main__":
    # Example: Sangli, Maharashtra (approx coordinates)
    climate = fetch_climatology(lat=16.85, lon=74.57)
    print(climate)
