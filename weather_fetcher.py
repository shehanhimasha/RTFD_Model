"""
SOURCE 2 — OpenWeatherMap Live Rainfall Fetcher
Runs every 1 hour (via GitHub Actions schedule).
Fetches current weather for 6 Gin Ganga basin stations and saves
results to data/weather_data.json.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

# Load .env for local development (no-op in GitHub Actions where secrets are injected)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
OUTPUT_FILE = Path("data/weather_data.json")

STATIONS = [
    {"station_id": "BAD01", "station_name": "Baddegama", "lat": 6.17, "lon": 80.18},
    {"station_id": "THA01", "station_name": "Thawalama",  "lat": 6.34, "lon": 80.33},
    {"station_id": "DEN01", "station_name": "Deniyaya",   "lat": 6.35, "lon": 80.56},
    {"station_id": "LAN01", "station_name": "Lankagama",  "lat": 6.37, "lon": 80.47},
    {"station_id": "MAK01", "station_name": "Makurugoda", "lat": 6.18, "lon": 80.17},
    {"station_id": "UDU01", "station_name": "Udugama",    "lat": 6.23, "lon": 80.32},
]

RIVER_NAME = "Gin Ganga"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [OWM] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def fetch_station(station: dict) -> dict | None:
    """
    Fetch current weather for one station.
    Returns a location dict on success, None on failure.
    """
    params = {
        "lat": station["lat"],
        "lon": station["lon"],
        "appid": API_KEY,
        "units": "metric",
    }
    try:
        resp = requests.get(OWM_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("Failed to fetch %s (%s): %s", station["station_name"], station["station_id"], exc)
        return None
    except ValueError as exc:
        log.error("JSON decode error for %s: %s", station["station_name"], exc)
        return None

    # Extract rainfall — OWM only returns 'rain' key when it's actually raining
    rain_block = data.get("rain", {})
    rainfall_1h_mm = rain_block.get("1h", 0.0) if isinstance(rain_block, dict) else 0.0

    weather_list = data.get("weather", [{}])
    weather_desc = weather_list[0].get("description", "") if weather_list else ""

    return {
        "station_id": station["station_id"],
        "station_name": station["station_name"],
        "river_name": RIVER_NAME,
        "lat": station["lat"],
        "lon": station["lon"],
        "rainfall_1h_mm": float(rainfall_1h_mm),
        "humidity": data.get("main", {}).get("humidity"),
        "clouds_pct": data.get("clouds", {}).get("all"),
        "weather_description": weather_desc,
    }


def save_output(locations: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "openweathermap",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "locations": locations,
    }
    
    # Load existing records
    existing_records = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    existing_records = data
                elif isinstance(data, dict):
                    # Migrate old single-object dict to list
                    existing_records = [data]
        except Exception as exc:
            log.warning("Could not read existing %s: %s", OUTPUT_FILE, exc)

    existing_records.append(payload)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing_records, fh, indent=2, ensure_ascii=False)
    log.info("Saved %d station(s) to %s (Total snapshots: %d)", len(locations), OUTPUT_FILE, len(existing_records))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== OpenWeatherMap Fetcher starting ===")

    if not API_KEY:
        log.error("OPENWEATHER_API_KEY environment variable is not set. Aborting.")
        return

    locations = []
    for station in STATIONS:
        result = fetch_station(station)
        if result is not None:
            locations.append(result)
        else:
            log.warning("Skipping %s due to fetch failure.", station["station_name"])

    if not locations:
        log.error("All station fetches failed. Not overwriting existing weather_data.json.")
        return

    save_output(locations)
    log.info("Done. Fetched %d/%d stations.", len(locations), len(STATIONS))


if __name__ == "__main__":
    main()
