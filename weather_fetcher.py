"""
SOURCE 2 — Open-Meteo Live Rainfall Fetcher
Runs every 1 hour (via GitHub Actions schedule).
Fetches current weather for 6 Gin Ganga basin stations and saves
results to data/weather_data.json.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

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
    level=logging.INFO, format="%(asctime)s [OPEN-METEO] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def fetch_station(station: dict) -> dict | None:
    """
    Fetch current weather for one station using Open-Meteo.
    Returns a location dict on success, None on failure.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "current": "precipitation,relative_humidity_2m,cloud_cover,weather_code",
        "timezone": "Asia/Colombo",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("Failed to fetch %s (%s): %s", station["station_name"], station["station_id"], exc)
        return None
    except ValueError as exc:
        log.error("JSON decode error for %s: %s", station["station_name"], exc)
        return None

    current = data.get("current", {})
    
    # Extract data, default to 0.0 or None if missing
    rainfall_1h_mm = current.get("precipitation", 0.0)
    humidity = current.get("relative_humidity_2m")
    clouds_pct = current.get("cloud_cover")
    weather_code = current.get("weather_code")

    # Map WMO weather code to description
    # Mapping based on Open-Meteo WMO code documentation
    weather_desc = f"WMO Code {weather_code}" if weather_code is not None else ""

    return {
        "station_id": station["station_id"],
        "station_name": station["station_name"],
        "river_name": RIVER_NAME,
        "lat": station["lat"],
        "lon": station["lon"],
        "rainfall_1h_mm": float(rainfall_1h_mm) if rainfall_1h_mm is not None else 0.0,
        "humidity": humidity,
        "clouds_pct": clouds_pct,
        "weather_description": weather_desc,
    }


def save_output(locations: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "open-meteo",
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
    log.info("=== Open-Meteo Fetcher starting ===")

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
