"""
SOURCE 3 — ArcGIS Dashboard River Level Fetcher
Runs every 1 hour (via GitHub Actions schedule).
Queries the ArcGIS REST API backing the Gin River (Gin Ganga) live dashboard
and extracts water level + rainfall for Baddegama & Thawalama.
Saves results to data/river_data.json.

Dashboard: https://www.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38

NOTE: The feature service URL below was identified from the dashboard's
      underlying layers. If readings come back empty, open the dashboard
      in a browser, go to DevTools → Network, filter for "FeatureServer",
      and update ARCGIS_SERVICE_URL with the correct endpoint.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

# Known ArcGIS feature service for Sri Lanka river gauging stations
# (Gin Ganga / Baddegama station layer from NBRO/irrigation dept)
ARCGIS_SERVICE_URL = (
    "https://services6.arcgis.com/t6lYS2Pmd8iVx1ux/arcgis/rest/services/"
    "Gin_River_Monitoring/FeatureServer/0/query"
)

# Fallback attempt — query the general public flood dashboard service
ARCGIS_ALT_URL = (
    "https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/"
    "ncov_cases_US/FeatureServer/0/query"  # placeholder; see NOTE above
)

TARGET_STATIONS = {
    "baddegama": {"station_id": "BAD01", "canonical": "Baddegama"},
    "thawalama":  {"station_id": "THA01", "canonical": "Thawalama"},
}

OUTPUT_FILE = Path("data/river_data.json")
RIVER_NAME = "Gin Ganga"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [ArcGIS] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── ArcGIS REST Query ─────────────────────────────────────────────────────────


def query_arcgis(service_url: str) -> list[dict] | None:
    """
    Query the ArcGIS feature layer for all features and return the list
    of attribute dicts, or None on error.
    """
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "json",
        "resultRecordCount": 200,
        "orderByFields": "OBJECTID DESC",
    }
    try:
        resp = requests.get(service_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("ArcGIS request failed (%s): %s", service_url, exc)
        return None
    except ValueError as exc:
        log.error("ArcGIS JSON decode error: %s", exc)
        return None

    if "error" in data:
        log.error("ArcGIS returned error: %s", data["error"])
        return None

    features = data.get("features", [])
    return [f.get("attributes", {}) for f in features]


def match_station(attrs: dict) -> dict | None:
    """
    Inspect attribute keys to find a station-name field,
    then return TARGET_STATIONS entry if it matches, else None.
    """
    # Common field names for station name in NBRO / irrigation ArcGIS layers
    name_keys = ["STATION", "Station", "station_name", "StationName", "NAME", "Name",
                 "GAUGE_STATION", "gauge_station", "StaName", "sta_name"]
    raw_name = ""
    for key in name_keys:
        val = attrs.get(key)
        if val:
            raw_name = str(val).strip()
            break

    if not raw_name:
        return None

    lower = raw_name.lower()
    for keyword, info in TARGET_STATIONS.items():
        if keyword in lower:
            return info
    return None


def parse_timestamp(attrs: dict) -> str:
    """
    Try to extract an observation timestamp from attribute fields.
    Returns ISO-format string; falls back to current UTC time.
    """
    ts_keys = ["DateTime", "datetime", "DATETIME", "Timestamp", "timestamp",
               "ObsTime", "obs_time", "DATE_TIME", "RecordTime", "record_time"]
    for key in ts_keys:
        val = attrs.get(key)
        if val is not None:
            # ArcGIS often stores epoch ms as integers
            try:
                epoch_ms = int(val)
                dt = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except (ValueError, TypeError, OSError):
                pass
            # Try string parsing
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
                try:
                    dt = datetime.strptime(str(val), fmt)
                    return dt.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    pass
    # Fallback: current UTC
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def extract_float(attrs: dict, *keys) -> float | None:
    """Try each key in turn and return the first parsable float."""
    for key in keys:
        val = attrs.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


def build_station_record(attrs: dict, station_info: dict) -> dict:
    """Convert raw ArcGIS attributes into the required schema."""
    # Water level field candidates
    water_level = extract_float(
        attrs,
        "WaterLevel", "water_level", "WATER_LEVEL", "Level", "level",
        "CurrentLevel", "current_level", "WL", "wl",
    )
    # Rainfall candidates
    rainfall = extract_float(
        attrs,
        "Rainfall", "rainfall", "RAINFALL", "Rain", "rain",
        "Rainfall_mm", "rainfall_mm", "RainMM", "rain_mm",
    )
    observed_at = parse_timestamp(attrs)

    return {
        "station_id": station_info["station_id"],
        "station_name": station_info["canonical"],
        "river_name": RIVER_NAME,
        "current_water_level_m": water_level,
        "rainfall_mm_per_hour": rainfall,
        "observed_at": observed_at,
    }


def save_output(stations: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "arcgis_gin_river",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "stations": stations,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Saved %d station(s) to %s", len(stations), OUTPUT_FILE)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== ArcGIS River Fetcher starting ===")

    features = query_arcgis(ARCGIS_SERVICE_URL)

    if features is None:
        log.error("Primary ArcGIS query failed. Exiting without changes.")
        return

    if not features:
        log.warning("ArcGIS returned 0 features from %s.", ARCGIS_SERVICE_URL)
        log.warning(
            "The feature service URL may need to be updated. "
            "See the NOTE at the top of river_fetcher.py."
        )
        # Write an empty-stations placeholder so the pipeline can still run
        save_output([])
        return

    # Collect records for target stations (latest per station)
    seen_stations: set[str] = set()
    station_records: list[dict] = []

    for attrs in features:
        info = match_station(attrs)
        if info is None:
            continue
        sid = info["station_id"]
        if sid in seen_stations:
            continue  # features are ordered DESC — first appearance is latest
        seen_stations.add(sid)
        record = build_station_record(attrs, info)
        station_records.append(record)

    if not station_records:
        log.warning(
            "No target-station records found in %d features. "
            "Check ARCGIS_SERVICE_URL and field names.",
            len(features),
        )

    save_output(station_records)
    log.info("Done. Found %d target station(s).", len(station_records))


if __name__ == "__main__":
    main()
