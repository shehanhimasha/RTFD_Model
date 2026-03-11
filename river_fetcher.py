"""
SOURCE 3 — ArcGIS Dashboard River Level Fetcher
Runs every 1 hour (via GitHub Actions schedule).
Queries the confirmed ArcGIS REST API for the Gin River (Gin Ganga) live
gauges layer and extracts water level + rainfall for Baddegama & Thawalama.
Saves results to data/river_data.json.

Dashboard: https://www.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38
Service:   https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/gauges_2_view/FeatureServer/0

Confirmed field names (verified live):
  gauge        — station name (e.g. "Baddegama", "Thawalama")
  water_level  — current water level in metres
  rain_fall    — rainfall in mm
  EditDate     — last update timestamp (epoch milliseconds)
  alertpull    — alert level threshold (m)
  minorpull    — minor flood threshold (m)
  majorpull    — major flood threshold (m)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

ARCGIS_SERVICE_URL = (
    "https://services3.arcgis.com/J7ZFXmR8rSmQ3FGf/arcgis/rest/services/"
    "gauges_2_view/FeatureServer/0/query"
)

OUTPUT_FILE = Path("data/river_data.json")
RIVER_NAME  = "Gin Ganga"

TARGET_STATIONS = {
    "baddegama": {"station_id": "BAD01", "canonical": "Baddegama"},
    "thawalama":  {"station_id": "THA01", "canonical": "Thawalama"},
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [ArcGIS] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── ArcGIS REST Query ─────────────────────────────────────────────────────────


def query_arcgis() -> list[dict] | None:
    """
    Query the gauges_2_view layer for all Gin Ganga features,
    ordered newest-first. Returns list of attribute dicts or None on error.
    """
    params = {
        "where": "basin = 'Gin Ganga'",
        "outFields": "gauge,water_level,rain_fall,EditDate,alertpull,minorpull,majorpull",
        "f": "json",
        "orderByFields": "EditDate DESC",
        "resultRecordCount": 50,
    }
    try:
        resp = requests.get(ARCGIS_SERVICE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("ArcGIS request failed: %s", exc)
        return None
    except ValueError as exc:
        log.error("ArcGIS JSON decode error: %s", exc)
        return None

    if "error" in data:
        log.error("ArcGIS returned error: %s", data["error"])
        return None

    features = data.get("features", [])
    log.info("ArcGIS returned %d Gin Ganga features.", len(features))
    return [f.get("attributes", {}) for f in features]


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_epoch_ms(val) -> str:
    """Convert epoch-milliseconds integer to ISO timestamp string."""
    try:
        dt = datetime.fromtimestamp(int(val) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def build_station_record(attrs: dict, station_info: dict) -> dict:
    """Convert confirmed ArcGIS attribute dict into the required schema."""
    water_level = attrs.get("water_level")
    rain_fall   = attrs.get("rain_fall")
    edit_date   = attrs.get("EditDate")

    return {
        "station_id":              station_info["station_id"],
        "station_name":            station_info["canonical"],
        "river_name":              RIVER_NAME,
        "current_water_level_m":   float(water_level) if water_level is not None else None,
        "rainfall_mm_per_hour":    float(rain_fall)   if rain_fall   is not None else None,
        "observed_at":             parse_epoch_ms(edit_date),
        # Bonus: flood thresholds from the layer itself
        "alert_level_m":           attrs.get("alertpull"),
        "minor_flood_level_m":     attrs.get("minorpull"),
        "major_flood_level_m":     attrs.get("majorpull"),
    }


def save_output(stations: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source":     "arcgis_gin_river",
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "stations":   stations,
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
    log.info("Saved %d station(s) to %s (Total snapshots: %d)", len(stations), OUTPUT_FILE, len(existing_records))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== ArcGIS River Fetcher starting ===")

    features = query_arcgis()
    if features is None:
        log.error("Query failed. Keeping existing river_data.json unchanged.")
        return

    if not features:
        log.warning("No Gin Ganga features returned. Writing empty-stations file.")
        save_output([])
        return

    # Collect the LATEST record per target station (features ordered DESC)
    seen: set[str] = set()
    station_records: list[dict] = []

    for attrs in features:
        gauge_name = str(attrs.get("gauge", "")).strip().lower()
        info = TARGET_STATIONS.get(gauge_name)
        if info is None:
            continue
        sid = info["station_id"]
        if sid in seen:
            continue          # already captured the latest reading
        seen.add(sid)
        record = build_station_record(attrs, info)
        station_records.append(record)
        log.info(
            "%s — water_level=%.2f m, rain_fall=%.1f mm, observed_at=%s",
            info["canonical"],
            record["current_water_level_m"] or 0.0,
            record["rainfall_mm_per_hour"]  or 0.0,
            record["observed_at"],
        )

    if not station_records:
        log.warning("Target stations not found in the %d returned features.", len(features))

    save_output(station_records)
    log.info("Done. Captured %d / %d target station(s).", len(station_records), len(TARGET_STATIONS))


if __name__ == "__main__":
    main()
