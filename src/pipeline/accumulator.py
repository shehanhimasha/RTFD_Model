# =============================================================================
#   Manages the daily accumulator — a running tally of today's hourly
#   readings per station.
#
# Problem it solves:
#   The model was trained on DAILY aggregates (w_avg, w_max, w_min,
#   rainfall_mm for the full day). Real-time data gives one reading
#   per hour. This file accumulates those hourly readings and computes
#   running daily stats as the day progresses.
#
# How it works:
#   Every hour when pipeline.py runs:
#     1. Load daily_accumulator.json (today's readings so far)
#     2. Add the new hourly reading
#     3. Recompute w_avg, w_max, w_min, rainfall_sum
#     4. Save back to daily_accumulator.json
#     5. At midnight — reset to empty for the new day
#
# Storage:
#   data/live/daily_accumulator.json
#   Format:
#   {
#     "date": "2026-04-05",
#     "BAD01": {
#       "readings": [1.22, 1.25, 1.30, ...],
#       "rainfall_readings": [0.0, 0.2, 0.0, ...],
#       "w_avg": 1.257,
#       "w_max": 1.30,
#       "w_min": 1.22,
#       "rainfall_mm": 0.2,
#       "rising_flag": 0
#     },
#     "THA01": { ... }
#   }
# =============================================================================

import json
from datetime import datetime, date
from pathlib import Path
from config.settings import paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Storage path ──────────────────────────────────────────────────────────────
LIVE_DIR         = paths.PROJECT_ROOT / "data" / "live"
ACCUMULATOR_PATH = LIVE_DIR / "daily_accumulator.json"

# ── Target stations ───────────────────────────────────────────────────────────
TARGET_STATIONS = ['BAD01', 'THA01']


# =============================================================================
# Load or initialise accumulator
# =============================================================================

def load_accumulator() -> dict:
    """
    Load the daily accumulator from disk.
    If it doesn't exist or is from a previous day, return a fresh empty one.

    Returns:
        dict with today's accumulated readings per station
    """
    today = str(date.today())

    if ACCUMULATOR_PATH.exists():
        with open(ACCUMULATOR_PATH, 'r') as f:
            acc = json.load(f)

        # If accumulator is from a previous day, reset it
        if acc.get('date') != today:
            logger.info(f"New day detected ({today}) — resetting accumulator")
            acc = _empty_accumulator(today)
    else:
        logger.info("No accumulator found — creating fresh one")
        acc = _empty_accumulator(today)

    return acc


def _empty_accumulator(today: str) -> dict:
    """Create a fresh empty accumulator for a new day."""
    acc = {'date': today}
    for station in TARGET_STATIONS:
        acc[station] = {
            'readings':          [],   # list of hourly water level values
            'rainfall_readings': [],   # list of hourly rainfall values
            'w_avg':             0.0,
            'w_max':             0.0,
            'w_min':             0.0,
            'rainfall_mm':       0.0,
            'rising_flag':       0,
        }
    return acc


# =============================================================================
# Add a new hourly reading
# =============================================================================

def add_reading(
    acc: dict,
    station_id: str,
    water_level: float,
    rainfall_mm: float,
    rising_flag: int = 0,
) -> dict:
    """
    Add one hourly reading to the accumulator and recompute daily stats.

    Called once per station per pipeline run with the latest ArcGIS
    water level and OWM rainfall for that station.

    Args:
        acc:          current accumulator dict (from load_accumulator)
        station_id:   'BAD01' or 'THA01'
        water_level:  current_water_level_m from ArcGIS
        rainfall_mm:  rainfall_1h_mm from OWM for this station
        rising_flag:  1 if DMC says rising, 0 otherwise

    Returns:
        Updated accumulator dict
    """
    if station_id not in acc:
        logger.warning(f"Station {station_id} not in accumulator — skipping")
        return acc

    s = acc[station_id]

    # Append new reading
    if water_level > 0:
        s['readings'].append(float(water_level))

    if rainfall_mm >= 0:
        s['rainfall_readings'].append(float(rainfall_mm))

    # Recompute running daily stats
    if s['readings']:
        s['w_avg'] = round(sum(s['readings']) / len(s['readings']), 4)
        s['w_max'] = round(max(s['readings']), 4)
        s['w_min'] = round(min(s['readings']), 4)

    # Total rainfall accumulated since midnight
    s['rainfall_mm'] = round(sum(s['rainfall_readings']), 2)

    # Update rising flag — if DMC says rising at any point today, flag=1
    # We keep the highest flag seen today (once rising, stays rising until reset)
    s['rising_flag'] = max(s['rising_flag'], int(rising_flag))

    acc[station_id] = s

    logger.info(
        f"  {station_id} accumulator updated: "
        f"w_avg={s['w_avg']:.3f}m  "
        f"w_max={s['w_max']:.3f}m  "
        f"w_min={s['w_min']:.3f}m  "
        f"rain={s['rainfall_mm']:.1f}mm  "
        f"readings={len(s['readings'])}"
    )

    return acc


# =============================================================================
# Get current daily stats for a station
# =============================================================================

def get_station_stats(acc: dict, station_id: str) -> dict:
    """
    Get the current running daily stats for a station.

    Returns:
        dict with w_avg, w_max, w_min, rainfall_mm, rising_flag
        Returns zeros if station has no readings yet today
    """
    if station_id not in acc or not acc[station_id]['readings']:
        logger.warning(
            f"No readings yet for {station_id} today — returning zeros"
        )
        return {
            'w_avg':       0.0,
            'w_max':       0.0,
            'w_min':       0.0,
            'rainfall_mm': 0.0,
            'rising_flag': 0,
        }

    s = acc[station_id]
    return {
        'w_avg':       s['w_avg'],
        'w_max':       s['w_max'],
        'w_min':       s['w_min'],
        'rainfall_mm': s['rainfall_mm'],
        'rising_flag': s['rising_flag'],
    }


# =============================================================================
# Save accumulator
# =============================================================================

def save_accumulator(acc: dict) -> None:
    """Save the updated accumulator to disk."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACCUMULATOR_PATH, 'w') as f:
        json.dump(acc, f, indent=2)


# =============================================================================
# Parse rising flag from DMC data
# =============================================================================

def parse_rising_flag(rising_or_falling: str) -> int:
    """
    Convert DMC rising_or_falling string to binary flag.

    DMC values seen in your data: 'Rising', 'Falling', null, ''
    We flag as rising only when explicitly 'Rising'.

    Args:
        rising_or_falling: string from DMC JSON field

    Returns:
        1 if Rising, 0 for anything else
    """
    if rising_or_falling and str(rising_or_falling).strip().lower() == 'rising':
        return 1
    return 0


# =============================================================================
# Get upstream rainfall from OWM data
# =============================================================================

def extract_upstream_rainfall(weather_data: dict) -> dict:
    """
    Extract today's hourly rainfall for all 6 stations from OWM data.
    Returns a dict of station_id → rainfall_1h_mm.

    Args:
        weather_data: the loaded weather_data.json dict

    Returns:
        dict like {'BAD01': 0.0, 'DEN01': 5.2, ...}
    """
    rainfall = {}

    locations = weather_data.get('locations', [])
    for loc in locations:
        station_id  = loc.get('station_id')
        rain_1h     = loc.get('rainfall_1h_mm', 0.0)
        if station_id:
            rainfall[station_id] = float(rain_1h) if rain_1h else 0.0

    return rainfall