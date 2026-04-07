# =============================================================================
#   Manages a rolling 7-day window of completed daily values per station.
#   Computes all lag and rolling average features that the model needs
#   but cannot get from a single live reading.
#
# Problem it solves:
#   Features like rainfall_mm_lag_1, w_avg_lag_2, rainfall_DEN01_lag_1
#   require knowing what happened on previous days.
#   This store keeps those previous days and computes the features.
#
# How it works:
#   At the END of each day (midnight), pipeline.py calls update_history()
#   with today's completed daily values. The store appends the new day
#   and drops anything older than 7 days.
#
#   During each hourly run, pipeline.py calls get_lag_features() to
#   retrieve all lag and rolling features for the current prediction.
#
# Storage:
#   data/live/history_store.json
#   Format:
#   {
#     "BAD01": [
#       {"date": "2026-03-29", "w_avg": 1.1, "w_max": 1.3, "w_min": 0.9,
#        "rainfall_mm": 0.0, "rainfall_DEN01": 0.0, ...},
#       {"date": "2026-03-30", ...},
#       ...   (up to 7 entries)
#     ],
#     "THA01": [ ... ]
#   }
# =============================================================================

import json
from datetime import date, timedelta
from config.settings import paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Storage path ──────────────────────────────────────────────────────────────
LIVE_DIR          = paths.PROJECT_ROOT / "data" / "live"
HISTORY_PATH      = LIVE_DIR / "history_store.json"

# ── How many days to keep ─────────────────────────────────────────────────────
# Must be at least max lag (3) + 1. We keep 7 for rolling_7 features.
MAX_HISTORY_DAYS  = 7

TARGET_STATIONS   = ['BAD01', 'THA01']

# ── Upstream stations whose rainfall we track ─────────────────────────────────
UPSTREAM_STATIONS = ['DEN01', 'LAN01', 'THA01', 'UDU01', 'MAK01']


# =============================================================================
# Load history store
# =============================================================================

def load_history() -> dict:
    """
    Load the history store from disk.
    If it doesn't exist, return an empty store.

    Returns:
        dict with up to 7 days of daily records per station
    """
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)
        logger.info(
            f"History loaded: "
            f"BAD01={len(history.get('BAD01', []))} days, "
            f"THA01={len(history.get('THA01', []))} days"
        )
    else:
        logger.warning("No history store found — starting empty")
        logger.warning(
            "First 7 days of predictions will use zeros for lag features. "
            "This is expected on first deployment."
        )
        history = {station: [] for station in TARGET_STATIONS}

    return history


# =============================================================================
# Update history with today's completed daily values
# =============================================================================

def update_history(
    history: dict,
    station_id: str,
    daily_values: dict,
) -> dict:
    """
    Append today's completed daily values to the history store.
    Drop anything older than MAX_HISTORY_DAYS.

    Call this once per station at end of day (midnight).

    Args:
        history:      current history dict
        station_id:   'BAD01' or 'THA01'
        daily_values: dict with today's completed daily stats:
            {
              'date':         '2026-04-05',
              'w_avg':        1.25,
              'w_max':        1.40,
              'w_min':        1.10,
              'rainfall_mm':  12.5,
              'rainfall_DEN01': 8.0,
              'rainfall_LAN01': 5.5,
              'rainfall_THA01': 10.2,
              'rainfall_UDU01': 3.1,
              'rainfall_MAK01': 2.8,
            }

    Returns:
        Updated history dict
    """
    if station_id not in history:
        history[station_id] = []

    # Append today's record
    history[station_id].append(daily_values)

    # Sort by date to ensure chronological order
    history[station_id].sort(key=lambda x: x['date'])

    # Keep only the last MAX_HISTORY_DAYS
    if len(history[station_id]) > MAX_HISTORY_DAYS:
        history[station_id] = history[station_id][-MAX_HISTORY_DAYS:]

    logger.info(
        f"History updated for {station_id}: "
        f"{len(history[station_id])} days stored "
        f"(latest: {daily_values['date']})"
    )

    return history


# =============================================================================
# Compute lag and rolling features from history
# =============================================================================

def get_lag_features(history: dict, station_id: str) -> dict:
    """
    Compute all lag and rolling features for a station from the history store.

    Features returned match exactly what feature_engineering.py created
    during training. Column names and computation logic must be identical.

    Args:
        history:    loaded history dict
        station_id: 'BAD01' or 'THA01'

    Returns:
        dict with all lag and rolling features for this station.
        Missing history days are filled with 0 (safe default).
    """
    records = history.get(station_id, [])

    # Helper — get a field value from N days ago (1 = yesterday)
    # Returns 0.0 if that day is not in history yet
    def get_lag(field: str, lag: int) -> float:
        idx = len(records) - lag
        if idx < 0 or idx >= len(records):
            return 0.0
        return float(records[idx].get(field, 0.0))

    # Helper — rolling mean over last N days
    def rolling_mean(field: str, window: int) -> float:
        values = [
            float(r.get(field, 0.0))
            for r in records[-window:]
        ]
        return round(sum(values) / len(values), 4) if values else 0.0

    # ── Own station lag features ──
    features = {

        # Rainfall lags
        'rainfall_mm_lag_1': get_lag('rainfall_mm', 1),
        'rainfall_mm_lag_2': get_lag('rainfall_mm', 2),
        'rainfall_mm_lag_3': get_lag('rainfall_mm', 3),

        # Water level lags
        'w_avg_lag_1':       get_lag('w_avg', 1),
        'w_avg_lag_2':       get_lag('w_avg', 2),
        'w_max_lag_1':       get_lag('w_max', 1),

        # Rolling averages
        'rainfall_mm_rolling_3': rolling_mean('rainfall_mm', 3),
        'rainfall_mm_rolling_7': rolling_mean('rainfall_mm', 7),
        'w_avg_rolling_3':       rolling_mean('w_avg', 3),

        # Delta — today minus yesterday
        # w_avg_delta is computed in pipeline.py using today's live w_avg
        # and yesterday's w_avg_lag_1 from here
        # rainfall_delta same pattern — done in pipeline.py

        # Upstream lag features
        'rainfall_DEN01_lag_1': get_lag('rainfall_DEN01', 1),
        'rainfall_DEN01_lag_2': get_lag('rainfall_DEN01', 2),
        'rainfall_LAN01_lag_1': get_lag('rainfall_LAN01', 1),
        'rainfall_LAN01_lag_2': get_lag('rainfall_LAN01', 2),
        'rainfall_THA01_lag_1': get_lag('rainfall_THA01', 1),
        'rainfall_THA01_lag_2': get_lag('rainfall_THA01', 2),
        'rainfall_UDU01_lag_1': get_lag('rainfall_UDU01', 1),
        'rainfall_UDU01_lag_2': get_lag('rainfall_UDU01', 2),
        'rainfall_MAK01_lag_1': get_lag('rainfall_MAK01', 1),
        'rainfall_MAK01_lag_2': get_lag('rainfall_MAK01', 2),
    }

    return features


# =============================================================================
# Build a completed daily record for history update
# =============================================================================

def build_daily_record(
    date_str: str,
    station_id: str,
    w_avg: float,
    w_max: float,
    w_min: float,
    rainfall_mm: float,
    upstream_rainfall: dict,
) -> dict:
    """
    Build a complete daily record ready to append to history.

    Called at end of each day with the completed daily values
    from the accumulator plus upstream rainfall from OWM.

    Args:
        date_str:          'YYYY-MM-DD'
        station_id:        'BAD01' or 'THA01'
        w_avg/max/min:     today's completed water level stats
        rainfall_mm:       today's total rainfall at this station
        upstream_rainfall: dict of all 6 stations' daily rainfall totals
                           {'DEN01': 12.5, 'LAN01': 8.0, ...}

    Returns:
        dict ready to pass to update_history()
    """
    record = {
        'date':        date_str,
        'w_avg':       round(w_avg,        4),
        'w_max':       round(w_max,        4),
        'w_min':       round(w_min,        4),
        'rainfall_mm': round(rainfall_mm,  2),
    }

    # Add upstream rainfall — use 0.0 if station not in upstream_rainfall
    for upstream in UPSTREAM_STATIONS:
        record[f'rainfall_{upstream}'] = round(
            float(upstream_rainfall.get(upstream, 0.0)), 2
        )

    return record


# =============================================================================
# Save history store
# =============================================================================

def save_history(history: dict) -> None:
    """Save the updated history store to disk."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved → {HISTORY_PATH}")