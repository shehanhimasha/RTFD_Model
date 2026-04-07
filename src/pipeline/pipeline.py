# =============================================================================
#   Main orchestrator — runs every hour via GitHub Actions.
#   Reads the three live JSON files, assembles 36 features per station,
#   runs the XGBoost + LightGBM ensemble, estimates alert timing,
#   and writes prediction.json.
#
# Run order (GitHub Actions cron):
#   Every hour → python -m src.pipeline.pipeline
#   At midnight → same script detects new day and updates history store
#
# Prerequisites (run once before first deployment):
#   python -m src.pipeline.rating_curve
#
# Input files (written by your existing fetchers in RTFD_Model):
#   data/dmc_data.json
#   data/weather_data.json
#   data/river_data.json
#
# Output:
#   data/prediction.json
# =============================================================================

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from config.settings import paths
from src.pipeline.accumulator  import (
    load_accumulator, add_reading, get_station_stats,
    save_accumulator, parse_rising_flag, extract_upstream_rainfall
)
from src.pipeline.history_store import (
    load_history, update_history, get_lag_features,
    build_daily_record, save_history
)
from src.pipeline.rating_curve import estimate_discharge
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# RTFD_Model data folder — where your fetchers write their JSON files
RTFD_DATA_DIR   = paths.PROJECT_ROOT / "data"

DMC_PATH        = RTFD_DATA_DIR / "dmc_data.json"
WEATHER_PATH    = RTFD_DATA_DIR / "weather_data.json"
RIVER_PATH      = RTFD_DATA_DIR / "river_data.json"
PREDICTION_PATH = RTFD_DATA_DIR / "prediction.json"

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR       = paths.MODELS
XGB_PATH        = MODEL_DIR / "xgboost_model.pkl"
LGBM_PATH       = MODEL_DIR / "lightgbm_model.pkl"
CURVES_PATH     = MODEL_DIR / "rating_curves.pkl"
WEIGHTS_PATH    = MODEL_DIR / "ensemble_weights.pkl"
FEATURE_COLS_PATH = paths.DATA_PROCESSED / "Gin river" / "feature_columns.txt"

# ── Station config ────────────────────────────────────────────────────────────
STATION_CODE = {'BAD01': 0, 'THA01': 1}
TARGET_STATIONS = ['BAD01', 'THA01']

LABEL_NAMES = {
    0: 'Normal',
    1: 'Alert',
    2: 'Minor Flood',
    3: 'Major Flood'
}

# ── Flood thresholds (from settings / station_master) ─────────────────────────
THRESHOLDS = {
    'BAD01': {'alert': 3.5, 'minor': 4.0, 'major': 5.0},
    'THA01': {'alert': 4.0, 'minor': 6.0, 'major': 7.5},
}


# =============================================================================
# Load models
# =============================================================================

def load_models() -> tuple:
    """Load XGBoost, LightGBM, rating curves and feature columns."""

    xgb_model  = joblib.load(XGB_PATH)
    lgbm_model = joblib.load(LGBM_PATH)
    curves     = joblib.load(CURVES_PATH)
    weights    = joblib.load(WEIGHTS_PATH)

    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    logger.info("Models loaded: XGBoost, LightGBM, rating curves")
    logger.info(f"Feature columns: {len(feature_cols)}")

    return xgb_model, lgbm_model, curves, weights, feature_cols


# =============================================================================
# Read live data files
# =============================================================================

def read_live_data() -> tuple:
    """
    Read the three JSON files written by the fetchers.
    Returns (dmc_data, weather_data, river_data).
    Any missing file returns an empty dict — pipeline continues with fallback.
    """
    def safe_load(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path.name}: {e}")
            return {}

    dmc_data     = safe_load(DMC_PATH)
    weather_data = safe_load(WEATHER_PATH)
    river_data   = safe_load(RIVER_PATH)

    logger.info(
        f"Live data loaded: "
        f"DMC={bool(dmc_data)}, "
        f"OWM={bool(weather_data)}, "
        f"ArcGIS={bool(river_data)}"
    )

    return dmc_data, weather_data, river_data


# =============================================================================
# Parse live data into per-station dicts
# =============================================================================

def parse_dmc(dmc_data: dict) -> dict:
    """
    Extract latest water level and rising flag per station from DMC JSON.
    DMC data may have multiple readings per day — we take the latest.

    Returns:
        {'BAD01': {'water_level': 1.09, 'rising_flag': 0, 'rainfall_mm': 4.9}, ...}
    """
    result = {}

    records = dmc_data if isinstance(dmc_data, list) else dmc_data.get('records', [])

    name_map = {'Baddegama': 'BAD01', 'Thawalama': 'THA01'}

    # Group by station and take latest record
    latest = {}
    for rec in records:
        name       = rec.get('gauging_station_name', '')
        station_id = name_map.get(name)
        if not station_id:
            continue
        time_str = rec.get('time_str', '')
        if station_id not in latest or time_str > latest[station_id]['time_str']:
            latest[station_id] = rec

    for station_id, rec in latest.items():
        result[station_id] = {
            'water_level':  float(rec.get('current_water_level', 0.0) or 0.0),
            'rising_flag':  parse_rising_flag(rec.get('rising_or_falling', '')),
            'rainfall_mm':  float(rec.get('rainfall_mm', 0.0) or 0.0),
        }

    return result


def parse_arcgis(river_data) -> dict:
    """
    Extract latest water level and rainfall per station from ArcGIS JSON.

    Returns:
        {'BAD01': {'water_level': 1.22, 'rainfall_mm': 0.0}, ...}
    """
    result = {}

    snapshots = river_data if isinstance(river_data, list) else [river_data]

    latest = {}
    for snapshot in snapshots:
        stations = snapshot.get('stations', []) if isinstance(snapshot, dict) else []
        for s in stations:
            station_id = s.get('station_id')
            if station_id in TARGET_STATIONS:
                obs_time = s.get('observed_at', '')
                if station_id not in latest or obs_time > latest[station_id]['observed_at']:
                    latest[station_id] = s

    for station_id, s in latest.items():
        result[station_id] = {
            'water_level': float(s.get('current_water_level_m', 0.0) or 0.0),
            'rainfall_mm': float(s.get('rainfall_mm_per_hour', 0.0) or 0.0),
        }

    return result


def parse_owm(weather_data: dict) -> dict:
    """
    Extract hourly rainfall per station from OWM JSON.
    All 6 stations present.

    Returns:
        {'BAD01': 0.0, 'THA01': 0.0, 'DEN01': 5.2, ...}
    """
    return extract_upstream_rainfall(weather_data)


# =============================================================================
# Build feature vector for one station
# =============================================================================

def build_feature_vector(
    station_id:        str,
    acc_stats:         dict,
    discharge:         dict,
    lag_features:      dict,
    upstream_rainfall: dict,
    feature_cols:      list,
) -> np.ndarray:
    """
    Assemble the 36-feature vector in the exact order from feature_columns.txt.

    This function is the critical bridge between live data and model input.
    Column order must exactly match what the models were trained on.

    Args:
        station_id:        'BAD01' or 'THA01'
        acc_stats:         today's running daily stats from accumulator
        discharge:         estimated q_avg, q_max, q_min from rating curve
        lag_features:      all lag and rolling features from history store
        upstream_rainfall: today's rainfall at all 6 stations from OWM
        feature_cols:      ordered list of column names from feature_columns.txt

    Returns:
        1D numpy array of shape (36,) ready for model.predict_proba()
    """
    # Build a dict of all available features
    # Keys must match feature_columns.txt exactly

    w_avg = acc_stats['w_avg']
    w_max = acc_stats['w_max']
    w_min = acc_stats['w_min']

    # Compute delta features using lag values from history
    w_avg_delta     = round(w_avg - lag_features.get('w_avg_lag_1', 0.0), 4)
    rainfall_delta  = round(
        acc_stats['rainfall_mm'] - lag_features.get('rainfall_mm_lag_1', 0.0), 4
    )

    all_features = {
        # Current snapshot features
        'rainfall_mm':   acc_stats['rainfall_mm'],
        'w_avg':         w_avg,
        'w_max':         w_max,
        'w_min':         w_min,
        'q_avg':         discharge['q_avg'],
        'q_max':         discharge['q_max'],
        'q_min':         discharge['q_min'],

        # Upstream rainfall today
        'rainfall_DEN01': upstream_rainfall.get('DEN01', 0.0),
        'rainfall_LAN01': upstream_rainfall.get('LAN01', 0.0),
        'rainfall_THA01': upstream_rainfall.get('THA01', 0.0),
        'rainfall_UDU01': upstream_rainfall.get('UDU01', 0.0),
        'rainfall_MAK01': upstream_rainfall.get('MAK01', 0.0),
        'rainfall_BAD01': upstream_rainfall.get('BAD01', 0.0),

        # Metadata
        'rising_flag':   acc_stats['rising_flag'],
        'station_code':  STATION_CODE[station_id],

        # Delta features
        'w_avg_delta':    w_avg_delta,
        'rainfall_delta': rainfall_delta,

        # All lag and rolling features from history store
        **lag_features,
    }

    # Assemble in exact column order
    vector = []
    for col in feature_cols:
        val = all_features.get(col, 0.0)
        vector.append(float(val))

    return np.array(vector, dtype=np.float32)


# =============================================================================
# Run ensemble prediction
# =============================================================================

def predict(
    xgb_model,
    lgbm_model,
    weights: dict,
    X: np.ndarray,
) -> tuple:
    """
    Run soft-voting ensemble prediction on a single feature vector.

    Args:
        xgb_model:  trained XGBClassifier
        lgbm_model: trained LGBMClassifier
        weights:    ensemble config dict with xgb_weight and lgbm_weight
        X:          feature vector shape (36,) — reshaped to (1, 36) internally

    Returns:
        (flood_label, flood_category, confidence, probabilities_dict)
    """
    X_2d = X.reshape(1, -1)   # model expects 2D input

    xgb_proba  = xgb_model.predict_proba(X_2d)[0]
    lgbm_proba = lgbm_model.predict_proba(X_2d)[0]

    avg_proba = (
        weights['xgb_weight']  * xgb_proba +
        weights['lgbm_weight'] * lgbm_proba
    )

    flood_label    = int(np.argmax(avg_proba))
    flood_category = LABEL_NAMES[flood_label]
    confidence     = round(float(avg_proba[flood_label]), 4)

    probabilities = {
        LABEL_NAMES[i]: round(float(avg_proba[i]), 4)
        for i in range(4)
    }

    return flood_label, flood_category, confidence, probabilities


# =============================================================================
# Estimate time to flood
# =============================================================================

def estimate_time_to_flood(
    station_id:  str,
    w_avg:       float,
    w_avg_delta: float,
    flood_label: int,
) -> dict:
    """
    Estimate how many hours until the next flood threshold is crossed,
    based on the current rate of water level rise.

    This gives the early warning time estimate —
    e.g. "flood conditions expected in approximately 4-6 hours"

    Only computed when flood_label >= 1 (Alert or above).

    Args:
        station_id:  'BAD01' or 'THA01'
        w_avg:       current average water level (m)
        w_avg_delta: rate of change m per day (from lag features)
                     converted to per hour internally
        flood_label: current prediction class (0-3)

    Returns:
        dict with estimated_hours_to_flood and next_threshold_m
        Returns empty dict if Normal or rate is not rising
    """
    if flood_label == 0:
        return {}

    thresholds = THRESHOLDS[station_id]

    # Determine the next threshold above current level
    if w_avg < thresholds['alert']:
        next_threshold = thresholds['alert']
        next_label     = 'Alert'
    elif w_avg < thresholds['minor']:
        next_threshold = thresholds['minor']
        next_label     = 'Minor Flood'
    elif w_avg < thresholds['major']:
        next_threshold = thresholds['major']
        next_label     = 'Major Flood'
    else:
        # Already at or above major flood threshold
        return {
            'estimated_hours_to_next_threshold': 0,
            'next_threshold_label': 'Major Flood',
            'next_threshold_m': thresholds['major'],
            'note': 'Already at or above major flood level'
        }

    # w_avg_delta is per day — convert to per hour
    rise_per_hour = w_avg_delta / 24.0

    if rise_per_hour <= 0:
        return {
            'note': 'Water level not rising — no flood time estimate available'
        }

    # Gap to next threshold
    gap_m = next_threshold - w_avg
    hours = gap_m / rise_per_hour

    # Round to nearest hour and add ±2 hour uncertainty window
    hours_rounded = max(1, round(hours))

    return {
        'estimated_hours_to_next_threshold': hours_rounded,
        'next_threshold_label': next_label,
        'next_threshold_m':     next_threshold,
        'current_water_level':  round(w_avg, 3),
        'rise_rate_m_per_hour': round(rise_per_hour, 4),
        'note': (
            f"At current rise rate, {next_label} conditions "
            f"expected in approximately {hours_rounded} hour(s)"
        )
    }


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline():
    """
    Main pipeline — runs every hour.
    Reads live data, assembles features, predicts, writes prediction.json.
    """
    now = datetime.now()
    today = str(date.today())
    is_midnight = now.hour == 0

    logger.info("=" * 55)
    logger.info(f"PIPELINE RUN — {now.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 55)

    # ── Load models ──
    xgb_model, lgbm_model, curves, weights, feature_cols = load_models()

    # ── Read live data ──
    dmc_data, weather_data, river_data = read_live_data()

    # ── Parse live data ──
    dmc_parsed    = parse_dmc(dmc_data)
    arcgis_parsed = parse_arcgis(river_data)
    owm_rainfall  = parse_owm(weather_data)

    # ── Load accumulator and history ──
    acc     = load_accumulator()
    history = load_history()

    # ── Process each target station ──
    station_predictions = {}

    for station_id in TARGET_STATIONS:

        logger.info(f"\nProcessing {station_id}...")

        # Get best available water level — prefer ArcGIS, fallback to DMC
        arcgis = arcgis_parsed.get(station_id, {})
        dmc    = dmc_parsed.get(station_id, {})

        water_level = arcgis.get('water_level') or dmc.get('water_level', 0.0)
        rainfall_1h = arcgis.get('rainfall_mm', 0.0)
        rising_flag = dmc.get('rising_flag', 0)

        # Add this hour's reading to accumulator
        acc = add_reading(acc, station_id, water_level, rainfall_1h, rising_flag)
        acc_stats = get_station_stats(acc, station_id)

        # Estimate discharge from water level using rating curve
        discharge = estimate_discharge(
            station_id,
            acc_stats['w_avg'],
            acc_stats['w_max'],
            acc_stats['w_min'],
            curves,
        )

        # Get lag features from history store
        lag_features = get_lag_features(history, station_id)

        # Build 36-feature vector
        X = build_feature_vector(
            station_id,
            acc_stats,
            discharge,
            lag_features,
            owm_rainfall,
            feature_cols,
        )

        # Run ensemble prediction
        flood_label, flood_category, confidence, probabilities = predict(
            xgb_model, lgbm_model, weights, X
        )

        # Estimate time to flood
        w_avg_delta = lag_features.get('w_avg_lag_1', 0.0)
        w_avg_delta_val = acc_stats['w_avg'] - w_avg_delta
        timing = estimate_time_to_flood(
            station_id, acc_stats['w_avg'], w_avg_delta_val, flood_label
        )

        thresholds = THRESHOLDS[station_id]

        station_predictions[station_id] = {
            'flood_category':        flood_category,
            'flood_label':           flood_label,
            'confidence':            confidence,
            'probabilities':         probabilities,
            'current_water_level_m': round(water_level, 3),
            'w_avg_m':               acc_stats['w_avg'],
            'w_max_m':               acc_stats['w_max'],
            'rainfall_today_mm':     acc_stats['rainfall_mm'],
            'rising_flag':           acc_stats['rising_flag'],
            'alert_level_m':         thresholds['alert'],
            'minor_flood_level_m':   thresholds['minor'],
            'major_flood_level_m':   thresholds['major'],
            'flood_timing':          timing,
            'data_source': {
                'arcgis_available': bool(arcgis),
                'dmc_available':    bool(dmc),
                'owm_available':    bool(owm_rainfall),
            }
        }

        logger.info(
            f"  {station_id} → {flood_category} "
            f"(confidence={confidence:.2%}, w={water_level:.3f}m)"
        )

        if timing.get('note'):
            logger.info(f"  Timing: {timing['note']}")

    # ── Save accumulator ──
    save_accumulator(acc)

    # ── At midnight — update history store with yesterday's completed values ──
    if is_midnight:
        logger.info("\nMidnight detected — updating history store...")
        yesterday = str(date.fromordinal(date.today().toordinal() - 1))

        for station_id in TARGET_STATIONS:
            acc_stats = get_station_stats(acc, station_id)

            # Build upstream rainfall totals for yesterday
            upstream_totals = {
                sid: owm_rainfall.get(sid, 0.0)
                for sid in ['DEN01', 'LAN01', 'THA01', 'UDU01', 'MAK01']
            }

            daily_record = build_daily_record(
                date_str=yesterday,
                station_id=station_id,
                w_avg=acc_stats['w_avg'],
                w_max=acc_stats['w_max'],
                w_min=acc_stats['w_min'],
                rainfall_mm=acc_stats['rainfall_mm'],
                upstream_rainfall=upstream_totals,
            )

            history = update_history(history, station_id, daily_record)

        save_history(history)

    # ── Write prediction.json ──
    prediction = {
        'predicted_at': now.strftime('%Y-%m-%dT%H:%M:%S'),
        'model':        'XGBoost + LightGBM ensemble',
        'stations':     station_predictions,
    }

    RTFD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_PATH, 'w') as f:
        json.dump(prediction, f, indent=2)

    logger.info(f"\nPrediction written → {PREDICTION_PATH}")
    logger.info("=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 55)

    return prediction


if __name__ == "__main__":
    prediction = run_pipeline()

    print("\n-- Prediction output --")
    for station_id, result in prediction['stations'].items():
        print(f"\n{station_id}:")
        print(f"  Category:   {result['flood_category']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Water level:{result['current_water_level_m']} m")
        if result['flood_timing'].get('note'):
            print(f"  Timing:     {result['flood_timing']['note']}")