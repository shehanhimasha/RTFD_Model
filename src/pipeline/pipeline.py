# Main orchestrator — runs every hour via GitHub Actions.
#
# Fixes applied:
#   FIX 1 — Upstream rainfall in history_store was always 0 because the
#            midnight update was reading the raw OWM hourly reading (almost
#            always 0 at midnight) instead of the day's accumulated total.
#            Now a separate upstream_accumulator.json sums OWM hourly
#            readings for all stations throughout the day, same as the
#            main accumulator does for BAD01 and THA01 water levels.
#
#   FIX 2 — Live data was never stored for future retraining.
#            After each run, all 36 features + prediction output are
#            appended to data/live/daily_log.csv. This grows into the
#            real-world training dataset for future model improvement.
#
# Input files:
#   data/dmc_data.json
#   data/weather_data.json
#   data/river_data.json
#
# Output:
#   data/prediction.json
#   data/live/daily_accumulator.json
#   data/live/upstream_accumulator.json   
#   data/live/history_store.json
#   data/live/daily_log.csv              
# =============================================================================

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo
from pathlib import Path
from config.settings import paths
from src.pipeline.accumulator import (
    load_accumulator, add_reading, get_station_stats,
    save_accumulator, parse_rising_flag, extract_upstream_rainfall,
)
from src.pipeline.history_store import (
    load_history, update_history, get_lag_features,
    build_daily_record, save_history,
)
from src.pipeline.rating_curve import estimate_discharge
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RTFD_DATA_DIR     = paths.PROJECT_ROOT / "data"
LIVE_DIR          = RTFD_DATA_DIR / "live"

DMC_PATH          = RTFD_DATA_DIR / "dmc_data.json"
WEATHER_PATH      = RTFD_DATA_DIR / "weather_data.json"
RIVER_PATH        = RTFD_DATA_DIR / "river_data.json"
PREDICTION_PATH   = RTFD_DATA_DIR / "prediction.json"

UPSTREAM_ACC_PATH = LIVE_DIR / "upstream_accumulator.json"   # FIX 1
DAILY_LOG_PATH    = LIVE_DIR / "daily_log.csv"               # FIX 2

MODEL_DIR         = paths.MODELS
XGB_PATH          = MODEL_DIR / "xgboost_model.pkl"
LGBM_PATH         = MODEL_DIR / "lightgbm_model.pkl"
CURVES_PATH       = MODEL_DIR / "rating_curves.pkl"
WEIGHTS_PATH      = MODEL_DIR / "ensemble_weights.pkl"
FEATURE_COLS_PATH = paths.DATA_PROCESSED / "Gin river" / "feature_columns.txt"

# ── Station config ────────────────────────────────────────────────────────────
STATION_CODE      = {'BAD01': 0, 'THA01': 1}
TARGET_STATIONS   = ['BAD01', 'THA01']
UPSTREAM_STATIONS = ['DEN01', 'LAN01', 'THA01', 'UDU01', 'MAK01']
ALL_STATIONS      = ['BAD01', 'THA01', 'DEN01', 'LAN01', 'UDU01', 'MAK01']

LABEL_NAMES = {0: 'Normal', 1: 'Alert', 2: 'Minor Flood', 3: 'Major Flood'}

THRESHOLDS = {
    'BAD01': {'alert': 3.5, 'minor': 4.0, 'major': 5.0},
    'THA01': {'alert': 4.0, 'minor': 6.0, 'major': 7.5},
}


# =============================================================================
# Upstream accumulator
# Accumulates Open-Meteo hourly rainfall for all stations throughout the day.
# At midnight these totals go into history_store as the day's upstream
# rainfall — replacing the broken approach of reading a single midnight value.
# =============================================================================

def load_upstream_accumulator() -> dict:
    """Load upstream accumulator, reset if from a previous day."""
    today = str(datetime.now(ZoneInfo("Asia/Colombo")).date())

    if UPSTREAM_ACC_PATH.exists():
        with open(UPSTREAM_ACC_PATH, 'r') as f:
            acc = json.load(f)
        if acc.get('date') != today:
            logger.info("New day — resetting upstream accumulator")
            acc = _empty_upstream_acc(today)
    else:
        acc = _empty_upstream_acc(today)

    return acc


def _empty_upstream_acc(today: str) -> dict:
    """Create a fresh upstream accumulator for a new day."""
    acc = {'date': today}
    for sid in ALL_STATIONS:
        acc[sid] = 0.0
    return acc


def update_upstream_accumulator(acc: dict, open_meteo_rainfall: dict) -> dict:
    """
    Add this hour's Open-Meteo rainfall reading to each station's running daily total.

    open_meteo_rainfall contains rainfall_1h_mm for all 6 stations.
    We sum these hourly values throughout the day so that at midnight
    we have the true daily total — not a single midnight reading.
    """
    for sid in ALL_STATIONS:
        hourly    = float(open_meteo_rainfall.get(sid, 0.0))
        acc[sid]  = round(acc.get(sid, 0.0) + hourly, 2)

    non_zero = {k: v for k, v in acc.items() if k != 'date' and v > 0}
    if non_zero:
        logger.info(f"  Upstream rainfall today: {non_zero}")
    else:
        logger.info("  Upstream rainfall today: all 0.0mm so far")

    return acc


def save_upstream_accumulator(acc: dict) -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(UPSTREAM_ACC_PATH, 'w') as f:
        json.dump(acc, f, indent=2)


# =============================================================================
# Daily log
# Appends all 36 features + prediction to daily_log.csv every run.
# Grows into real-world training data for future model retraining.
# =============================================================================

def append_to_daily_log(
    station_id:    str,
    run_time:      str,
    feature_cols:  list,
    feature_vec:   np.ndarray,
    flood_label:   int,
    flood_category: str,
    confidence:    float,
) -> None:
    """
    Append one row to daily_log.csv.

    Columns: run_time, station_id, flood_label, flood_category,
             confidence, then all 36 feature values.

    When you have 6-12 months of this data, run the training pipeline
    using daily_log.csv as additional training data to improve the model
    on recent real-world conditions.
    """
    row = {
        'run_time':      run_time,
        'station_id':    station_id,
        'flood_label':   flood_label,
        'flood_category': flood_category,
        'confidence':    confidence,
    }
    for col, val in zip(feature_cols, feature_vec):
        row[col] = round(float(val), 6)

    row_df = pd.DataFrame([row])

    if DAILY_LOG_PATH.exists():
        row_df.to_csv(DAILY_LOG_PATH, mode='a', header=False, index=False)
    else:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        row_df.to_csv(DAILY_LOG_PATH, mode='w', header=True, index=False)

    logger.info(f"  Appended to daily_log.csv")


# =============================================================================
# Load models
# =============================================================================

def load_models() -> tuple:
    xgb_model  = joblib.load(XGB_PATH)
    lgbm_model = joblib.load(LGBM_PATH)
    curves     = joblib.load(CURVES_PATH)
    weights    = joblib.load(WEIGHTS_PATH)

    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    logger.info(f"Models loaded — {len(feature_cols)} feature columns")
    return xgb_model, lgbm_model, curves, weights, feature_cols


# =============================================================================
# Read and parse live data
# =============================================================================

def safe_load(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)

    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return {}


def read_live_data() -> tuple:

    dmc_data     = safe_load(DMC_PATH)
    weather_data = safe_load(WEATHER_PATH)
    river_data   = safe_load(RIVER_PATH)

    logger.info(
        f"Live data: DMC={bool(dmc_data)}, "
        f"Open-Meteo={bool(weather_data)}, ArcGIS={bool(river_data)}"
    )

    return dmc_data, weather_data, river_data


def parse_dmc(dmc_data: dict) -> dict:
    result   = {}
    records  = dmc_data if isinstance(dmc_data, list) else dmc_data.get('records', [])
    name_map = {'Baddegama': 'BAD01', 'Thawalama': 'THA01'}
    latest   = {}

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
            'water_level': float(rec.get('current_water_level', 0.0) or 0.0),
            'rising_flag': parse_rising_flag(rec.get('rising_or_falling', '')),
            'rainfall_mm': float(rec.get('rainfall_mm', 0.0) or 0.0),
        }
    return result


def parse_arcgis(river_data) -> dict:
    result = {}
    
    # Extract all station objects
    stations_list = []
    if isinstance(river_data, list):
        for item in river_data:
            if 'stations' in item:
                stations_list.extend(item['stations'])
            else:
                stations_list.append(item)
    elif isinstance(river_data, dict):
        stations_list = river_data.get('stations', [])

    # Get the latest reading for each station
    latest = {}
    for s in stations_list:
        sid = s.get('station_id')
        if sid in TARGET_STATIONS:
            obs_time = s.get('observed_at', '')
            if sid not in latest or obs_time > latest[sid].get('observed_at', ''):
                latest[sid] = s

    for sid, s in latest.items():
        result[sid] = {
            'water_level': float(s.get('current_water_level_m', 0.0) or 0.0),
            'rainfall_mm': float(s.get('rainfall_mm_per_hour', 0.0) or 0.0),
            'rainfall_mm_per_day': s.get('rainfall_mm_per_day'),
        }
    return result


def parse_open_meteo(weather_data: dict) -> dict:
    return extract_upstream_rainfall(weather_data)


# =============================================================================
# Build 36-feature vector
# =============================================================================

def build_feature_vector(
    station_id:        str,
    acc_stats:         dict,
    discharge:         dict,
    lag_features:      dict,
    upstream_daily:    dict,
    feature_cols:      list,
) -> np.ndarray:
    """
    Assemble the exact 36-column feature vector in feature_columns.txt order.

    upstream_daily: accumulated daily totals from upstream_accumulator
                    (not raw hourly Open-Meteo — that was FIX 1)
    """
    w_avg = acc_stats['w_avg']

    w_avg_delta    = round(w_avg - lag_features.get('w_avg_lag_1', 0.0), 4)
    rainfall_delta = round(
        acc_stats['rainfall_mm'] - lag_features.get('rainfall_mm_lag_1', 0.0), 4
    )

    all_features = {
        'rainfall_mm':    acc_stats['rainfall_mm'],
        'w_avg':          w_avg,
        'w_max':          acc_stats['w_max'],
        'w_min':          acc_stats['w_min'],
        'q_avg':          discharge['q_avg'],
        'q_max':          discharge['q_max'],
        'q_min':          discharge['q_min'],

        # Today's accumulated upstream rainfall totals
        'rainfall_DEN01': upstream_daily.get('DEN01', 0.0),
        'rainfall_LAN01': upstream_daily.get('LAN01', 0.0),
        'rainfall_THA01': upstream_daily.get('THA01', 0.0),
        'rainfall_UDU01': upstream_daily.get('UDU01', 0.0),
        'rainfall_MAK01': upstream_daily.get('MAK01', 0.0),
        'rainfall_BAD01': upstream_daily.get('BAD01', 0.0),

        'rising_flag':    acc_stats['rising_flag'],
        'station_code':   STATION_CODE[station_id],

        'w_avg_delta':    w_avg_delta,
        'rainfall_delta': rainfall_delta,

        **lag_features,
    }

    vector = [float(all_features.get(col, 0.0)) for col in feature_cols]
    return np.array(vector, dtype=np.float32)


# =============================================================================
# Ensemble prediction
# =============================================================================

def predict(
    xgb_model,
    lgbm_model,
    weights:      dict,
    X:            np.ndarray,
    feature_cols: list,
) -> tuple:
    # Wrap in DataFrame so LightGBM doesn't warn about missing feature names
    X_df = pd.DataFrame(X.reshape(1, -1), columns=feature_cols)

    xgb_proba  = xgb_model.predict_proba(X_df)[0]
    lgbm_proba = lgbm_model.predict_proba(X_df)[0]

    avg_proba = (
        weights['xgb_weight']  * xgb_proba +
        weights['lgbm_weight'] * lgbm_proba
    )

    flood_label    = int(np.argmax(avg_proba))
    flood_category = LABEL_NAMES[flood_label]
    confidence     = round(float(avg_proba[flood_label]), 4)
    probabilities  = {
        LABEL_NAMES[i]: round(float(avg_proba[i]), 4)
        for i in range(4)
    }

    return flood_label, flood_category, confidence, probabilities


# =============================================================================
# Alert timing estimator
# =============================================================================

def estimate_time_to_flood(
    station_id:  str,
    w_avg:       float,
    w_avg_delta: float,
    flood_label: int,
) -> dict:
    if flood_label == 0:
        return {}

    thresholds = THRESHOLDS[station_id]

    if w_avg < thresholds['alert']:
        next_threshold, next_label = thresholds['alert'], 'Alert'
    elif w_avg < thresholds['minor']:
        next_threshold, next_label = thresholds['minor'], 'Minor Flood'
    elif w_avg < thresholds['major']:
        next_threshold, next_label = thresholds['major'], 'Major Flood'
    else:
        return {
            'estimated_hours_to_next_threshold': 0,
            'next_threshold_label': 'Major Flood',
            'next_threshold_m': thresholds['major'],
            'note': 'Already at or above major flood level',
        }

    rise_per_hour = w_avg_delta / 24.0
    if rise_per_hour <= 0:
        return {'note': 'Water level not rising — no flood time estimate'}

    hours = max(1, round((next_threshold - w_avg) / rise_per_hour))
    return {
        'estimated_hours_to_next_threshold': hours,
        'next_threshold_label': next_label,
        'next_threshold_m':     next_threshold,
        'current_water_level':  round(w_avg, 3),
        'rise_rate_m_per_hour': round(rise_per_hour, 4),
        'note': (
            f"At current rise rate, {next_label} conditions "
            f"expected in approximately {hours} hour(s)"
        ),
    }


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline():
    now         = datetime.now(ZoneInfo("Asia/Colombo"))
    is_midnight = now.hour in [0, 1]

    logger.info("=" * 55)
    logger.info(f"PIPELINE RUN — {now.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 55)

    xgb_model, lgbm_model, curves, weights, feature_cols = load_models()
    dmc_data, weather_data, river_data = read_live_data()

    dmc_parsed    = parse_dmc(dmc_data)
    arcgis_parsed = parse_arcgis(river_data)
    open_meteo_rainfall = parse_open_meteo(weather_data)

    acc          = load_accumulator()
    history      = load_history()
    upstream_acc = load_upstream_accumulator()

    # accumulate upstream rainfall
    upstream_acc = update_upstream_accumulator(upstream_acc, open_meteo_rainfall)

    station_predictions = {}

    for station_id in TARGET_STATIONS:
        logger.info(f"\nProcessing {station_id}...")

        arcgis = arcgis_parsed.get(station_id, {})
        dmc    = dmc_parsed.get(station_id, {})

        water_level = arcgis.get('water_level') or dmc.get('water_level', 0.0)
        rainfall_1h = arcgis.get('rainfall_mm', 0.0)
        rainfall_day = arcgis.get('rainfall_mm_per_day')
        rising_flag = dmc.get('rising_flag', 0)

        lag_features = get_lag_features(history, station_id)

        # Forward-fill missing water level to prevent sudden drops to 0.
        if water_level <= 0:
            prev_stats = get_station_stats(acc, station_id)
            if prev_stats['w_avg'] > 0:
                water_level = prev_stats['w_avg']
                logger.warning(f"  Missing live water level for {station_id}. Forward-filling with today's average {water_level:.3f}m")
            elif lag_features.get('w_avg_lag_1', 0.0) > 0:
                water_level = lag_features.get('w_avg_lag_1', 0.0)
                logger.warning(f"  Missing live water level for {station_id}. Forward-filling with yesterday's average {water_level:.3f}m")

        acc       = add_reading(acc, station_id, water_level, rainfall_1h, rising_flag, rainfall_day_mm=rainfall_day)
        acc_stats = get_station_stats(acc, station_id)

        discharge = estimate_discharge(
            station_id,
            acc_stats['w_avg'], acc_stats['w_max'], acc_stats['w_min'],
            curves,
        )

        # pass accumulated upstream totals
        X = build_feature_vector(
            station_id, acc_stats, discharge,
            lag_features, upstream_acc, feature_cols,
        )

        flood_label, flood_category, confidence, probabilities = predict(
            xgb_model, lgbm_model, weights, X, feature_cols
        )

        # log to daily_log.csv
        append_to_daily_log(
            station_id     = station_id,
            run_time       = now.strftime('%Y-%m-%dT%H:%M:%S'),
            feature_cols   = feature_cols,
            feature_vec    = X,
            flood_label    = flood_label,
            flood_category = flood_category,
            confidence     = confidence,
        )

        w_avg_delta_val = acc_stats['w_avg'] - lag_features.get('w_avg_lag_1', 0.0)
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
                'open_meteo_available': bool(open_meteo_rainfall),
            },
        }

        logger.info(
            f"  {station_id} → {flood_category} "
            f"(confidence={confidence:.2%}, w={water_level:.3f}m)"
        )

    # Save accumulators
    save_accumulator(acc)
    save_upstream_accumulator(upstream_acc)   # FIX 1

    # Midnight — update history with correctly accumulated values
    if is_midnight:
        logger.info("\nMidnight — updating history store...")
        yesterday = str(date.fromordinal(datetime.now(ZoneInfo("Asia/Colombo")).date().toordinal() - 1))

        for station_id in TARGET_STATIONS:
            acc_stats = get_station_stats(acc, station_id)

            # upstream_acc has the real daily totals
            upstream_totals = {
                sid: upstream_acc.get(sid, 0.0)
                for sid in UPSTREAM_STATIONS
            }

            daily_record = build_daily_record(
                date_str          = yesterday,
                station_id        = station_id,
                w_avg             = acc_stats['w_avg'],
                w_max             = acc_stats['w_max'],
                w_min             = acc_stats['w_min'],
                rainfall_mm       = acc_stats['rainfall_mm'],
                upstream_rainfall = upstream_totals,
            )
            history = update_history(history, station_id, daily_record)

        save_history(history)

    # Write prediction.json
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

    print("\n── Prediction output ──")
    for sid, result in prediction['stations'].items():
        print(f"\n{sid}:")
        print(f"  Category:    {result['flood_category']}")
        print(f"  Confidence:  {result['confidence']:.2%}")
        print(f"  Water level: {result['current_water_level_m']}m")
        if result['flood_timing'].get('note'):
            print(f"  Timing:      {result['flood_timing']['note']}")