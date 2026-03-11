"""
PIPELINE — Data Merge + Model Inference
Loads all 3 JSON data sources, merges features by station, runs the
XGBoost + LSTM models, and saves data/prediction.json.

Falls back to last saved JSON for any source that failed to update —
the pipeline never crashes due to a missing source.
"""

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

DMC_FILE     = Path("data/dmc_data.json")
WEATHER_FILE = Path("data/weather_data.json")
RIVER_FILE   = Path("data/river_data.json")
OUTPUT_FILE  = Path("data/prediction.json")

XGB_MODEL_PATH  = Path("models/xgboost_model.pkl")
LSTM_MODEL_PATH = Path("models/lstm_model.keras")

STATIONS = [
    {"station_id": "BAD01", "station_name": "Baddegama"},
    {"station_id": "THA01", "station_name": "Thawalama"},
]

# Flood alert thresholds (metres) — used for rule-based fallback
FLOOD_THRESHOLDS = {
    "Baddegama": {"alert": 4.0, "major": 5.0},
    "Thawalama": {"alert": 4.5, "major": 5.5},
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [Pipeline] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── Safe JSON Loader ──────────────────────────────────────────────────────────


def load_json(path: Path, label: str):
    """Load JSON from path; return None on any error (never raise)."""
    if not path.exists():
        log.warning("[%s] File not found: %s", label, path)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.error("[%s] Failed to read %s: %s", label, path, exc)
        return None


# ── Feature Extraction ────────────────────────────────────────────────────────


def get_dmc_features(dmc_data, station_name: str) -> dict:
    """Extract the most recent DMC record for a station."""
    defaults = {
        "dmc_current_wl": None,
        "dmc_previous_wl": None,
        "dmc_rainfall_mm": None,
        "dmc_rising": None,
    }
    if not dmc_data:
        return defaults

    # Filter to the target station (latest record = last in list)
    records = [
        r for r in dmc_data
        if str(r.get("gauging_station_name", "")).lower() == station_name.lower()
    ]
    if not records:
        return defaults

    latest = records[-1]
    # Determine rise flag
    rf = str(latest.get("rising_or_falling", "")).lower()
    rising = 1 if "ris" in rf else (0 if "fall" in rf else None)

    return {
        "dmc_current_wl":   latest.get("current_water_level"),
        "dmc_previous_wl":  latest.get("previous_water_level"),
        "dmc_rainfall_mm":  latest.get("rainfall_mm"),
        "dmc_rising":       rising,
    }


def get_weather_features(weather_data, station_name: str) -> dict:
    """Extract OWM features for a station."""
    defaults = {
        "owm_rainfall_1h": None,
        "owm_humidity":    None,
        "owm_clouds_pct":  None,
    }
    if not weather_data:
        return defaults

    if isinstance(weather_data, list):
        latest_snapshot = weather_data[-1] if weather_data else {}
    else:
        latest_snapshot = weather_data

    locations = latest_snapshot.get("locations", [])
    for loc in locations:
        if str(loc.get("station_name", "")).lower() == station_name.lower():
            return {
                "owm_rainfall_1h": loc.get("rainfall_1h_mm"),
                "owm_humidity":    loc.get("humidity"),
                "owm_clouds_pct":  loc.get("clouds_pct"),
            }
    return defaults


def get_river_features(river_data, station_name: str) -> dict:
    """Extract ArcGIS river features for a station."""
    defaults = {
        "arc_water_level_m":    None,
        "arc_rainfall_mm_hr":   None,
    }
    if not river_data:
        return defaults

    if isinstance(river_data, list):
        latest_snapshot = river_data[-1] if river_data else {}
    else:
        latest_snapshot = river_data

    stations = latest_snapshot.get("stations", [])
    for st in stations:
        if str(st.get("station_name", "")).lower() == station_name.lower():
            return {
                "arc_water_level_m":  st.get("current_water_level_m"),
                "arc_rainfall_mm_hr": st.get("rainfall_mm_per_hour"),
            }
    return defaults


def build_feature_vector(features: dict) -> np.ndarray:
    """
    Flatten the merged feature dict into a 1-D numpy array.
    NaN is used for missing values so models can handle them.
    """
    ordered_keys = [
        "dmc_current_wl", "dmc_previous_wl", "dmc_rainfall_mm", "dmc_rising",
        "owm_rainfall_1h", "owm_humidity", "owm_clouds_pct",
        "arc_water_level_m", "arc_rainfall_mm_hr",
    ]
    vec = np.array(
        [float(features.get(k) if features.get(k) is not None else np.nan)
         for k in ordered_keys],
        dtype=np.float32,
    )
    return vec


# ── Model Loading ─────────────────────────────────────────────────────────────


def load_xgb_model():
    """Load XGBoost model from disk. Returns None if not available."""
    if not XGB_MODEL_PATH.exists():
        log.warning("XGBoost model not found at %s.", XGB_MODEL_PATH)
        return None
    try:
        import joblib
        model = joblib.load(XGB_MODEL_PATH)
        log.info("XGBoost model loaded from %s.", XGB_MODEL_PATH)
        return model
    except Exception as exc:
        log.error("Failed to load XGBoost model: %s", exc)
        return None


def load_lstm_model():
    """Load LSTM Keras model from disk. Returns None if not available."""
    if not LSTM_MODEL_PATH.exists():
        log.warning("LSTM model not found at %s.", LSTM_MODEL_PATH)
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        log.info("LSTM model loaded from %s.", LSTM_MODEL_PATH)
        return model
    except Exception as exc:
        log.error("Failed to load LSTM model: %s", exc)
        return None


# ── Inference ─────────────────────────────────────────────────────────────────


def rule_based_prediction(station_name: str, features: dict) -> tuple[str, float, float | None]:
    """
    Simple threshold-based fallback when ML models are not available.
    Returns (flood_risk, predicted_water_level_m, confidence).
    """
    # Best available water level estimate
    wl = (
        features.get("arc_water_level_m")
        or features.get("dmc_current_wl")
        or 0.0
    )
    # Best available rainfall estimate
    rain = (
        features.get("dmc_rainfall_mm")
        or features.get("arc_rainfall_mm_hr")
        or features.get("owm_rainfall_1h")
        or 0.0
    )

    thresholds = FLOOD_THRESHOLDS.get(station_name, {"alert": 4.0, "major": 5.0})

    if wl >= thresholds["major"] or rain >= 50:
        risk = "High"
        confidence = 0.80
    elif wl >= thresholds["alert"] or rain >= 25:
        risk = "Medium"
        confidence = 0.70
    else:
        risk = "Low"
        confidence = 0.65

    return risk, float(wl), confidence


def run_xgb_inference(model, vec: np.ndarray) -> tuple[float, float]:
    """
    Run XGBoost inference.
    Returns (predicted_water_level_m, confidence).
    Expects model outputs: [wl_prediction] or probability.
    """
    try:
        # Replace NaN with column mean (0 here as a safe fallback)
        clean = np.where(np.isnan(vec), 0.0, vec)
        pred = model.predict(clean.reshape(1, -1))
        proba = model.predict_proba(clean.reshape(1, -1)) if hasattr(model, "predict_proba") else None
        wl = float(pred[0]) if pred is not None else 0.0
        conf = float(np.max(proba[0])) if proba is not None else 0.75
        return wl, conf
    except Exception as exc:
        log.error("XGBoost inference error: %s", exc)
        return 0.0, 0.5


def run_lstm_inference(model, vec: np.ndarray) -> tuple[float, float]:
    """
    Run LSTM inference. Reshapes the vector to (1, timesteps, features).
    Returns (predicted_water_level_m, confidence).
    """
    try:
        clean = np.where(np.isnan(vec), 0.0, vec)
        # LSTM expects 3-D input: (batch, timesteps, features)
        # With a single snapshot we treat it as 1 timestep
        x = clean.reshape(1, 1, -1).astype(np.float32)
        pred = model.predict(x, verbose=0)
        wl = float(pred[0][0])
        conf = 0.80  # LSTM doesn't natively output confidence; use fixed value
        return wl, conf
    except Exception as exc:
        log.error("LSTM inference error: %s", exc)
        return 0.0, 0.5


def classify_risk(predicted_wl: float, station_name: str) -> str:
    thresholds = FLOOD_THRESHOLDS.get(station_name, {"alert": 4.0, "major": 5.0})
    if predicted_wl >= thresholds["major"]:
        return "High"
    elif predicted_wl >= thresholds["alert"]:
        return "Medium"
    return "Low"


def predict_station(
    station: dict,
    features: dict,
    xgb_model,
    lstm_model,
) -> dict:
    """Produce a prediction record for one station."""
    station_name = station["station_name"]
    vec = build_feature_vector(features)

    if xgb_model is not None and lstm_model is not None:
        xgb_wl, xgb_conf = run_xgb_inference(xgb_model, vec)
        lstm_wl, lstm_conf = run_lstm_inference(lstm_model, vec)
        # Ensemble: average the two predictions
        predicted_wl = round((xgb_wl + lstm_wl) / 2, 2)
        confidence = round((xgb_conf + lstm_conf) / 2, 2)
        risk = classify_risk(predicted_wl, station_name)
    elif xgb_model is not None:
        predicted_wl, confidence = run_xgb_inference(xgb_model, vec)
        predicted_wl = round(predicted_wl, 2)
        confidence = round(confidence, 2)
        risk = classify_risk(predicted_wl, station_name)
    elif lstm_model is not None:
        predicted_wl, confidence = run_lstm_inference(lstm_model, vec)
        predicted_wl = round(predicted_wl, 2)
        confidence = round(confidence, 2)
        risk = classify_risk(predicted_wl, station_name)
    else:
        log.warning(
            "No ML models available for %s — using rule-based fallback.", station_name
        )
        risk, predicted_wl, confidence = rule_based_prediction(station_name, features)
        predicted_wl = round(predicted_wl, 2)
        confidence = round(confidence, 2)

    return {
        "station_id": station["station_id"],
        "station_name": station_name,
        "flood_risk": risk,
        "predicted_water_level_m": predicted_wl,
        "confidence": confidence,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== Pipeline starting ===")

    # Load all sources (graceful fallback on missing/corrupt files)
    dmc_data     = load_json(DMC_FILE,     "DMC")
    weather_data = load_json(WEATHER_FILE, "OWM")
    river_data   = load_json(RIVER_FILE,   "ArcGIS")

    sources_ok = sum(x is not None for x in [dmc_data, weather_data, river_data])
    log.info("%d/3 data sources loaded.", sources_ok)

    # Load models
    xgb_model  = load_xgb_model()
    lstm_model = load_lstm_model()

    predictions = []
    for station in STATIONS:
        name = station["station_name"]
        features = {}
        features.update(get_dmc_features(dmc_data, name))
        features.update(get_weather_features(weather_data, name))
        features.update(get_river_features(river_data, name))

        log.info("Features for %s: %s", name, features)
        pred = predict_station(station, features, xgb_model, lstm_model)
        predictions.append(pred)
        log.info("Prediction for %s: %s", name, pred)

    payload = {
        "predicted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "predictions": predictions,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    log.info("Saved prediction to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
