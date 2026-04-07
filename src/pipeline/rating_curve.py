# =============================================================================
#   Fits a power curve (q = a × w^b) for each station using 15 years
#   of historical water level and discharge data.
#   Saves the fitted coefficients so pipeline.py can estimate discharge
#   from real-time water level readings (ArcGIS gives water level only,
#   no discharge).
#
# Why a power curve:
#   The relationship between river water level (stage) and discharge
#   is well established in hydrology as a power function.
#   As water level doubles, discharge more than doubles because the
#   river gets both deeper AND wider. The exponent b captures this.
#
# Run once before deploying pipeline:
#   python -m src.pipeline.rating_curve
#
# Input:
#   data/processed/Gin river/Flood data BAD01.csv
#   data/processed/Gin river/Flood data THA01.csv
#
# Output:
#   models/trained/rating_curves.pkl
# =============================================================================

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from config.settings import paths
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
GIN_PROCESSED  = paths.DATA_PROCESSED / "Gin river"
CURVES_PATH    = paths.MODELS / "rating_curves.pkl"

# ── Stations to fit curves for ────────────────────────────────────────────────
STATIONS = {
    'BAD01': 'Flood data BAD01.csv',
    'THA01': 'Flood data THA01.csv',
}


# =============================================================================
# Power curve function
# =============================================================================

def power_curve(w, a, b):
    """
    Rating curve equation: q = a × w^b

    w: water level (m)
    a: scale coefficient — how much discharge at w=1
    b: shape exponent — how steeply discharge grows with level
       typical values: b between 1.5 and 3.0 for natural rivers

    Returns estimated discharge q (m³/s)
    """
    return a * np.power(w, b)


# =============================================================================
# Fit curve for one station
# =============================================================================

def fit_station_curve(station_id: str, filename: str) -> dict:
    """
    Fit the power curve for one station using historical data.

    Uses w_avg → q_avg as the primary pair.
    Also computes w_max → q_max and w_min → q_min coefficients
    by fitting separate curves for each pair.

    Args:
        station_id: e.g. 'BAD01'
        filename:   CSV filename in the processed directory

    Returns:
        dict with fitted coefficients for avg, max, min pairs
    """
    logger.info(f"\nFitting rating curve for {station_id}...")

    df = pd.read_csv(GIN_PROCESSED / filename)

    cols = ['w_avg', 'q_avg', 'w_max', 'q_max', 'w_min', 'q_min']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop invalid rows
    before = len(df)
    df = df.dropna(subset=cols)
    after = len(df)

    logger.info(f"  Using {len(df):,} clean rows for fitting")

    curves = {}

    # Fit one curve per water level / discharge pair
    pairs = [
        ('w_avg', 'q_avg', 'avg'),
        ('w_max', 'q_max', 'max'),
        ('w_min', 'q_min', 'min'),
    ]

    for w_col, q_col, label in pairs:
        w = df[w_col].values
        q = df[q_col].values

        try:
            # curve_fit finds a and b that minimise sum of squared errors
            # p0 = initial guess for [a, b] — helps convergence
            # bounds = keep a > 0 and b > 0 (physically required)
            popt, _ = curve_fit(
                power_curve,
                w, q,
                p0=[10.0, 2.0],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=10000,
            )

            a, b = popt

            # Evaluate fit quality — R² score
            q_pred = power_curve(w, a, b)
            ss_res = np.sum((q - q_pred) ** 2)
            ss_tot = np.sum((q - np.mean(q)) ** 2)
            r2     = 1 - (ss_res / ss_tot)

            curves[label] = {'a': float(a), 'b': float(b), 'r2': float(r2)}

            logger.info(
                f"  {label:3}: q = {a:.4f} × w^{b:.4f}  "
                f"R² = {r2:.4f}"
                + (" ✅" if r2 >= 0.90 else " ⚠️  R² below 0.90")
            )

        except RuntimeError as e:
            logger.warning(f"  {label}: curve_fit failed — {e}")
            # Fallback: use simple linear ratio as estimate
            ratio = np.median(q / w)
            curves[label] = {'a': float(ratio), 'b': 1.0, 'r2': 0.0}
            logger.warning(f"  {label}: using linear fallback ratio={ratio:.4f}")

    return curves


# =============================================================================
# Estimate discharge from water level
# =============================================================================

def estimate_discharge(
    station_id: str,
    w_avg: float,
    w_max: float,
    w_min: float,
    curves: dict,
) -> dict:
    """
    Estimate q_avg, q_max, q_min from water level using fitted curves.

    Called by pipeline.py every hour to estimate discharge from
    ArcGIS water level readings.

    Args:
        station_id: 'BAD01' or 'THA01'
        w_avg:      today's average water level so far
        w_max:      today's max water level so far
        w_min:      today's min water level so far
        curves:     the loaded rating_curves dict

    Returns:
        dict with q_avg, q_max, q_min estimates
    """
    station_curves = curves.get(station_id, {})

    def apply_curve(w, label):
        if label not in station_curves:
            return 0.0
        c = station_curves[label]
        if w <= 0:
            return 0.0
        return float(power_curve(w, c['a'], c['b']))

    return {
        'q_avg': apply_curve(w_avg, 'avg'),
        'q_max': apply_curve(w_max, 'max'),
        'q_min': apply_curve(w_min, 'min'),
    }


# =============================================================================
# Main — fit and save
# =============================================================================

def run_fit_rating_curves() -> dict:
    """
    Fit rating curves for all stations and save to disk.
    Run this once before deploying the pipeline.
    """
    logger.info("=" * 55)
    logger.info("RATING CURVE FITTING — Start")
    logger.info("=" * 55)

    all_curves = {}

    for station_id, filename in STATIONS.items():
        curves = fit_station_curve(station_id, filename)
        all_curves[station_id] = curves

    # Save
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_curves, CURVES_PATH)

    logger.info(f"\nRating curves saved → {CURVES_PATH}")
    logger.info("=" * 55)
    logger.info("RATING CURVE FITTING — Complete")
    logger.info("=" * 55)

    return all_curves


# =============================================================================
# Quick test — verify estimate works correctly
# =============================================================================

if __name__ == "__main__":
    curves = run_fit_rating_curves()

    print("\nQuick estimate test (w_avg=1.5m, w_max=2.0m, w_min=1.0m):")
    for station_id in ['BAD01', 'THA01']:
        est = estimate_discharge(station_id, 1.5, 2.0, 1.0, curves)
        print(f"\n  {station_id}:")
        print(f"    q_avg ≈ {est['q_avg']:.2f} m³/s")
        print(f"    q_max ≈ {est['q_max']:.2f} m³/s")
        print(f"    q_min ≈ {est['q_min']:.2f} m³/s")

    print("\nFlood-level estimate (w_avg=4.0m — near minor flood at BAD01):")
    est = estimate_discharge('BAD01', 4.0, 4.2, 3.8, curves)
    print(f"  q_avg ≈ {est['q_avg']:.2f} m³/s")
    print(f"  q_max ≈ {est['q_max']:.2f} m³/s")