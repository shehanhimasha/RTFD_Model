# =============================================================================
#
# Run once to back-fill the last 7 days into history_store.json
# from the processed CSV data.
#
# Run: python -m src.pipeline.seed_history
# =============================================================================

import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from src.pipeline.history_store import (
    load_history, update_history, save_history
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Path to processed CSVs
PROCESSED_DIR = Path("data/processed/Gin river")

STATION_FILES = {
    'BAD01': 'Flood data BAD01.csv',
    'THA01': 'Flood data THA01.csv',
}

UPSTREAM_STATIONS = ['DEN01', 'LAN01', 'THA01', 'UDU01', 'MAK01']


def run_seed():
    logger.info("Seeding history store from processed CSV...")

    history = load_history()

    # Get last 7 days
    today = date.today()
    last_7_days = [
        str(today - timedelta(days=i))
        for i in range(7, 0, -1)
    ]

    logger.info(f"Back-filling dates: {last_7_days}")

    for station_id, filename in STATION_FILES.items():
        df = pd.read_csv(PROCESSED_DIR / filename)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

        for day in last_7_days:
            row = df[df['date_str'] == day]

            if row.empty:
                logger.warning(f"  {station_id} {day} — no data, skipping")
                continue

            row = row.iloc[0]

            # Build upstream rainfall — all zeros from CSV
            # (CSV only has own station rainfall, not upstream)
            # Real upstream values will fill in naturally going forward
            upstream = {s: 0.0 for s in UPSTREAM_STATIONS}

            daily_record = {
                'date':        day,
                'w_avg':       float(row.get('w_avg', 0.0)),
                'w_max':       float(row.get('w_max', 0.0)),
                'w_min':       float(row.get('w_min', 0.0)),
                'rainfall_mm': float(row.get('rainfall_mm', 0.0)),
                **{f'rainfall_{s}': upstream[s] for s in UPSTREAM_STATIONS},
            }

            history = update_history(history, station_id, daily_record)
            logger.info(
                f"  {station_id} {day}: "
                f"w_avg={daily_record['w_avg']:.3f}m "
                f"rain={daily_record['rainfall_mm']:.1f}mm"
            )

    save_history(history)
    logger.info("History store seeded successfully")


if __name__ == "__main__":
    run_seed()