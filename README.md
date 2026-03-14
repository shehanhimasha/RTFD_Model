# RTFD Model — Real-Time Flood Detection Pipeline

Automated disaster prediction pipeline for the **Gin Ganga (Gin River)** basin — Baddegama & Thawalama stations. Fully automated via GitHub Actions.

---

## Architecture

```
DMC PDF (every 10 min)      OpenWeatherMap (every hr)      ArcGIS REST API (every hr)
       │                              │                              │
 pdf_watcher.py               weather_fetcher.py            river_fetcher.py
       │                              │                              │
data/dmc_data.json         data/weather_data.json         data/river_data.json
       └──────────────────────────────┴──────────────────────────────┘
                                      │
                                 pipeline.py
                          (merge + XGBoost + LSTM)
                                      │
                           data/prediction.json
```

---

## File Structure

```
/
├── pdf_watcher.py          # Source 1: DMC PDF scrape → dmc_data.json
├── weather_fetcher.py      # Source 2: OpenWeatherMap → weather_data.json
├── river_fetcher.py        # Source 3: ArcGIS Gin River → river_data.json
├── pipeline.py             # Merge all 3 + run model → prediction.json
├── last_seen_pdf.txt       # Tracks last processed DMC PDF filename
├── requirements.txt
│
├── data/
│   ├── dmc_data.json       # Source 1 latest output
│   ├── weather_data.json   # Source 2 latest output
│   ├── river_data.json     # Source 3 latest output
│   └── prediction.json     # Final model prediction
│
├── models/                 # 
│   ├── xgboost_model.pkl   # Trained XGBoost model (joblib)
│   └── lstm_model.keras    # Trained LSTM model (TensorFlow/Keras)
│
└── .github/workflows/
    ├── pdf_watcher.yml     # cron: every 10 min
    ├── weather_fetch.yml   # cron: every hour (:00)
    └── river_fetch.yml     # cron: every hour (:30)
```

---

## Setup

### 1. GitHub Secret
Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|------|-------|
| `OPENWEATHER_API_KEY` | OpenWeatherMap free-tier API key |

### 2. Enable GitHub Actions
Go to the **Actions** tab and enable workflows.  
GitHub Actions will run automatically on the cron schedule.

### 3. Run Locally (optional)
```bash
pip install -r requirements.txt

# Source 2 (needs API key)
set OPENWEATHER_API_KEY=your_key_here   # Windows
export OPENWEATHER_API_KEY=your_key_here  # Linux/Mac
python weather_fetcher.py

# Source 3
python river_fetcher.py

# Source 1
python pdf_watcher.py

# Full pipeline
python pipeline.py
```

---

## ML Models

Place your trained models in the `models/` directory:
- `models/xgboost_model.pkl` — trained with `joblib.dump(model, path)`
- `models/lstm_model.keras` — saved with `model.save(path)`

**If models are absent**, `pipeline.py` falls back to a rule-based threshold predictor so the pipeline never crashes.

Feature vector (per station):
| Feature | Source |
|---|---|
| `dmc_current_wl` | DMC PDF |
| `dmc_previous_wl` | DMC PDF |
| `dmc_rainfall_mm` | DMC PDF |
| `dmc_rising` | DMC PDF (1=Rising, 0=Falling) |
| `owm_rainfall_1h` | OpenWeatherMap |
| `owm_humidity` | OpenWeatherMap |
| `owm_clouds_pct` | OpenWeatherMap |
| `arc_water_level_m` | ArcGIS |
| `arc_rainfall_mm_hr` | ArcGIS |

---

## ArcGIS Note

`river_fetcher.py` targets the feature service behind the [Gin River Dashboard](https://www.arcgis.com/apps/dashboards/2cffe83c9ff5497d97375498bdf3ff38).  
If readings come back empty, open the dashboard, press **F12 → Network → filter "FeatureServer"**, copy the actual query URL, and update `ARCGIS_SERVICE_URL` at the top of `river_fetcher.py`.

---

## Failure Handling

| Scenario | Behaviour |
|---|---|
| DMC site unreachable | Log warning, keep existing `dmc_data.json` |
| Same PDF seen twice | Skip silently (tracked by `last_seen_pdf.txt`) |
| PDF parse yields 0 records | Keep existing JSON, don't overwrite |
| OWM fetch fails for one station | Log warning, omit that station only |
| All OWM fetches fail | Log error, keep existing `weather_data.json` |
| ArcGIS returns 0 features | Log warning, write empty-stations JSON |
| Any source JSON missing in pipeline | Load last committed version; continue |
| Models not found | Rule-based threshold fallback |

---

## Stations

| station_id | station_name | lat | lon | River |
|---|---|---|---|---|
| BAD01 | Baddegama | 6.17 | 80.18 | Gin Ganga |
| THA01 | Thawalama | 6.34 | 80.33 | Gin Ganga |
| DEN01 | Deniyaya | 6.35 | 80.56 | Gin Ganga |
| LAN01 | Lankagama | 6.37 | 80.47 | Gin Ganga |
| MAK01 | Makurugoda | 6.18 | 80.17 | Gin Ganga |
| UDU01 | Udugama | 6.23 | 80.32 | Gin Ganga |

*(OWM fetches all 6; DMC + ArcGIS focus on Baddegama & Thawalama only)*
