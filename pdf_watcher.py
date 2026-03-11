"""
SOURCE 1 — DMC Rainfall PDF Watcher
Polls the DMC website every 10 minutes (via GitHub Actions schedule).
Downloads the newest rainfall PDF, extracts data for Baddegama & Thawalama
(Gin Ganga basin only), and saves results to data/dmc_data.json.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pdfplumber
import requests
from bs4 import BeautifulSoup

# ── Configuration ────────────────────────────────────────────────────────────

DMC_URL = (
    "https://www.dmc.gov.lk/index.php"
    "?option=com_dmcreports&view=reports&Itemid=277"
    "&report_type_id=6&lang=en"
)
BASE_URL = "https://www.dmc.gov.lk"
LAST_SEEN_FILE = Path("last_seen_pdf.txt")
OUTPUT_FILE = Path("data/dmc_data.json")

# Only keep records for these stations in Gin Ganga
TARGET_STATIONS = {"baddegama", "thawalama"}

# Map loose spellings → canonical name
STATION_ALIASES = {
    "baddegama": "Baddegama",
    "badde gama": "Baddegama",
    "thawalama": "Thawalama",
    "thawalame": "Thawalama",
    "thawalamau": "Thawalama",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [PDF] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_last_seen() -> str:
    """Return the last processed PDF filename, or '' if none."""
    if LAST_SEEN_FILE.exists():
        return LAST_SEEN_FILE.read_text(encoding="utf-8").strip()
    return ""


def save_last_seen(filename: str) -> None:
    LAST_SEEN_FILE.write_text(filename.strip(), encoding="utf-8")


def get_latest_pdf_link() -> tuple[str, str] | None:
    """
    Scrape the DMC reports page and return (pdf_url, pdf_filename).
    Returns None if nothing is found or on network error.
    """
    try:
        resp = requests.get(DMC_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to fetch DMC page: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # PDFs are listed as <a href="...pdf..."> links
    pdf_links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.lower().endswith(".pdf") or ".pdf" in href.lower():
            pdf_links.append(href)

    if not pdf_links:
        log.warning("No PDF links found on DMC page.")
        return None

    # Take the first link (newest report typically listed first)
    href = pdf_links[0]
    if not href.startswith("http"):
        href = BASE_URL + "/" + href.lstrip("/")

    filename = href.split("/")[-1].split("?")[0]
    log.info("Latest PDF found: %s", filename)
    return href, filename


def download_pdf(url: str, dest: Path) -> bool:
    """Download the PDF to dest. Returns True on success."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
        log.info("Downloaded PDF to %s (%d bytes)", dest, dest.stat().st_size)
        return True
    except requests.RequestException as exc:
        log.error("Failed to download PDF: %s", exc)
        return False


def parse_float(value: str) -> float | None:
    """Try to parse a string as float; return None on failure."""
    try:
        return float(str(value).strip().replace(",", "."))
    except (ValueError, TypeError):
        return None


def match_station(name: str) -> str | None:
    """
    Return the canonical station name if the cell value matches a target
    station, otherwise return None.
    """
    clean = str(name).strip().lower()
    for alias, canonical in STATION_ALIASES.items():
        if alias in clean:
            return canonical
    return None


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract flood data rows for target stations from the PDF table.
    Returns a list of records in the required schema.
    """
    records = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # Determine if this table has the expected columns
                    # Expected: Station | Time | Previous WL | Current WL | Remarks | Rising/Falling | Rainfall
                    header = [str(c).strip().lower() for c in (table[0] or [])]
                    # Try to locate relevant column indices by keyword
                    col_station = next(
                        (i for i, h in enumerate(header) if "station" in h or "gauging" in h),
                        0,
                    )
                    col_time = next(
                        (i for i, h in enumerate(header) if "time" in h or "date" in h),
                        1,
                    )
                    col_prev = next(
                        (i for i, h in enumerate(header) if "prev" in h),
                        2,
                    )
                    col_curr = next(
                        (i for i, h in enumerate(header) if "curr" in h),
                        3,
                    )
                    col_remark = next(
                        (i for i, h in enumerate(header) if "remark" in h),
                        4,
                    )
                    col_rf = next(
                        (i for i, h in enumerate(header)
                         if "rising" in h or "fall" in h or "trend" in h),
                        5,
                    )
                    col_rain = next(
                        (i for i, h in enumerate(header) if "rain" in h or "mm" in h),
                        6,
                    )

                    for row in table[1:]:
                        if not row or len(row) < 4:
                            continue
                        station_raw = row[col_station] if col_station < len(row) else ""
                        canonical = match_station(station_raw or "")
                        if canonical is None:
                            continue

                        # Parse time string
                        time_raw = str(row[col_time]).strip() if col_time < len(row) else ""
                        time_dt = None
                        for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M"):
                            try:
                                time_dt = datetime.strptime(time_raw, fmt)
                                break
                            except ValueError:
                                pass
                        if time_dt is None:
                            # Use current time as fallback
                            time_dt = datetime.utcnow()
                            log.warning("Could not parse time '%s'; using UTC now.", time_raw)

                        time_str = time_dt.strftime("%Y-%m-%d %H:%M:%S")
                        time_ut = time_dt.timestamp()

                        prev_wl = parse_float(row[col_prev]) if col_prev < len(row) else None
                        curr_wl = parse_float(row[col_curr]) if col_curr < len(row) else None
                        remark = str(row[col_remark]).strip() if col_remark < len(row) else ""
                        rising_raw = str(row[col_rf]).strip().lower() if col_rf < len(row) else ""
                        rain_mm = parse_float(row[col_rain]) if col_rain < len(row) else None

                        # Normalise rising_or_falling
                        if "ris" in rising_raw:
                            rising = "Rising"
                        elif "fall" in rising_raw:
                            rising = "Falling"
                        elif "steady" in rising_raw or "stable" in rising_raw:
                            rising = "Steady"
                        else:
                            rising = rising_raw.title() or "Unknown"

                        record = {
                            "gauging_station_name": canonical,
                            "time_str": time_str,
                            "time_ut": float(time_ut),
                            "previous_water_level": prev_wl,
                            "current_water_level": curr_wl,
                            "remarks": remark if remark else None,
                            "rising_or_falling": rising,
                            "rainfall_mm": rain_mm,
                        }
                        records.append(record)

    except Exception as exc:
        log.error("Error parsing PDF %s: %s", pdf_path, exc)

    return records


def save_output(records: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing records if any
    existing_records = []
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
                existing_records = json.load(fh)
                if not isinstance(existing_records, list):
                    existing_records = []
        except Exception as exc:
            log.warning("Could not read existing %s: %s", OUTPUT_FILE, exc)
            
    # Append the new records
    existing_records.extend(records)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(existing_records, fh, indent=2, ensure_ascii=False)
    log.info("Saved %d new records to %s (Total: %d)", len(records), OUTPUT_FILE, len(existing_records))


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    log.info("=== DMC PDF Watcher starting ===")

    result = get_latest_pdf_link()
    if result is None:
        log.warning("Could not retrieve PDF link. Exiting without changes.")
        return

    pdf_url, pdf_filename = result
    last_seen = load_last_seen()

    if pdf_filename == last_seen:
        log.info("PDF '%s' already processed. Nothing to do.", pdf_filename)
        return

    tmp_pdf = Path(f"rainfall_pdf/{pdf_filename}")
    if not download_pdf(pdf_url, tmp_pdf):
        log.error("Download failed. Keeping existing dmc_data.json unchanged.")
        return

    records = parse_pdf(tmp_pdf)

    if not records:
        log.warning("No target-station records extracted from PDF. Not overwriting output.")
        tmp_pdf.unlink(missing_ok=True)
        return

    save_output(records)
    save_last_seen(pdf_filename)
    log.info("Done. Processed PDF: %s", pdf_filename)

    log.info("Done. Processed PDF: %s", pdf_filename)


if __name__ == "__main__":
    main()
