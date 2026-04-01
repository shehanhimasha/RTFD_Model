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
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        resp = requests.get(DMC_URL, timeout=30, verify=False)
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
        resp = requests.get(url, timeout=60, stream=True, verify=False)
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
            time_str_parsed = None
            time_ut_parsed = None
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text()
                if text:
                    time_match = re.search(r"DATE\s*:\s*([\d-]+[A-Za-z]+[\d-]+)\s+TIME\s*:\s*([\d:]+\s*[APMapm]+)", text)
                    if time_match:
                        date_str = time_match.group(1).strip()
                        time_str = time_match.group(2).strip()
                        try:
                            time_dt = datetime.strptime(f"{date_str} {time_str}", "%d-%b-%Y %I:%M %p")
                            time_str_parsed = time_dt.strftime("%Y-%m-%d %H:%M:%S")
                            time_ut_parsed = time_dt.timestamp()
                        except ValueError:
                            pass

            if not time_str_parsed:
                time_dt = datetime.utcnow()
                time_str_parsed = time_dt.strftime("%Y-%m-%d %H:%M:%S")
                time_ut_parsed = time_dt.timestamp()
                log.warning("Could not parse time from header; using UTC now.")

            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    
                    for row in table[1:]:
                        if not row or len(list(row)) < 4:
                            continue
                        
                        canonical = None
                        idx_station: int = -1
                        row_items = list(row)
                        for idx, cell in enumerate(row_items[:5]):
                            canonical = match_station(str(cell) if cell is not None else "")
                            if canonical:
                                idx_station = int(idx)
                                break
                        
                        if canonical is None or idx_station == -1:
                            continue
                        
                        prev_wl = parse_float(row_items[idx_station + 5]) if (idx_station + 5) < len(row_items) else None
                        curr_wl = parse_float(row_items[idx_station + 6]) if (idx_station + 6) < len(row_items) else None
                        remark = str(row_items[idx_station + 7]).strip() if (idx_station + 7) < len(row_items) and row_items[idx_station + 7] is not None else ""
                        rising_raw = str(row_items[idx_station + 8]).strip().lower() if (idx_station + 8) < len(row_items) and row_items[idx_station + 8] is not None else ""
                        
                        rain_parts = [str(x).strip() for x in row_items[idx_station + 9:] if x is not None]
                        rain_mm_str = "".join(rain_parts).replace(" ", "").replace("\n", "")
                        rain_mm = parse_float(rain_mm_str) if rain_mm_str else None
                        
                        if "ris" in rising_raw:
                            rising = "Rising"
                        elif "fall" in rising_raw:
                            rising = "Falling"
                        else:
                            rising = None

                        record = {
                            "gauging_station_name": canonical,
                            "time_str": time_str_parsed,
                            "time_ut": float(time_ut_parsed) if time_ut_parsed is not None else 0.0,
                            "previous_water_level": prev_wl,
                            "current_water_level": curr_wl,
                            "remarks": remark if remark else "Normal",
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
