import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import requests
from config.settings import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlertEngine")

DOTNET_WEBHOOK_URL = config.DOTNET_WEBHOOK_URL
INTERNAL_API_KEY   = config.INTERNAL_API_KEY

# ── Station metadata ───────────────────────────────────────────────────────────
STATION_META = {
    "BAD01": {"area": "Baddegama Region", "district": "Galle"},
    "THA01": {"area": "Tawalama Region",  "district": "Galle"},
}

# ── Early warning thresholds ───────────────────────────────────────────────────
PROXIMITY_FRACTION   = 0.80   # warn when water reaches 80% of alert threshold
PROB_ALERT_THRESHOLD = 0.05   # P(Alert) > 5%
PROB_MINOR_THRESHOLD = 0.02   # P(Minor Flood) > 2%
PROB_MAJOR_THRESHOLD = 0.01   # P(Major Flood) > 1%
RATE_OF_RISE_THRESHOLD = 0.25 # metres per hour
TIMING_WARNING_HOURS = 6      # warn if flood expected within 6 hours


class AlertGenerator:

    def __init__(self, data_path: str = "data/prediction.json"):
        self.data_path = Path(data_path)

    def load_predictions(self):
        if not self.data_path.exists():
            logger.error(f"Prediction file not found at {self.data_path}")
            return None
        with open(self.data_path, 'r') as f:
            return json.load(f)

    # ==========================================================================
    # TRIGGER 1 — Proximity Warning
    # ==========================================================================

    def check_proximity_warning(self, station_code: str, data: dict) -> dict | None:
        """
        Fires when water level reaches 80% of alert threshold while Normal.
        BAD01: alert=3.50m → warn at 2.80m
        THA01: alert=4.00m → warn at 3.20m
        """
        if data.get("flood_category", "Normal") != "Normal":
            return None

        water_level = data.get("current_water_level_m", 0.0)
        alert_level = data.get("alert_level_m", 0.0)
        area        = STATION_META.get(station_code, {}).get("area", station_code)

        if alert_level <= 0:
            return None

        if water_level < alert_level * PROXIMITY_FRACTION:
            return None

        headroom_m   = round(alert_level - water_level, 2)
        pct_of_alert = round((water_level / alert_level) * 100, 1)

        return {
            "trigger":        "PROXIMITY_WARNING",
            "severity_level": "LOW",
            "event_type":     "APPROACHING_ALERT_LEVEL",
            "title":          "Water Level Approaching Alert Threshold",
            "short_message": (
                f"LOW: Water level at {area} is {pct_of_alert}% of alert threshold. "
                f"{headroom_m}m remaining before alert level."
            ),
            "detailed_message": (
                f"Current water level at {area} is {water_level}m. "
                f"Alert threshold is {alert_level}m — only {headroom_m}m of headroom remains. "
                f"No flood yet but situation requires monitoring."
            ),
            "recommended_action": [
                "Monitor river levels closely over the next few hours.",
                "Prepare emergency kits in case situation escalates.",
                "Avoid parking vehicles or storing valuables near the river bank.",
            ],
        }

    # ==========================================================================
    # TRIGGER 2 — Probability Warning
    # FIX (Low): removed unused max_flood_prob variable
    # ==========================================================================

    def check_probability_warning(self, station_code: str, data: dict) -> dict | None:
        """
        Reads the model's internal class probabilities.
        Fires when non-Normal probability exceeds threshold even while
        flood_category is still Normal — often hours before the category flips.

        FIX: Removed unused max_flood_prob = max(p_alert, p_minor, p_major).
        """
        if data.get("flood_category", "Normal") != "Normal":
            return None

        probs       = data.get("probabilities", {})
        p_alert     = probs.get("Alert", 0.0)
        p_minor     = probs.get("Minor Flood", 0.0)
        p_major     = probs.get("Major Flood", 0.0)
        water_level = data.get("current_water_level_m", 0.0)
        area        = STATION_META.get(station_code, {}).get("area", station_code)

        # FIX: max_flood_prob was computed here but never used — removed

        if p_major >= PROB_MAJOR_THRESHOLD:
            return {
                "trigger":        "PROBABILITY_WARNING",
                "severity_level": "MEDIUM",
                "event_type":     "ELEVATED_MAJOR_FLOOD_PROBABILITY",
                "title":          "Elevated Major Flood Probability Detected",
                "short_message": (
                    f"MEDIUM: Model detecting major flood risk at {area} "
                    f"(P={p_major:.1%}). Water level {water_level}m."
                ),
                "detailed_message": (
                    f"The flood model has assigned a {p_major:.1%} probability of Major Flood "
                    f"at {area}. Water level is currently {water_level}m — no threshold crossed "
                    f"yet — but upstream patterns suggest conditions are developing."
                ),
                "recommended_action": [
                    "Begin monitoring river levels every 30 minutes.",
                    "Prepare evacuation plans for low-lying areas.",
                    "Alert local authorities and emergency services.",
                ],
            }

        if p_minor >= PROB_MINOR_THRESHOLD:
            return {
                "trigger":        "PROBABILITY_WARNING",
                "severity_level": "LOW",
                "event_type":     "ELEVATED_MINOR_FLOOD_PROBABILITY",
                "title":          "Elevated Flood Probability Detected",
                "short_message": (
                    f"LOW: Model detecting flood risk at {area} "
                    f"(P(Minor)={p_minor:.1%}). Water level {water_level}m."
                ),
                "detailed_message": (
                    f"The model has assigned a {p_minor:.1%} probability of Minor Flood "
                    f"at {area}. Current level of {water_level}m is within normal range "
                    f"but upstream conditions suggest this may change."
                ),
                "recommended_action": [
                    "Stay informed via app updates.",
                    "Move valuables away from river-adjacent areas as a precaution.",
                ],
            }

        if p_alert >= PROB_ALERT_THRESHOLD:
            return {
                "trigger":        "PROBABILITY_WARNING",
                "severity_level": "LOW",
                "event_type":     "ELEVATED_ALERT_PROBABILITY",
                "title":          "Rising Water Risk Detected",
                "short_message": (
                    f"LOW: Model detecting rising risk at {area} "
                    f"(P(Alert)={p_alert:.1%}). Water level {water_level}m."
                ),
                "detailed_message": (
                    f"Flood model probability for Alert at {area} has risen to {p_alert:.1%}. "
                    f"Water level is {water_level}m. No immediate danger but conditions are developing."
                ),
                "recommended_action": [
                    "No immediate action needed.",
                    "Check app for updates over the next few hours.",
                ],
            }

        return None

    # ==========================================================================
    # TRIGGER 3 — Rate of Rise Warning
    # FIX (Medium): no longer depends on flood_timing.rise_rate_m_per_hour
    # Now reads w_avg_delta_m directly from prediction.json (always populated)
    # ==========================================================================

    def check_rate_of_rise_warning(self, station_code: str, data: dict) -> dict | None:
        """
        Fires when the river is rising rapidly while in Normal category.

        FIX: Previously read rise_rate_m_per_hour from flood_timing, which is
        only populated when flood_label >= 1. This meant the rate-of-rise
        early warning never fired for Normal stations.

        Now reads current_rise_rate_m_per_hour directly from prediction.json.
        This is computed from the last two accumulator readings and the
        actual time between pipeline runs.
        """
        if data.get("flood_category", "Normal") != "Normal":
            return None

        rise_per_hour = data.get("current_rise_rate_m_per_hour", 0.0)
        if rise_per_hour <= 0:
            return None

        if rise_per_hour < RATE_OF_RISE_THRESHOLD:
            return None

        water_level = data.get("current_water_level_m", 0.0)
        w_avg       = data.get("w_avg_m", 0.0)
        alert_level = data.get("alert_level_m", 0.0)
        area        = STATION_META.get(station_code, {}).get("area", station_code)
        projected_1h = round(w_avg + rise_per_hour, 2)
        projected_3h = round(w_avg + (rise_per_hour * 3), 2)

        return {
            "trigger":        "RATE_OF_RISE_WARNING",
            "severity_level": "MEDIUM",
            "event_type":     "RAPID_WATER_LEVEL_RISE",
            "title":          "Rapid River Rise Detected",
            "short_message": (
                f"MEDIUM: River at {area} rising at ~{rise_per_hour:.2f}m/hour. "
                f"Current level {water_level}m. Alert level {alert_level}m."
            ),
            "detailed_message": (
                f"Water level at {area} is rising at approximately {rise_per_hour:.2f}m/hour. "
                f"Current level is {water_level}m. Projected: {projected_1h}m in 1 hour, "
                f"{projected_3h}m in 3 hours. Alert threshold is {alert_level}m."
            ),
            "recommended_action": [
                "Monitor river levels every 15-30 minutes.",
                "Be ready to move to higher ground at short notice.",
                "Do not attempt to cross rivers or low-lying roads.",
            ],
        }

    # ==========================================================================
    # TRIGGER 4 — Timing Warning
    # FIX (Medium): now fires for Normal stations because pipeline.py
    # populates flood_timing for all categories when river is rising
    # ==========================================================================

    def check_timing_warning(self, station_code: str, data: dict) -> dict | None:
        """
        Fires when the pipeline estimates flood threshold crossing within N hours.

        FIX: Previously never fired for Normal stations because pipeline.py
        returned empty flood_timing when flood_label == 0. The pipeline fix
        (removing the early return for Normal) means this now fires correctly
        whenever the river is rising toward a threshold — regardless of category.
        """
        flood_timing = data.get("flood_timing", {})
        hours        = flood_timing.get("estimated_hours_to_next_threshold")
        next_label   = flood_timing.get("next_threshold_label", "")
        next_level   = flood_timing.get("next_threshold_m", 0.0)
        note         = flood_timing.get("note", "")
        water_level  = data.get("current_water_level_m", 0.0)
        area         = STATION_META.get(station_code, {}).get("area", station_code)

        if hours is None or hours > TIMING_WARNING_HOURS:
            return None

        if hours <= 1:
            severity, urgency = "HIGH",   "IMMINENT"
        elif hours <= 3:
            severity, urgency = "MEDIUM", "SOON"
        else:
            severity, urgency = "LOW",    "DEVELOPING"

        return {
            "trigger":        "TIMING_WARNING",
            "severity_level": severity,
            "event_type":     f"FLOOD_EXPECTED_{urgency}",
            "title":          f"{next_label} Expected in ~{hours} Hour(s)",
            "short_message": (
                f"{severity}: {note} Area: {area}. "
                f"Current water level: {water_level}m."
            ),
            "detailed_message": (
                f"{note} Current water level at {area} is {water_level}m. "
                f"{next_label} threshold is {next_level}m. "
                f"{'Immediate action required.' if hours <= 1 else 'Prepare now.'}"
            ),
            "recommended_action": self._timing_actions(hours, next_label),
        }

    def _timing_actions(self, hours: int, next_label: str) -> list:
        if hours <= 1:
            return [
                "Move to higher ground immediately if in a flood-prone area.",
                "Do not wait — evacuate now.",
                "Call emergency services if assistance is needed.",
            ]
        elif hours <= 3:
            return [
                "Begin moving valuables and vehicles to higher ground.",
                "Prepare emergency kit: documents, medicine, food, water.",
                "Know your evacuation route before it is needed.",
            ]
        return [
            "Monitor the situation closely over the next few hours.",
            "Prepare emergency kit as a precaution.",
            "Stay tuned to app updates and local weather broadcasts.",
        ]

    # ==========================================================================
    # Confirmed flood alerts — last line when threshold already crossed
    # ==========================================================================

    def check_confirmed_alert(self, station_code: str, data: dict) -> dict | None:
        category    = data.get("flood_category", "Normal")
        water_level = data.get("current_water_level_m", 0.0)
        rainfall    = data.get("rainfall_today_mm", 0.0)
        rising_flag = data.get("rising_flag", 0)
        flood_timing = data.get("flood_timing", {})
        area        = STATION_META.get(station_code, {}).get("area", station_code)

        timing_str = flood_timing.get("note", "")
        if timing_str:
            timing_str = f" {timing_str}."

        trend_str = (
            "rising"  if rising_flag == 1  else
            "falling" if rising_flag == -1 else
            "steady"
        )

        if category == "Major Flood":
            actions = (
                ["Water levels are high but receding. Remain extremely cautious."]
                if rising_flag == -1 else
                [
                    "Evacuate to higher ground immediately.",
                    "Do not attempt to cross flooded roads or bridges.",
                ]
            )
            return {
                "trigger":        "CONFIRMED_FLOOD",
                "severity_level": "CRITICAL",
                "event_type":     "FLASH_FLOOD",
                "title":          "Flash Flood Warning",
                "short_message":  f"CRITICAL: Severe flooding at {area}.{timing_str}",
                "detailed_message": (
                    f"River levels at {area} have reached {water_level}m ({trend_str}) "
                    f"with {rainfall}mm of rainfall today.{timing_str}"
                ),
                "recommended_action": actions,
            }

        if category == "Minor Flood":
            return {
                "trigger":        "CONFIRMED_FLOOD",
                "severity_level": "HIGH",
                "event_type":     "MODERATE_FLOOD",
                "title":          "Flood Warning",
                "short_message":  f"HIGH: Moderate flooding at {area}.{timing_str}",
                "detailed_message": (
                    f"Water levels at {area} are at {water_level}m ({trend_str}). "
                    f"Minor flooding in low-lying areas.{timing_str}"
                ),
                "recommended_action": [
                    "Move valuables to higher ground.",
                    "Avoid parking vehicles near river banks.",
                ],
            }

        if category == "Alert":
            return {
                "trigger":        "CONFIRMED_FLOOD",
                "severity_level": "MEDIUM",
                "event_type":     "HIGH_WATER_ALERT",
                "title":          "Rising Water Alert",
                "short_message":  f"MEDIUM: High water levels at {area}.{timing_str}",
                "detailed_message": (
                    f"River levels at {area} are at {water_level}m ({trend_str}). "
                    f"Situation is being monitored.{timing_str}"
                ),
                "recommended_action": [
                    "Stay informed via local weather channels and app updates.",
                    "Prepare emergency kits.",
                ],
            }

        if category == "Normal" and rainfall >= 30.0:
            return {
                "trigger":        "HEAVY_RAIN",
                "severity_level": "LOW",
                "event_type":     "HEAVY_RAIN_ADVISORY",
                "title":          "Heavy Rain Advisory",
                "short_message": (
                    f"LOW: Heavy rainfall at {area} ({rainfall}mm today). "
                    f"River level currently normal."
                ),
                "detailed_message": (
                    f"Significant rainfall of {rainfall}mm recorded today. "
                    f"Current river level is {water_level}m — within safe range — "
                    f"but may rise if rain continues."
                ),
                "recommended_action": [
                    "No immediate threat.",
                    "Monitor weather updates in case rainfall continues.",
                ],
            }

        return None

    # ==========================================================================
    # Main processing
    # ==========================================================================

    def get_all_alerts(self, station_code: str, data: dict) -> list:
        alerts = []

        confirmed = self.check_confirmed_alert(station_code, data)
        if confirmed:
            alerts.append(confirmed)
            # Still check timing even in confirmed flood state
            # so "X hours to Major Flood" fires during Minor Flood
            timing = self.check_timing_warning(station_code, data)
            if timing:
                alerts.append(timing)
            return alerts

        # No confirmed flood — run all early warning checks in priority order
        for check in [
            self.check_timing_warning,
            self.check_rate_of_rise_warning,
            self.check_probability_warning,
            self.check_proximity_warning,
        ]:
            result = check(station_code, data)
            if result:
                alerts.append(result)

        return alerts

    def process_and_send_alerts(self):
        data = self.load_predictions()
        if not data:
            return

        stations    = data.get("stations", {})
        alerts_sent = 0

        for station_code, station_data in stations.items():
            alerts = self.get_all_alerts(station_code, station_data)

            if not alerts:
                logger.info(
                    f"No alerts for {station_code} "
                    f"(Status: {station_data.get('flood_category', 'Normal')})."
                )
                continue

            for alert_info in alerts:
                self._dispatch(station_code, station_data, alert_info)
                alerts_sent += 1

        logger.info(f"Alert processing complete. {alerts_sent} alert(s) dispatched.")

    def _dispatch(self, station_code: str, station_data: dict, alert_info: dict):
        """
        Build and send one alert payload to the backend.

        FIX (High): alert_id now includes trigger name to prevent collision
        when multiple alerts fire for the same station in one run.
        Previously: flood_Baddegama_Region_1747123456 (same for all triggers)
        Now:        flood_Baddegama_Region_PROXIMITY_WARNING_1747123456
        """
        now       = datetime.now(ZoneInfo("Asia/Colombo"))
        area_meta = STATION_META.get(station_code, {})
        area_name = area_meta.get("area", station_code)
        district  = area_meta.get("district", "Unknown")
        safe_name = area_name.replace(" ", "_")
        trigger   = alert_info.get("trigger", "UNKNOWN")

        payload = {
            # FIX: trigger name included — no more same-second collisions
            "alert_id":   f"flood_{station_code}_{safe_name}_{trigger}_{int(now.timestamp())}",
            "confidence": station_data.get("confidence", 0.0),
            "trigger":    trigger,
            "metrics": {
                "water_level_m":  station_data.get("current_water_level_m", 0.0),
                "rainfall_mm":    station_data.get("rainfall_today_mm", 0.0),
                "w_avg_m":        station_data.get("w_avg_m", 0.0),
                "w_max_m":        station_data.get("w_max_m", 0.0),
                "w_avg_delta_m":  station_data.get("w_avg_delta_m", 0.0),
                "current_rise_rate_m_per_hour": station_data.get(
                    "current_rise_rate_m_per_hour", 0.0
                ),
                "rising_flag":    station_data.get("rising_flag", 0),
            },
            "location": {
                "name":         area_name,
                "district":     district,
                "station_code": f"GIN_{station_code}",
            },
            "created_at": now.isoformat(),
        }
        payload.update({k: v for k, v in alert_info.items() if k != "trigger"})

        logger.info(
            f"[{trigger}] {alert_info['severity_level']} alert for {station_code}: "
            f"{alert_info['short_message']}"
        )
        self.send_to_backend(payload)

    def send_to_backend(self, payload: dict):
        try:
            response = requests.post(
                DOTNET_WEBHOOK_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": INTERNAL_API_KEY,
                },
                timeout=5,
            )
            if response.status_code in (200, 201, 202):
                logger.info(f"Sent alert {payload['alert_id']} successfully.")
            else:
                logger.error(
                    f"Backend error {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error: {e}")


if __name__ == "__main__":
    engine = AlertGenerator(data_path="data/prediction.json")
    engine.process_and_send_alerts()