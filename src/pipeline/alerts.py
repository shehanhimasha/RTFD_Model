import os
import json
import uuid
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from pathlib import Path
from config.settings import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlertEngine")

# Configuration loaded from settings.py / .env
DOTNET_WEBHOOK_URL = config.DOTNET_WEBHOOK_URL
INTERNAL_API_KEY = config.INTERNAL_API_KEY

class AlertGenerator:
    def __init__(self, data_path: str = "data/prediction.json"):
        self.data_path = Path(data_path)
        
        # Area code mappings to readable names and districts (Expand this based on your stations)
        self.station_mappings = {
            "BAD01": {"area": "Baddegama Region", "district": "Galle"},
            "THA01": {"area": "Tawalama Region", "district": "Galle"}
        }

    def load_predictions(self):
        """Load the latest predictions from the JSON file."""
        if not self.data_path.exists():
            logger.error(f"Prediction file not found at {self.data_path}")
            return None
            
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def determine_severity_and_messages(self, station_code, prediction_data):
        """Map the predicted flood category to standard alert severities and messages."""
        category = prediction_data.get("flood_category", "Normal")
        water_level = prediction_data.get("current_water_level_m", 0)
        rainfall = prediction_data.get("rainfall_today_mm", 0)
        rising_flag = prediction_data.get("rising_flag", 0)
        flood_timing = prediction_data.get("flood_timing", {})
        
        station_info = self.station_mappings.get(station_code, {})
        area_name = station_info.get("area", f"Region {station_code}")
        district = station_info.get("district", "Unknown")
        
        timing_str = flood_timing.get("note", "")
        if timing_str:
            timing_str = f" {timing_str}."
            
        trend_str = "rising" if rising_flag == 1 else "falling" if rising_flag == -1 else "steady"

        # Default empty payload if no alert is needed
        alert_payload = None

        if category == "Major Flood":
            actions = [
                "Evacuate to higher ground immediately.",
                "Do not attempt to cross flooded roads or bridges."
            ]
            if rising_flag == -1:
                actions = ["Water levels are high but receding. Remain extremely cautious."]
                
            alert_payload = {
                "severity_level": "CRITICAL",
                "event_type": "FLASH_FLOOD",
                "title": "Flash Flood Warning",
                "short_message": f"CRITICAL: Severe flooding imminent or occurring in {area_name}.{timing_str}",
                "detailed_message": f"River levels at {area_name} have reached {water_level}m ({trend_str}) with {rainfall}mm of rainfall.{timing_str} Immediate action required.",
                "recommended_action": actions
            }
        elif category == "Minor Flood":
            alert_payload = {
                "severity_level": "HIGH",
                "event_type": "MODERATE_FLOOD",
                "title": "Flood Warning",
                "short_message": f"HIGH: Moderate flood risk in {area_name}.{timing_str}",
                "detailed_message": f"Water levels are at {water_level}m ({trend_str}). Minor flooding is expected in immediate low-lying areas.{timing_str}",
                "recommended_action": [
                    "Prepare to move valuables to higher ground.",
                    "Avoid parking vehicles near river banks."
                ]
            }
        elif category == "Alert":
            alert_payload = {
                "severity_level": "MEDIUM",
                "event_type": "HIGH_WATER_ALERT",
                "title": "Rising Water Alert",
                "short_message": f"MEDIUM: High water levels detected in {area_name}.{timing_str}",
                "detailed_message": f"River levels are increasing (currently {water_level}m, {trend_str}). Situation is being closely monitored.{timing_str}",
                "recommended_action": [
                    "Stay tuned to local weather channels and app updates.",
                    "Prepare emergency kits."
                ]
            }
        elif category == "Normal" and rainfall >= 30.0:
            # Heavy rainfall advisory even when water levels are normal
            alert_payload = {
                "severity_level": "LOW",
                "event_type": "HEAVY_RAIN_ADVISORY",
                "title": "Heavy Rain Advisory",
                "short_message": f"LOW: Heavy rainfall detected in {area_name} ({rainfall}mm). River levels are normal but monitor closely.",
                "detailed_message": f"Significant rainfall of {rainfall}mm recorded today. Current river level is safe at {water_level}m ({trend_str}), but may rise if rain continues.",
                "recommended_action": [
                    "No immediate threat.",
                    "Monitor weather updates in case rainfall continues."
                ]
            }
            
        return alert_payload

    def process_and_send_alerts(self):
        """Main method to check predictions and dispatch alerts."""
        data = self.load_predictions()
        if not data:
            return

        stations = data.get("stations", {})
        alerts_sent = 0

        for station_code, station_data in stations.items():
            alert_info = self.determine_severity_and_messages(station_code, station_data)
            
            if alert_info:
                timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat()
                area_name = self.station_mappings.get(station_code, {}).get("area", station_code)
                district = self.station_mappings.get(station_code, {}).get("district", "Unknown")
                safe_area_name = area_name.replace(" ", "_")
                
                # Construct the full alert payload matching the required standard format
                full_payload = {
                    "alert_id": f"flood_{safe_area_name}_{int(datetime.now(ZoneInfo('Asia/Colombo')).timestamp())}",
                    "confidence": station_data.get("confidence", 0.0),
                    "metrics": {
                        "water_level_m": station_data.get("current_water_level_m", 0.0),
                        "rainfall_mm": station_data.get("rainfall_today_mm", 0.0)
                    },
                    "location": {
                        "name": area_name,
                        "district": district,
                        "station_code": f"GIN_{station_code}"
                    },
                    "created_at": timestamp
                }
                
                # Merge the mapped info (title, severity, messages, recommended_action list) into the payload
                full_payload.update(alert_info)
                
                logger.info(f"Generated {full_payload['severity_level']} alert for {station_code}.")
                
                # Send to Backend
                self.send_to_backend(full_payload)
                alerts_sent += 1
            else:
                logger.info(f"No active alerts needed for {station_code} (Status: Normal).")

        logger.info(f"Alert processing complete. {alerts_sent} alert(s) dispatched to backend.")

    def send_to_backend(self, payload):
        """Execute the HTTP POST payload to the .NET API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": INTERNAL_API_KEY
            }
            
            # Using timeout to prevent hanging
            # If you don't have the .NET backend running yet, this will catch the error
            response = requests.post(DOTNET_WEBHOOK_URL, json=payload, headers=headers, timeout=5)
            
            if response.status_code in (200, 201, 202):
                logger.info(f"Successfully sent alert {payload['alert_id']} to backend.")
            else:
                logger.error(f"Failed to send alert to backend. Status: {response.status_code}, Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error while sending alert to backend: {e}")

if __name__ == "__main__":
    # Test execution
    engine = AlertGenerator(data_path="data/prediction.json")
    engine.process_and_send_alerts()
