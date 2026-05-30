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

FLOOD_LABELS = {
    "Alert": {
        "en": "Alert",
        "si": "අවදානම්",
        "ta": "எச்சரிக்கை",
    },
    "Minor Flood": {
        "en": "Minor Flood",
        "si": "කුඩා ගංවතුර",
        "ta": "சிறிய வெள்ளம்",
    },
    "Major Flood": {
        "en": "Major Flood",
        "si": "විශාල ගංවතුර",
        "ta": "பெரிய வெள்ளம்",
    },
}

TREND_LABELS = {
    "rising": {
        "en": "rising",
        "si": "ඉහළ යමින්",
        "ta": "உயர்ந்து வருகிறது",
    },
    "falling": {
        "en": "falling",
        "si": "පහළ යමින්",
        "ta": "குறைந்து வருகிறது",
    },
    "steady": {
        "en": "steady",
        "si": "ස්ථාවරයි",
        "ta": "நிலையாக உள்ளது",
    },
}


class AlertGenerator:

    def __init__(self, data_path: str = "data/prediction.json"):
        self.data_path = Path(data_path)

    def load_predictions(self):
        if not self.data_path.exists():
            logger.error(f"Prediction file not found at {self.data_path}")
            return None
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def _label_for_lang(self, label: str, lang: str) -> str:
        return FLOOD_LABELS.get(label, {}).get(lang, label)

    def _trend_for_lang(self, trend: str, lang: str) -> str:
        return TREND_LABELS.get(trend, {}).get(lang, trend)

    def _render_localized(self, key: str, ctx: dict, lang: str) -> dict:
        area = ctx.get("area", "")

        if key == "PROXIMITY_WARNING":
            pct = ctx.get("pct_of_alert", 0.0)
            headroom = ctx.get("headroom_m", 0.0)
            water = ctx.get("water_level", 0.0)
            alert = ctx.get("alert_level", 0.0)
            if lang == "si":
                return {
                    "title": "ගඟේ ජල මට්ටම අනතුරු සීමාවට ළඟා වෙමින් පවතී",
                    "short_message": (
                        f"{area} ප්‍රදේශයේ ජල මට්ටම දැන් {water}m දක්වා ඉහළ ගොස් ඇත. "
                        f"අනතුරු මට්ටමට ළඟා වීමට තවත් {headroom}m පමණක් ඉතිරිව ඇත."
                    ),
                    "detailed_message": (
                        f"{area} ප්‍රදේශයේ ගඟේ ජල මට්ටම {water}m දක්වා ඉහළ ගොස් ඇති අතර, "
                        f"නිල අනතුරු මට්ටම වන {alert}m ට ළඟා වීමට {headroom}m පමණක් ඉතිරිව ඇත. "
                        "ගංවතුරක් තවම සිදු නොවූවත්, ගඟ ආශ්‍රිත ජනතාව දැඩි අවධානයෙන් සිටිය යුතු බව "
                        "ආපදා කළමනාකරණ මධ්‍යස්ථානය දන්වා සිටී."
                    ),
                    "recommended_actions": [
                        "ඉදිරි පැය කිහිපය තුළ ගඟේ ජල මට්ටම නිතරම නිරීක්ෂණය කරන්න.",
                        "තත්ත්වය නරක වුවහොත් ඉක්මනින් ක්‍රියා කළ හැකිව සිටින්න.",
                        "ගඟ අසල වාහන නවතා නොතබා, වටිනා දේවල් ආරක්ෂිත ස්ථානයකට ගෙන යන්න.",
                    ],
                }
            if lang == "ta":
                return {
                    "title": "எச்சரிக்கை மட்டத்திற்கு நெருக்கமான நீர்மட்டம்",
                    "short_message": (
                        f"குறைவு: {area} பகுதியில் நீர்மட்டம் எச்சரிக்கை மட்டத்தின் {pct}% ஆக உள்ளது. "
                        f"மேலும் {headroom}m மட்டுமே உள்ளது."
                    ),
                    "detailed_message": (
                        f"தற்போது {area} இல் நீர்மட்டம் {water}m. "
                        f"எச்சரிக்கை மட்டம் {alert}m — மேலும் {headroom}m மட்டுமே உள்ளது. "
                        "இன்னும் வெள்ளம் இல்லை, ஆனால் கண்காணிப்பு அவசியம்."
                    ),
                    "recommended_actions": [
                        "அடுத்த சில மணிநேரங்களில் நீர்மட்டத்தை கவனமாக கண்காணிக்கவும்.",
                        "நிலை மோசமானால் தயாராக இருங்கள்.",
                        "ஆற்றுப்பகுதிக்கு அருகே வாகனங்கள் அல்லது மதிப்புள்ள பொருட்களை வைக்க வேண்டாம்.",
                    ],
                }
            return {
                "title": "Water Level Approaching Alert Threshold",
                "short_message": (
                    f"LOW: Water level at {area} is {pct}% of alert threshold. "
                    f"{headroom}m remaining before alert level."
                ),
                "detailed_message": (
                    f"Current water level at {area} is {water}m. "
                    f"Alert threshold is {alert}m — only {headroom}m of headroom remains. "
                    "No flood yet but situation requires monitoring."
                ),
                "recommended_actions": [
                    "Monitor river levels closely over the next few hours.",
                    "Prepare emergency kits in case situation escalates.",
                    "Avoid parking vehicles or storing valuables near the river bank.",
                ],
            }

        if key in {"PROBABILITY_MAJOR", "PROBABILITY_MINOR", "PROBABILITY_ALERT"}:
            prob = ctx.get("probability", 0.0)
            water = ctx.get("water_level", 0.0)
            label = ctx.get("label", "Alert")
            label_local = self._label_for_lang(label, lang)
            if lang == "si":
                return {
                    "title": "ගංවතුර ඇතිවීමේ අවදානමක් හඳුනාගෙන ඇත",
                    "short_message": (
                        f"{area} ප්‍රදේශය සම්බන්ධයෙන් {label_local} අවදානමේ සම්භාවිතාව "
                        f"{prob:.1%} දක්වා ඉහළ ගොස් ඇත. දැනට ජල මට්ටම {water}m යි."
                    ),
                    "detailed_message": (
                        f"ගංවතුර පුරෝකථන ආකෘතිය {area} ප්‍රදේශය සඳහා {label_local} "
                        f"ඇතිවීමේ සම්භාවිතාව {prob:.1%} ලෙස ගණනය කර ඇත. "
                        f"දැනට ගඟේ ජල මට්ටම {water}m වන නමුත්, "
                        "ජලධාරා ඉහළ ප්‍රදේශවල ඇති රටාව ඉදිරි පැය කිහිපය තුළ "
                        "තත්ත්වය වෙනස් විය හැකි බව පෙන්නුම් කරයි."
                    ),
                    "recommended_actions": [
                        "යෙදුම හරහා නිකුත් කෙරෙන නිවේදන දිගටම නිරීක්ෂණය කරන්න.",
                        "ගඟ ආශ්‍රිත ස්ථානවල ඇති වටිනා දේවල් ඉදිරියේදී ආරක්ෂිත ස්ථානයකට ගෙනයාමට සූදානම් වන්න.",
                    ],
                }
            if lang == "ta":
                return {
                    "title": "உயர்ந்த அபாயச் சாத்தியம் கண்டறியப்பட்டது",
                    "short_message": (
                        f"{area} பகுதியில் {label_local} அபாயம் உயர்ந்துள்ளது "
                        f"(P={prob:.1%}). தற்போது {water}m."
                    ),
                    "detailed_message": (
                        f"தற்போது நீர்மட்டம் {water}m ஆக இருந்தாலும், "
                        f"{label_local} அபாயத்திற்கு மாதிரி {prob:.1%} சாத்தியத்தை காட்டுகிறது. "
                        "இது வளர்ந்து வரும் நிலையை குறிக்கிறது."
                    ),
                    "recommended_actions": [
                        "செயலி புதுப்பிப்புகளை கவனமாகப் பாருங்கள்.",
                        "ஆற்றருகிலுள்ள மதிப்புள்ள பொருட்களை பாதுகாப்பாக மாற்றவும்.",
                    ],
                }
            return {
                "title": "Elevated Flood Probability Detected",
                "short_message": (
                    f"LOW: Model detecting elevated {label} risk at {area} "
                    f"(P={prob:.1%}). Water level {water}m."
                ),
                "detailed_message": (
                    f"The model has assigned a {prob:.1%} probability of {label} at {area}. "
                    f"Current water level is {water}m but conditions are developing."
                ),
                "recommended_actions": [
                    "Stay informed via app updates.",
                    "Move valuables away from river-adjacent areas as a precaution.",
                ],
            }

        if key == "RATE_OF_RISE_WARNING":
            rise = ctx.get("rise_rate", 0.0)
            water = ctx.get("water_level", 0.0)
            alert = ctx.get("alert_level", 0.0)
            projected_1h = ctx.get("projected_1h", 0.0)
            projected_3h = ctx.get("projected_3h", 0.0)
            if lang == "si":
                return {
                    "title": "ගඟේ ජල මට්ටම වේගයෙන් ඉහළ යමින් පවතී",
                    "short_message": (
                        f"{area} ප්‍රදේශයේ ගඟේ ජල මට්ටම පැයකට {rise:.2f}m බැගින් ඉහළ යමින් "
                        f"ඇති බව වාර්තා වේ. දැනට {water}m යි. නිල අනතුරු මට්ටම {alert}m යි."
                    ),
                    "detailed_message": (
                        f"{area} ප්‍රදේශයේ ගඟ දැනට පැයකට {rise:.2f}m පමණ වේගයෙන් ඉහළ "
                        f"යමින් ඇත. වත්මන් ජල මට්ටම {water}m වන අතර, මෙම වේගය පවතී නම් "
                        f"පැයකින් {projected_1h}m ට සහ පැය තුනකින් {projected_3h}m ට ළඟා "
                        f"විය හැකිය. නිල අනතුරු මට්ටම {alert}m යි."
                    ),
                    "recommended_actions": [
                        "විනාඩි 15 සිට 30 ත් අතර කාල පරතරයෙන් ගඟේ ජල මට්ටම නිරීක්ෂණය කරන්න.",
                        "ඉහළ ස්ථානයකට ඉක්මනින් ගෙනයා හැකි ළදරුවන්, වයෝවෘද්ධයන් සහ ගෘහ සතුන් සූදානම් කරගන්න.",
                        "ජලය නොකඩවා ගලා යන මාර්ග හෝ පාලම් හරහා ගමන් නොකරන්න.",
                    ],
                }
            if lang == "ta":
                return {
                    "title": "நதி வேகமாக உயர்வது கண்டறியப்பட்டது",
                    "short_message": (
                        f"நடுத்தரம்: {area} பகுதியில் நீர்மட்டம் மணிக்கு {rise:.2f}m உயர்கிறது. "
                        f"தற்போது {water}m. எச்சரிக்கை மட்டம் {alert}m."
                    ),
                    "detailed_message": (
                        f"{area} இல் நீர்மட்டம் மணிக்கு சுமார் {rise:.2f}m உயர்கிறது. "
                        f"தற்போது {water}m. 1 மணி நேரத்தில் {projected_1h}m, 3 மணிநேரத்தில் {projected_3h}m. "
                        f"எச்சரிக்கை மட்டம் {alert}m."
                    ),
                    "recommended_actions": [
                        "15–30 நிமிடங்களுக்கு ஒருமுறை நீர்மட்டத்தை பார்க்கவும்.",
                        "தேவைப்பட்டால் உயர்ந்த இடத்திற்கு செல்ல தயாராக இருக்கவும்.",
                        "தாழ்வான சாலைகளில் செல்ல வேண்டாம்.",
                    ],
                }
            return {
                "title": "Rapid River Rise Detected",
                "short_message": (
                    f"MEDIUM: River at {area} rising at ~{rise:.2f}m/hour. "
                    f"Current level {water}m. Alert level {alert}m."
                ),
                "detailed_message": (
                    f"Water level at {area} is rising at approximately {rise:.2f}m/hour. "
                    f"Current level is {water}m. Projected: {projected_1h}m in 1 hour, "
                    f"{projected_3h}m in 3 hours. Alert threshold is {alert}m."
                ),
                "recommended_actions": [
                    "Monitor river levels every 15-30 minutes.",
                    "Be ready to move to higher ground at short notice.",
                    "Do not attempt to cross rivers or low-lying roads.",
                ],
            }

        if key == "TIMING_WARNING":
            hours = ctx.get("hours", 0)
            next_label = ctx.get("next_label", "Alert")
            next_level = ctx.get("next_level", 0.0)
            water = ctx.get("water_level", 0.0)
            label_local = self._label_for_lang(next_label, lang)
            if lang == "si":
                return {
                    "title": f"{area} ප්‍රදේශයේ {label_local} තත්ත්වයක් ළඟදීම ඇතිවිය හැකිය",
                    "short_message": (
                        f"{area} ප්‍රදේශයේ ජල මට්ටම දැනට {water}m යි. "
                        f"මෙම වේගයෙන් ඉහළ ගියහොත් {label_local} මට්ටම ({next_level}m) "
                        f"ළඟා වීමට ආසන්නයෙන් පැය {hours}ක් ගත විය හැකිය."
                    ),
                    "detailed_message": (
                        f"ගංවතුර නිරීක්ෂණ දත්ත අනුව, {area} ප්‍රදේශයේ ගඟේ ජල මට්ටම "
                        f"දිගින් දිගටම ඉහළ ගියහොත් {label_local} මට්ටම වන {next_level}m ට "
                        f"ළඟා වීමට ආසන්නයෙන් පැය {hours}ක් ගත විය හැකි බව ඇස්තමේන්තු කෙරේ. "
                        f"දැනට ජල මට්ටම {water}m ලෙස පවතී."
                    ),
                    "recommended_actions": self._timing_actions_localized(hours, next_label, lang),
                }
            if lang == "ta":
                return {
                    "title": f"{label_local} ~ {hours} மணிநேரத்தில் எதிர்பார்க்கப்படுகிறது",
                    "short_message": (
                        f"{area} இல் நீர்மட்டம் {water}m. "
                        f"{label_local} மட்டம் {next_level}m."
                    ),
                    "detailed_message": (
                        f"தற்போதைய உயர்வு வேகத்தில் {label_local} மட்டம் சுமார் {hours} மணிநேரத்தில் அடையும். "
                        f"தற்போது {water}m."
                    ),
                    "recommended_actions": self._timing_actions_localized(hours, next_label, lang),
                }
            return {
                "title": f"{next_label} Expected in ~{hours} Hour(s)",
                "short_message": (
                    f"Current water level at {area} is {water}m. "
                    f"{next_label} threshold is {next_level}m."
                ),
                "detailed_message": (
                    f"At the current rise rate, {next_label} threshold is expected in approximately "
                    f"{hours} hour(s). Current water level at {area} is {water}m."
                ),
                "recommended_actions": self._timing_actions_localized(hours, next_label, lang),
            }

        if key in {"CONFIRMED_MAJOR", "CONFIRMED_MINOR", "CONFIRMED_ALERT"}:
            water = ctx.get("water_level", 0.0)
            rainfall = ctx.get("rainfall", 0.0)
            trend = self._trend_for_lang(ctx.get("trend", "steady"), lang)
            label = ctx.get("label", "Alert")
            label_local = self._label_for_lang(label, lang)
            receding = ctx.get("receding", False)
            if lang == "si":
                return {
                    "title": f"{area} ප්‍රදේශයේ {label_local} ඇඟවීම නිකුත් කෙරේ",
                    "short_message": (
                        f"{area} ප්‍රදේශයේ ගඟ {label_local} මට්ටමට ළඟා වී ඇත. "
                        f"දැනට ජල මට්ටම {water}m ක් ලෙස {trend} ලෙස ඇත."
                    ),
                    "detailed_message": (
                        f"{area} ප්‍රදේශයේ ගඟේ ජල මට්ටම {water}m ({trend}) දක්වා ළඟා වී ඇති අතර, "
                        f"අදට ලැබුණු මුළු වර්ෂාපතනය {rainfall}mm ලෙස වාර්තා වේ. "
                        + (
                            "ජල මට්ටම ඉහළ ශ්‍රේණියක පවතිනවා නමුත් දැනට සෙමෙන් පහළ "
                            "යමින් ඇති බව සටහන් කෙරේ. ප්‍රදේශය තවමත් අවදානම් කලාපයේ "
                            "පවතින බැවින් ජනතාව සතුටෙන් නිවෙස්වල රැඳෙන ලෙස ඉල්ලා සිටිනු ලැබේ."
                            if receding and label == "Major Flood"
                            else
                            "ගඟ ආශ්‍රිත ජනතාව ඉක්මනින් ඉහළ ස්ථානවලට ඉවත් වන ලෙස "
                            "ආපදා කළමනාකරණ මධ්‍යස්ථානය දැඩිව දන්වා සිටී."
                            if label == "Major Flood"
                            else
                            "පහත් බිම් ප්‍රදේශවල ගෘහ සතුන් හා අය්ත්‍ය ඉහළ ස්ථානවලට "
                            "ගෙනයාමට ඉල්ලා සිටිනු ලැබේ."
                        )
                    ),
                    "recommended_actions": (
                        [
                            "ජල මට්ටම ක්‍රමයෙන් පහළ යමින් ඇතත්, ගංවතුර ස්ථාන ආසන්නයේ "
                            "රැඳීමෙන් වළකින්න. ආපදා නිලධාරීන්ගේ දැනුම් දෙන ලද ස්ථානවල "
                            "නිවෙස්වලට ආපසු යාම ආරම්භ කිරීමට රැඳෙන්න."
                        ]
                        if receding and label == "Major Flood"
                        else [
                            "ගංවතුරට ලක් වූ ප්‍රදේශ ආශ්‍රිත ජනතාව ඉක්මනින් ආරක්ෂිත ස්ථානවලට ඉවත් වන්න.",
                            "ජලයෙන් යටවූ මාර්ග, පාලම් හෝ ගඟ ආශ්‍රිත ස්ථාන ළඟා නොවන්න.",
                        ]
                        if label == "Major Flood"
                        else [
                            "ගෘහ භාණ්ඩ, ලේඛන සහ ගෘහ සතුන් ඉහළ ස්ථානවලට ගෙනයන්න.",
                            "ගඟ ආශ්‍රිතව නවතා ඇති වාහන ඉවත් කරන්න.",
                        ]
                        if label == "Minor Flood"
                        else [
                            "ජාතික ආපදා ව්‍යාප්ති ශාලාව සහ යෙදුම හරහා නවතම නිවේදන නිරීක්ෂණය කරන්න.",
                            "හදිසි ආරක්ෂිත කට්ටලයක් සූදානම් කරගන්න.",
                        ]
                    ),
                }
            if lang == "ta":
                return {
                    "title": f"{label_local} எச்சரிக்கை",
                    "short_message": f"{area} பகுதியில் {label_local} நிலை கண்டறியப்பட்டுள்ளது.",
                    "detailed_message": (
                        f"{area} இல் நீர்மட்டம் {water}m ({trend}). "
                        f"இன்றைய மழை {rainfall}mm."
                    ),
                    "recommended_actions": (
                        ["நீர்மட்டம் உயர்ந்தாலும் குறைந்து வருகிறது. மிகுந்த கவனமாக இருங்கள்."]
                        if receding and label == "Major Flood"
                        else [
                            "உடனடியாக உயரமான இடத்திற்கு செல்லவும்.",
                            "வெள்ளம் உள்ள சாலைகள் அல்லது பாலங்களை கடக்க வேண்டாம்.",
                        ]
                        if label == "Major Flood"
                        else [
                            "மதிப்புள்ள பொருட்களை உயர்ந்த இடத்திற்கு மாற்றவும்.",
                            "ஆற்றருகில் வாகனங்களை நிறுத்த வேண்டாம்.",
                        ]
                        if label == "Minor Flood"
                        else [
                            "உள்ளூர் வானிலை அறிவிப்புகளை கவனிக்கவும்.",
                            "அவசர கிட்டை தயாராக்கவும்.",
                        ]
                    ),
                }
            return {
                "title": f"{label} Warning",
                "short_message": f"{label} conditions detected at {area}.",
                "detailed_message": (
                    f"Water level at {area} is {water}m ({trend}). "
                    f"Today's rainfall is {rainfall}mm."
                ),
                "recommended_actions": ctx.get("actions", []),
            }

        if key == "HEAVY_RAIN":
            rainfall = ctx.get("rainfall", 0.0)
            water = ctx.get("water_level", 0.0)
            if lang == "si":
                return {
                    "title": f"{area} ප්‍රදේශයේ දැඩි වර්ෂාව",
                    "short_message": (
                        f"{area} ප්‍රදේශයේ අදට {rainfall}mm ක් ලෙස දැඩි වර්ෂාවක් "
                        "ලැබී ඇති නමුත් දැනට ගඟේ ජල මට්ටම සාමාන්‍ය පරාසය තුළ පවතී."
                    ),
                    "detailed_message": (
                        f"අද {area} ප්‍රදේශයේ {rainfall}mm ක් ලෙස ප්‍රබල වර්ෂාවක් ලැබී ඇත. "
                        f"දැනට ගඟේ ජල මට්ටම {water}m ලෙස ආරක්ෂිත සීමාවෙහි පවතී. "
                        "කෙසේ නමුත් වර්ෂාව දිගටම ඇදී ගියහොත් ජල මට්ටම ඉහළ "
                        "යා හැකි බැවින් ජනතාව දිගටම අවධානයෙන් සිටිය යුතු බව "
                        "ආපදා කළමනාකරණ ජාතික මධ්‍යස්ථානය දන්වා සිටී."
                    ),
                    "recommended_actions": [
                        "කාලගුණ දෙපාර්තමේන්තුවේ නවතම නිවේදන අනුගමනය කරන්න.",
                        "දැනට හදිසි තත්ත්වයක් නොමැත; නමුත් සූදානම්ව සිටීම යෝග්‍ය වේ.",
                    ],
                }
            if lang == "ta":
                return {
                    "title": "கடுமையான மழை அறிவுரை",
                    "short_message": (
                        f"குறைவு: {area} பகுதியில் இன்று {rainfall}mm மழை. "
                        "நீர்மட்டம் சாதாரணமாக உள்ளது."
                    ),
                    "detailed_message": (
                        f"இன்று {rainfall}mm மழை பதிவாகியுள்ளது. தற்போது {area} இல் நீர்மட்டம் {water}m. "
                        "மழை தொடர்ந்தால் நீர்மட்டம் உயரலாம்."
                    ),
                    "recommended_actions": [
                        "வானிலை புதுப்பிப்புகளை கவனிக்கவும்.",
                        "தேவைப்பட்டால் தயாராக இருங்கள்.",
                    ],
                }
            return {
                "title": "Heavy Rain Advisory",
                "short_message": (
                    f"LOW: Heavy rainfall at {area} ({rainfall}mm today). "
                    "River level currently normal."
                ),
                "detailed_message": (
                    f"Significant rainfall of {rainfall}mm recorded today. "
                    f"Current water level is {water}m but may rise if rain continues."
                ),
                "recommended_actions": [
                    "Monitor weather updates in case rainfall continues.",
                    "No immediate threat.",
                ],
            }

        return {}

    def _build_localized_fields(self, alert_info: dict) -> dict:
        key = alert_info.get("localization_key")
        ctx = alert_info.get("localization_context", {})
        if not key:
            return {}

        si = self._render_localized(key, ctx, "si")
        ta = self._render_localized(key, ctx, "ta")
        return {
            "titleSi": si.get("title"),
            "titleTa": ta.get("title"),
            "shortMessageSi": si.get("short_message"),
            "shortMessageTa": ta.get("short_message"),
            "detailedMessageSi": si.get("detailed_message"),
            "detailedMessageTa": ta.get("detailed_message"),
            "recommendedActionsSi": si.get("recommended_actions", []),
            "recommendedActionsTa": ta.get("recommended_actions", []),
        }

    # ==========================================================================
    # TRIGGER 1 — Proximity Warning
    # ==========================================================================

    def check_proximity_warning(self, station_code: str, data: dict) -> dict | None:
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
            "localization_key": "PROXIMITY_WARNING",
            "localization_context": {
                "area": area,
                "pct_of_alert": pct_of_alert,
                "headroom_m": headroom_m,
                "water_level": water_level,
                "alert_level": alert_level,
            },
        }

    # ==========================================================================
    # TRIGGER 2 — Probability Warning
    # ==========================================================================

    def check_probability_warning(self, station_code: str, data: dict) -> dict | None:
        if data.get("flood_category", "Normal") != "Normal":
            return None

        probs       = data.get("probabilities", {})
        p_alert     = probs.get("Alert", 0.0)
        p_minor     = probs.get("Minor Flood", 0.0)
        p_major     = probs.get("Major Flood", 0.0)
        water_level = data.get("current_water_level_m", 0.0)
        area        = STATION_META.get(station_code, {}).get("area", station_code)

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
                "localization_key": "PROBABILITY_MAJOR",
                "localization_context": {
                    "area": area,
                    "probability": p_major,
                    "water_level": water_level,
                    "label": "Major Flood",
                },
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
                "localization_key": "PROBABILITY_MINOR",
                "localization_context": {
                    "area": area,
                    "probability": p_minor,
                    "water_level": water_level,
                    "label": "Minor Flood",
                },
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
                "localization_key": "PROBABILITY_ALERT",
                "localization_context": {
                    "area": area,
                    "probability": p_alert,
                    "water_level": water_level,
                    "label": "Alert",
                },
            }

        return None

    # ==========================================================================
    # TRIGGER 3 — Rate of Rise Warning
    # ==========================================================================

    def check_rate_of_rise_warning(self, station_code: str, data: dict) -> dict | None:
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
            "localization_key": "RATE_OF_RISE_WARNING",
            "localization_context": {
                "area": area,
                "rise_rate": round(rise_per_hour, 2),
                "water_level": water_level,
                "alert_level": alert_level,
                "projected_1h": projected_1h,
                "projected_3h": projected_3h,
            },
        }

    # ==========================================================================
    # TRIGGER 4 — Timing Warning
    # ==========================================================================

    def check_timing_warning(self, station_code: str, data: dict) -> dict | None:
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
            "localization_key": "TIMING_WARNING",
            "localization_context": {
                "area": area,
                "hours": hours,
                "next_label": next_label,
                "next_level": next_level,
                "water_level": water_level,
            },
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

    def _timing_actions_localized(self, hours: int, next_label: str, lang: str) -> list:
        if lang == "si":
            if hours <= 1:
                return [
                    "ගංවතුරට ලක් විය හැකි ප්‍රදේශයක සිටිනවා නම් ඉතා ඉක්මනින් ඉහළ ස්ථානයකට ගෙනයන්න.",
                    "ප්‍රමාද නොකරන්න — දැන් ම ඉවත් වන්න. ඔරලෝසු ගෙවෙමින් ඇත.",
                    "ඉවත් වීමට ආධාර අවශ්‍ය නම් ආපදා හදිසි දුරකථන අංකය (1938) අමතන්න.",
                ]
            if hours <= 3:
                return [
                    "ගෘහ භාණ්ඩ, ලේඛන, ඖෂධ සහ ගෘහ සතුන් ඉහළ ස්ථානවලට ගෙනයාම දැන් ම ආරම්භ කරන්න.",
                    "ජල බෝතල්, ආහාර, ඖෂධ සහ ගෘහ ලේඛන ඇතුළත් හදිසි ඇසුරුමක් සූදානම් කරගන්න.",
                    "ඉවත් වීමේ මාර්ගය කල්තියා නිශ්චිත කරගන්න.",
                ]
            return [
                "ඉදිරි පැය කිහිපය තුළ ගඟේ ජල මට්ටම දිගටම නිරීක්ෂණය කරන්න.",
                "හදිසි ආරක්ෂිත ඇසුරුමක් ලෑස්ති කරගෙන සූදානම්ව සිටින්න.",
                "ජාතික ආපදා ව්‍යාප්ති ශාලාවේ සහ යෙදුමේ නිවේදන දිගටම පරීක්ෂා කරන්න.",
            ]
        if lang == "ta":
            if hours <= 1:
                return [
                    "வெள்ள அபாயப் பகுதியில் இருந்தால் உடனடியாக உயரமான இடத்திற்கு செல்லவும்.",
                    "தாமதிக்க வேண்டாம் — உடனடியாக வெளியேறவும்.",
                    "தேவைப்பட்டால் அவசர சேவைகளை அழைக்கவும்.",
                ]
            if hours <= 3:
                return [
                    "மதிப்புள்ள பொருட்கள் மற்றும் வாகனங்களை உயர்ந்த இடத்திற்கு மாற்றவும்.",
                    "அவசர உதவி பொருட்கள் கொண்ட கிட்டை தயாராக்கவும்.",
                    "வெளியேறும் பாதையை முன்பே தெரிந்து கொள்ளவும்.",
                ]
            return [
                "அடுத்த சில மணிநேரங்களில் நிலையை கவனமாக கண்காணிக்கவும்.",
                "அவசர கிட்டை தயார் நிலையில் வைத்திருக்கவும்.",
                "செயலி மற்றும் உள்ளூர் வானிலை புதுப்பிப்புகளை பார்க்கவும்.",
            ]
        return self._timing_actions(hours, next_label)

    # ==========================================================================
    # Confirmed flood alerts
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
                "localization_key": "CONFIRMED_MAJOR",
                "localization_context": {
                    "area": area,
                    "water_level": water_level,
                    "rainfall": rainfall,
                    "trend": trend_str,
                    "label": "Major Flood",
                    "receding": rising_flag == -1,
                },
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
                "localization_key": "CONFIRMED_MINOR",
                "localization_context": {
                    "area": area,
                    "water_level": water_level,
                    "rainfall": rainfall,
                    "trend": trend_str,
                    "label": "Minor Flood",
                    "receding": False,
                },
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
                "localization_key": "CONFIRMED_ALERT",
                "localization_context": {
                    "area": area,
                    "water_level": water_level,
                    "rainfall": rainfall,
                    "trend": trend_str,
                    "label": "Alert",
                    "receding": False,
                },
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
                "localization_key": "HEAVY_RAIN",
                "localization_context": {
                    "area": area,
                    "water_level": water_level,
                    "rainfall": rainfall,
                },
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
            timing = self.check_timing_warning(station_code, data)
            if timing:
                alerts.append(timing)
            return alerts

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
        now       = datetime.now(ZoneInfo("Asia/Colombo"))
        area_meta = STATION_META.get(station_code, {})
        area_name = area_meta.get("area", station_code)
        district  = area_meta.get("district", "Unknown")
        safe_name = area_name.replace(" ", "_")
        trigger   = alert_info.get("trigger", "UNKNOWN")

        payload = {
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

        if "short_message" in alert_info:
            payload["shortMessage"] = alert_info.get("short_message")
        if "detailed_message" in alert_info:
            payload["detailedMessage"] = alert_info.get("detailed_message")
        if "recommended_action" in alert_info:
            payload["recommendedActions"] = alert_info.get("recommended_action")

        payload.update(self._build_localized_fields(alert_info))

        payload.pop("localization_key", None)
        payload.pop("localization_context", None)

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