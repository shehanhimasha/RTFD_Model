import requests

from config.settings import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_headers() -> dict:
    api_key = config.INTERNAL_API_KEY
    return {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }


def _post_payload(url: str, payload: dict, payload_name: str) -> bool:
    if not url:
        logger.warning(f"{payload_name} endpoint not set — skipping sync")
        return False

    if not config.INTERNAL_API_KEY:
        logger.warning(f"INTERNAL_API_KEY not set — skipping {payload_name} sync")
        return False

    try:
        response = requests.post(
            url,
            json=payload,
            headers=_build_headers(),
            timeout=10,
        )
        if response.status_code in (200, 201, 202):
            logger.info(f"{payload_name} synced to .NET API")
            return True

        logger.error(
            f"{payload_name} sync failed: {response.status_code} {response.text}"
        )
        return False
    except requests.RequestException as exc:
        logger.error(f"{payload_name} sync connection error: {exc}")
        return False


def send_prediction_data(prediction: dict) -> bool:
    return _post_payload(
        config.DOTNET_WEBHOOK_URL,
        prediction,
        "Prediction",
    )


def send_intermediate_data(intermediate_data: dict) -> bool:
    return _post_payload(
        config.DOTNET_PREDICTION_URL,
        intermediate_data,
        "Intermediate data",
    )
