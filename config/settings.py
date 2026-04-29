from pathlib import Path
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Paths:
    PROJECT_ROOT   = Path(__file__).parent.parent
    DATA_RAW       = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS         = PROJECT_ROOT / "models" / "trained"
    LOGS           = PROJECT_ROOT / "logs"

@dataclass
class Config:
    DOTNET_WEBHOOK_URL = os.getenv("DOTNET_WEBHOOK_URL")
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

paths = Paths()
config = Config()