from pathlib import Path
from dataclasses import dataclass

@dataclass
class Paths:
    PROJECT_ROOT   = Path(__file__).parent.parent
    DATA_RAW       = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    MODELS         = PROJECT_ROOT / "models" / "trained"
    LOGS           = PROJECT_ROOT / "logs"

paths = Paths()