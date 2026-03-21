import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    api_title: str
    api_version: str
    cors_origins: List[str]
    upload_path: Path
    model_path: Path
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cors_raw = os.getenv("CORS_ORIGINS", "*")
    cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()]

    upload_path = Path(os.getenv("UPLOAD_PATH", "uploads/latest_upload.csv"))
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/student_model.joblib"))

    return Settings(
        api_title=os.getenv("API_TITLE", "AI Student Performance Predictor API"),
        api_version=os.getenv("API_VERSION", "2.0.0"),
        cors_origins=cors_origins or ["*"],
        upload_path=upload_path,
        model_path=model_path,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
