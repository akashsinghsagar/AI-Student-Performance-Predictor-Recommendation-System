import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.utils.config import ROOT_DIR, get_settings


def configure_logging() -> None:
    settings = get_settings()
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(formatter)

        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
