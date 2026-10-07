"""
Application-wide logging configuration.

Provides structured, rotating file logs plus console output. Every module
in the application should retrieve its logger via ``logging.getLogger(__name__)``
after :func:`configure_logging` has been called once at startup.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    """Configure root logging handlers exactly once."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured (e.g. re-import during tests / reload).
        root_logger.setLevel(log_level)
        return

    root_logger.setLevel(log_level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    log_file = settings.log_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party loggers.
    for noisy_logger in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured (level=%s, file=%s)", settings.log_level, log_file
    )
