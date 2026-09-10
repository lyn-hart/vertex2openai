import logging
import os
import sys


def mask_key(key: str) -> str:
    """Return a masked form of a secret, showing only the last 4 characters."""
    if not key:
        return "<empty>"
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def configure_logging() -> None:
    """Configure root logging based on LOG_LEVEL (default INFO)."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if level_name not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        print(f"WARNING: Invalid LOG_LEVEL {level_name!r}. Using INFO.")

    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet down noisy third-party loggers unless debugging.
    if level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "uvicorn.access"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
