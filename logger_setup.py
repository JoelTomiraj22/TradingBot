"""
Logging setup.
Console + daily rotating file logger in logs/ folder.
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def get_logger(name: str = "crypto_bot") -> logging.Logger:
    """
    Create a logger that writes to both console and a daily log file.
    Log files rotate daily, keeping the last 30 days.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Create logs directory
    os.makedirs(LOG_DIR, exist_ok=True)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (DEBUG+, daily rotation, keep 30 days)
    log_file = os.path.join(LOG_DIR, "bot.log")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.suffix = "%Y-%m-%d"
    logger.addHandler(file_handler)

    return logger


# Convenience: module-level logger
logger = get_logger()


if __name__ == "__main__":
    log = get_logger("test")
    log.debug("This is a debug message (file only)")
    log.info("This is an info message (console + file)")
    log.warning("This is a warning")
    log.error("This is an error")
    print(f"\nLog directory: {LOG_DIR}")
