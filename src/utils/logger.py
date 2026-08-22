"""
Logging utility.
Provides a factory for named, coloured loggers used throughout the project.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional

# Reconfigure stdout to UTF-8 on Windows so Unicode log messages don't crash
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ANSI colour codes for console output
_COLOURS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[35m", # Magenta
    "RESET": "\033[0m",
}


class _ColouredFormatter(logging.Formatter):
    """Custom formatter that adds colour to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:<8}{reset}"
        return super().format(record)


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with optional file output.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.
    level : str
        Logging level string: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str or Path, optional
        If provided, also write logs to this file (appended).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colour formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # Optional file handler (plain text, no colour codes)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def setup_project_logger(log_dir: str | Path = "results/logs") -> logging.Logger:
    """
    Set up the root project logger, writing to a persistent log file.

    Parameters
    ----------
    log_dir : str or Path
        Directory where the log file will be created.

    Returns
    -------
    logging.Logger
        Root project logger.
    """
    log_path = Path(log_dir) / "warehouse_rl.log"
    return get_logger("warehouse_rl", level="INFO", log_file=log_path)
