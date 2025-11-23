"""
Structured logging setup for training pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup structured logging.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "training.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

