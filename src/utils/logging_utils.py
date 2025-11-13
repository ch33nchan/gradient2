"""Utilities for logging experiments without decorative output."""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files. If None, only console output.
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}.log')
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Logger for structured metrics (CSV format)."""

    def __init__(self, filepath: str):
        """
        Initialize metrics logger.

        Args:
            filepath: Path to CSV file for metrics
        """
        self.filepath = filepath
        self.headers_written = False

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        """
        Log metrics to CSV file.

        Args:
            metrics: Dictionary of metric name to value
        """
        if not self.headers_written:
            with open(self.filepath, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
            self.headers_written = True

        with open(self.filepath, 'a') as f:
            f.write(','.join(str(v) for v in metrics.values()) + '\n')
