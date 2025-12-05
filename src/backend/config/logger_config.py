"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path


class LoggerConfig:
    """Centralized logging configuration."""

    @staticmethod
    def setup_logger(name: str = "csv_generator", level: int = logging.INFO) -> logging.Logger:
        """Setup and return a configured logger."""
        logger = logging.getLogger(name)

        # Don't configure if already configured
        if logger.handlers:
            return logger

        logger.setLevel(level)

        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

        return logger