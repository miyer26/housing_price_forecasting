import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with consistent configuration.

    Args:
        name (str): Name of the logger
        log_file (str, optional): Specific log file name. If None, timestamp will be used

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handlers if they don't exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
