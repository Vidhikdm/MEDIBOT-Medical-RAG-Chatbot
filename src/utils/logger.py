import logging
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import (
    LOG_LEVEL,
    LOG_FILE,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    CONSOLE_LOG_LEVEL,
)


# INTERNAL HELPERS

def _get_log_level(level: str) -> int:
    """Safely convert string log level to logging constant."""
    return getattr(logging, level.upper(), logging.INFO)



# LOGGER SETUP

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = LOG_LEVEL,
    console_level: str = CONSOLE_LOG_LEVEL,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Formatter 
    detailed_formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )

    # Console Handler 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_get_log_level(console_level))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File Handler 
    if log_file is None:
        log_file = LOG_FILE

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(_get_log_level(level))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Stop propagation to root logger
    logger.propagate = False

    return logger



# PUBLIC API

def get_logger(name: str) -> logging.Logger:
    """Standard logger for any module."""
    return setup_logger(name)


def get_ingestion_logger() -> logging.Logger:
    return get_logger("medibot.ingestion")


def get_retrieval_logger() -> logging.Logger:
    return get_logger("medibot.retrieval")


def get_generation_logger() -> logging.Logger:
    return get_logger("medibot.generation")


def get_main_logger() -> logging.Logger:
    return get_logger("medibot.main")



# LOG HELPERS

def log_section(logger: logging.Logger, title: str, width: int = 70):
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_subsection(logger: logging.Logger, title: str, width: int = 70):
    logger.info("-" * width)
    logger.info(title)
    logger.info("-" * width)



# PERFORMANCE LOGGER

class PerformanceLogger:
    """Context manager to log execution time."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.operation} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"Failed: {self.operation} ({elapsed:.2f}s)")
        return False


# SELF TEST

if __name__ == "__main__":
    logger = get_logger(__name__)

    log_section(logger, "LOGGER TEST")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_subsection(logger, "Performance Test")

    import time
    with PerformanceLogger(logger, "Sleep operation"):
        time.sleep(1)

    print(f"\n Logs written to: {LOG_FILE}")