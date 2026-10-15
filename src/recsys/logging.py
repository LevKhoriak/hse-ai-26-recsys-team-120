import logging
import sys

PARENT = "recsys"
FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure the parent logger once. Safe to call repeatedly (e.g. notebook re-runs)."""
    logger = logging.getLogger(PARENT)
    logger.setLevel(level)
    if not logger.handlers:  # avoid duplicate lines on re-run / autoreload
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
        logger.addHandler(handler)
    logger.propagate = False  # don't double-print via the root logger (Jupyter has one)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger, e.g. get_logger("convert_raw") -> 'recsys.convert_raw'."""
    return logging.getLogger(f"{PARENT}.{name}")