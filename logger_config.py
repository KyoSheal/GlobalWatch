"""Centralized logging configuration for GlobalWatch.

Usage in any module:
    import logging
    logger = logging.getLogger(__name__)

Call setup_logging() once at application entry points (paper_trading main,
GlobalWatch_V2 app startup). All other modules just get a logger by name.
"""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(
    log_dir: str = "outputs",
    log_file: str = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure root logger with rotating file handler and console handler.

    Args:
        log_dir: Directory for log files (created if missing).
        log_file: Log file name inside log_dir.
        console_level: Minimum level printed to stdout.
        file_level: Minimum level written to the log file.
        max_bytes: Max size per log file before rotation.
        backup_count: Number of rotated log files to keep.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Avoid adding duplicate handlers if setup_logging() is called more than once.
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    # Rotating file handler — captures DEBUG and above.
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler — INFO and above by default.
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)
