from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class LogSignalHandler(logging.Handler):
    """Custom logging handler that emits Qt signals."""

    def __init__(self, log_signal: pyqtSignal) -> None:
        super().__init__()
        self.log_signal = log_signal

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.log_signal.emit(msg)


class LogEmitter(QObject):
    """Emits log messages as Qt signals."""
    log_message = pyqtSignal(str)


def setup_logging(log_file: Optional[Path] = None) -> tuple[logging.Logger, LogEmitter]:
    """Set up logging with both file and Qt signal handlers."""
    logger = logging.getLogger("ai_videoclipper")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Qt signal handler
    log_emitter = LogEmitter()
    signal_handler = LogSignalHandler(log_emitter.log_message)
    signal_handler.setLevel(logging.DEBUG)
    signal_handler.setFormatter(formatter)
    logger.addHandler(signal_handler)

    return logger, log_emitter
