"""Console logging for long-running training code.

A thin wrapper over :mod:`logging`, not a replacement for
:class:`safe_rl.utils.logger.Logger`: that one records *metrics*, this one writes
*messages*. INFO renders bare so hand-aligned banners keep their layout; WARNING and
above are prefixed so they stand out in a scrolling log. Level via ``SAFE_RL_LOG_LEVEL``.
"""

from __future__ import annotations

import logging
import os

PACKAGE_LOGGER = "safe_rl"
DEFAULT_LEVEL = "INFO"
LEVEL_ENV_VAR = "SAFE_RL_LOG_LEVEL"


class LevelPrefixFormatter(logging.Formatter):
    """Render INFO and below bare; prefix WARNING and above with level + module."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno < logging.WARNING:
            return message
        return f"[{record.levelname}] {record.name.rsplit('.', 1)[-1]}: {message}"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name``, configuring the package logger on first use."""
    configure()
    return logging.getLogger(name)


def configure(force: bool = False) -> logging.Logger:
    """Install the package's console handler.

    Idempotent, and a no-op when the application has already configured logging itself.

    Args:
        force: Reinstall even if a handler is present.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    elif logger.handlers or logging.getLogger().handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(LevelPrefixFormatter())
    logger.addHandler(handler)
    logger.propagate = False  # our handler already writes to the console
    logger.setLevel(os.environ.get(LEVEL_ENV_VAR, DEFAULT_LEVEL).upper())
    return logger
