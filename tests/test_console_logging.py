"""The package console logger: bare INFO (banners keep their layout), prefixed WARNING."""

from __future__ import annotations

import logging

from safe_rl.utils.console import PACKAGE_LOGGER, LevelPrefixFormatter, configure, get_logger


def _record(level: int, message: str, name: str = "safe_rl.runners.off_policy_runner") -> logging.LogRecord:
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1, msg=message, args=(), exc_info=None)


def test_info_renders_bare() -> None:
    """A banner logged at INFO must survive byte-for-byte."""
    banner = "────────\n  SAC  |  8 envs\n────────"
    assert LevelPrefixFormatter().format(_record(logging.INFO, banner)) == banner


def test_warning_is_prefixed_with_level_and_module() -> None:
    formatted = LevelPrefixFormatter().format(_record(logging.WARNING, "no final_observation"))
    assert formatted == "[WARNING] off_policy_runner: no final_observation"


def test_error_is_prefixed_too() -> None:
    assert LevelPrefixFormatter().format(_record(logging.ERROR, "boom")).startswith("[ERROR] ")


def test_configure_is_idempotent() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER)
    before = list(logger.handlers)
    for _ in range(3):
        configure()
    assert logger.handlers == before, "repeated configure() must not stack handlers"


def test_get_logger_returns_a_child_of_the_package_logger() -> None:
    logger = get_logger("safe_rl.runners.off_policy_runner")
    assert logger.name.startswith(PACKAGE_LOGGER)
    # Records reach handlers bound to the package logger regardless of propagation.
    seen: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = seen.append
    package_logger = logging.getLogger(PACKAGE_LOGGER)
    package_logger.addHandler(handler)
    try:
        logger.warning("hello")
    finally:
        package_logger.removeHandler(handler)
    assert [r.getMessage() for r in seen] == ["hello"]
