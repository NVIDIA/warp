# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging infrastructure for Warp.

Provides a pluggable logger interface (:class:`Logger`) and an internal
default implementation. Frameworks can supply custom loggers via
:func:`warp.set_logger`.
"""

import linecache
import sys
import warnings
from typing import Protocol, runtime_checkable

LOG_DEBUG = 10
"""Log level for verbose compilation and debugging output."""

LOG_INFO = 20
"""Log level for informational messages (init banner, launch info)."""

LOG_WARNING = 30
"""Log level for warnings and deprecation notices."""

LOG_ERROR = 40
"""Log level for error messages."""


@runtime_checkable
class Logger(Protocol):
    """Protocol for Warp loggers.

    Any object with ``debug``, ``info``, ``warning``, and ``error`` methods
    matching these signatures can be used as a Warp logger.  Frameworks
    register an instance via :func:`warp.set_logger`.
    """

    def debug(self, message: str) -> None:
        """Emit a debug-level message."""
        ...

    def info(self, message: str) -> None:
        """Emit an info-level message."""
        ...

    def warning(self, message: str, category=None, stacklevel: int = 1) -> None:
        """Emit a warning-level message.

        Implementations should typically call ``warnings.warn(message, category,
        stacklevel=stacklevel)`` to integrate with Python's warning filter
        machinery. The provided ``stacklevel`` is already adjusted for direct
        use with ``warnings.warn()``.
        """
        ...

    def error(self, message: str) -> None:
        """Emit an error-level message."""
        ...


def _validate_logger(logger):
    # Protocol isinstance only checks attribute presence, not that they are
    # callable -- an object with debug=None would otherwise pass and fail
    # later inside an emitter.
    if (
        not isinstance(logger, Logger)
        or not callable(getattr(logger, "debug", None))
        or not callable(getattr(logger, "info", None))
        or not callable(getattr(logger, "warning", None))
        or not callable(getattr(logger, "error", None))
    ):
        raise TypeError(
            "logger must implement the Logger protocol "
            "(debug(message), info(message), warning(message, category, stacklevel), error(message)), "
            f"got {type(logger)!r}"
        )


def _format_warning(message, category, filename, lineno, line=None):
    """Format a Warp warning, optionally including source location."""
    import warp.config  # noqa: PLC0415 -- circular: logger.py is imported before config in warp/__init__.py

    s = f"Warp {category.__name__}: {message}"
    if warp.config.verbose_warnings:
        s += f" ({filename}:{lineno})"
        if line is None:
            try:
                line = linecache.getline(filename, lineno)
            except Exception:
                line = None
        if line:
            s += f"\n  {line.strip()}"
    return s + "\n"


def _warp_showwarning_stderr(message, category, filename, lineno, file=None, line=None):
    """Format and write a Warp warning to sys.stderr."""
    sys.stderr.write(_format_warning(message, category, filename, lineno, line))


class LoggerBasic:
    """Default Warp logger.

    Routes debug/info to ``sys.stdout`` and warnings/errors to ``sys.stderr``.
    Warnings go through ``warnings.warn()`` to respect user-configured filters.
    """

    def debug(self, message: str) -> None:
        sys.stdout.write(message + "\n")

    def info(self, message: str) -> None:
        sys.stdout.write(message + "\n")

    def warning(self, message: str, category=None, stacklevel: int = 1) -> None:
        with warnings.catch_warnings():
            warnings.showwarning = _warp_showwarning_stderr
            warnings.warn(message, category, stacklevel=stacklevel)

    def error(self, message: str) -> None:
        sys.stderr.write("Warp Error: " + message + "\n")


# Module-level logger state
_active_logger: Logger = LoggerBasic()
_warnings_seen: set = set()


def set_logger(logger: Logger | None) -> None:
    """Set the active Warp logger.

    Any object satisfying the :class:`Logger` protocol can be used. Frameworks
    typically wrap their existing logging machinery (e.g. Omniverse Kit's
    ``carb``) in a small adapter class with the four methods.

    Pass ``None`` to restore Warp's built-in default logger.

    Args:
        logger: An object satisfying the :class:`Logger` protocol, or ``None``.

    Raises:
        TypeError: If ``logger`` does not satisfy the :class:`Logger` protocol.
    """
    global _active_logger
    if logger is None:
        _active_logger = LoggerBasic()
        return
    _validate_logger(logger)
    _active_logger = logger


def get_logger() -> Logger:
    """Return the active Warp logger."""
    return _active_logger


def log_debug(message: str) -> None:
    """Emit a debug-level message if debug logging or legacy verbose mode is enabled."""
    import warp.config  # noqa: PLC0415

    if warp.config.verbose or warp.config.log_level <= LOG_DEBUG:
        _active_logger.debug(message)


def log_info(message: str) -> None:
    """Emit an info-level message if ``warp.config.log_level <= LOG_INFO``."""
    import warp.config  # noqa: PLC0415

    if warp.config.log_level <= LOG_INFO:
        _active_logger.info(message)


def log_warning(message: str, category=None, stacklevel=1, once=False) -> None:
    """Emit a warning-level message if ``warp.config.log_level <= LOG_WARNING``.

    Args:
        message: Warning text.
        category: Warning category (e.g., ``DeprecationWarning``).
        stacklevel: Stack frames to skip for location reporting.
        once: If ``True``, suppress subsequent calls with the same
            ``(category, message)`` pair regardless of call site.
    """
    import warp.config  # noqa: PLC0415

    if warp.config.log_level > LOG_WARNING:
        return

    # DeprecationWarnings are deduplicated unconditionally to match the legacy
    # warn() helper this replaced; ``once=True`` extends that to other categories.
    dedup = once or category is DeprecationWarning
    if dedup and (category, message) in _warnings_seen:
        return

    # +2: skip the logger adapter and log_warning() so warning locations point
    # at the caller. Custom logger adapters can pass this directly to
    # warnings.warn().
    _active_logger.warning(message, category=category, stacklevel=stacklevel + 2)

    if dedup:
        _warnings_seen.add((category, message))


def log_error(message: str) -> None:
    """Emit an error-level message. Always emits regardless of log level."""
    _active_logger.error(message)
