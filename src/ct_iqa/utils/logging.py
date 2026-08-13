"""Structured logging setup, used in place of ad-hoc ``print`` calls in
library code (scripts may still print to stdout for CLI output)."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger with a consistent, timestamped format."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
