"""Structured logging helpers (stdlib ``logging`` only)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonLineFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON-serialised log line."""
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root ``autobio`` logger.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    root = logging.getLogger("autobio")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLineFormatter())
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``autobio`` hierarchy.

    Args:
        name: Dotted sub-name (e.g. ``"core.workspace"``).
    """
    return logging.getLogger(f"autobio.{name}")
