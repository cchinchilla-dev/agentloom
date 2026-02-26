"""Structured JSON logging — uses only stdlib, always available."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields
        if hasattr(record, "workflow_id"):
            log_entry["workflow_id"] = getattr(record, "workflow_id")
        if hasattr(record, "step_id"):
            log_entry["step_id"] = getattr(record, "step_id")
        if hasattr(record, "provider"):
            log_entry["provider"] = getattr(record, "provider")
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = getattr(record, "correlation_id")

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Simple human-readable formatter."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(
    level: str = "INFO",
    format: str = "json",
    logger_name: str = "agentloom",
) -> logging.Logger:
    """Configure structured logging for AgentLoom.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: Output format ('json' or 'text').
        logger_name: Root logger name to configure.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger
