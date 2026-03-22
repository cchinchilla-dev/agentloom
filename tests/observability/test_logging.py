"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

from agentloom.observability.logging import JSONFormatter, TextFormatter, setup_logging


class TestJSONFormatter:
    def test_formats_as_json(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_extra_fields(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="step done",
            args=(),
            exc_info=None,
        )
        record.workflow_id = "wf-1"  # type: ignore[attr-defined]
        record.step_id = "step-a"  # type: ignore[attr-defined]
        output = formatter.format(record)
        data = json.loads(output)
        assert data["workflow_id"] == "wf-1"
        assert data["step_id"] == "step-a"

    def test_exception_info(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "boom"


class TestTextFormatter:
    def test_format_includes_level(self) -> None:
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warning msg",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "WARNI" in output
        assert "warning msg" in output


class TestSetupLogging:
    def test_json_format(self) -> None:
        logger = setup_logging(level="DEBUG", format="json", logger_name="test.json")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)
        assert logger.level == logging.DEBUG

    def test_text_format(self) -> None:
        logger = setup_logging(level="INFO", format="text", logger_name="test.text")
        assert isinstance(logger.handlers[0].formatter, TextFormatter)

    def test_replaces_existing_handlers(self) -> None:
        logger = setup_logging(logger_name="test.replace")
        setup_logging(logger_name="test.replace")
        assert len(logger.handlers) == 1
