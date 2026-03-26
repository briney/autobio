"""Tests for autobio.utils.logging."""

from __future__ import annotations

import json
import logging

from autobio.utils.logging import JsonLineFormatter, get_logger, setup_logging


class TestSetupLogging:
    def test_configures_level(self) -> None:
        setup_logging("DEBUG")
        root = logging.getLogger("autobio")
        assert root.level == logging.DEBUG
        # Reset
        root.handlers.clear()

    def test_does_not_duplicate_handlers(self) -> None:
        root = logging.getLogger("autobio")
        root.handlers.clear()
        setup_logging("INFO")
        setup_logging("INFO")
        assert len(root.handlers) == 1
        root.handlers.clear()


class TestJsonLineFormatter:
    def test_output_is_valid_json(self) -> None:
        formatter = JsonLineFormatter()
        record = logging.LogRecord(
            name="autobio.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        line = formatter.format(record)
        parsed = json.loads(line)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
        assert parsed["logger"] == "autobio.test"
        assert "timestamp" in parsed

    def test_exception_included(self) -> None:
        formatter = JsonLineFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="autobio.err",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="fail",
            args=(),
            exc_info=exc_info,
        )
        line = formatter.format(record)
        parsed = json.loads(line)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestGetLogger:
    def test_namespaced(self) -> None:
        logger = get_logger("core.workspace")
        assert logger.name == "autobio.core.workspace"

    def test_different_names_different_loggers(self) -> None:
        a = get_logger("a")
        b = get_logger("b")
        assert a is not b
