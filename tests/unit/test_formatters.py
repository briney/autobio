"""Tests for autobio.cli.formatters."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from autobio.cli.formatters import (
    OutputFormat,
    format_image_list,
    format_tool_info,
    format_tool_list,
    format_workspace_result,
)
from autobio.core.container import ImageInfo
from autobio.core.registry import ToolCategory, ToolEntry
from autobio.core.result import RunResult
from autobio.schemas.base import BaseInput, BaseOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockInput(BaseInput):
    sequences: dict[str, str]


class _MockOutput(BaseOutput):
    scores: list[float]


def _make_entry(
    *,
    category: ToolCategory = ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu: bool = True,
) -> ToolEntry:
    return ToolEntry(
        image_tag="mock-tool:1.0",
        category=category,
        requires_gpu=requires_gpu,
        gpu_count=1,
        input_schema=_MockInput,
        output_schema=_MockOutput,
        default_timeout=600,
        supports_batch=False,
        description="A mock tool for testing.",
        version="1.0",
    )


def _make_run_result(*, status: str = "success") -> RunResult:
    return RunResult(
        status=status,
        exit_code=0,
        phase="inference",
        wall_time_seconds=12.5,
        completed=1,
        total=1,
    )


def _make_image_info() -> ImageInfo:
    return ImageInfo(
        uri="ghcr.io/briney/autobio-mock:1.0",
        tag="1.0",
        size=500_000_000,
        created=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# OutputFormat
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_values(self) -> None:
        assert OutputFormat.JSON == "json"
        assert OutputFormat.TABLE == "table"


# ---------------------------------------------------------------------------
# format_tool_list
# ---------------------------------------------------------------------------


class TestFormatToolList:
    def test_json_empty(self) -> None:
        result = format_tool_list({}, OutputFormat.JSON)
        assert json.loads(result) == []

    def test_json_populated(self) -> None:
        tools = {"mock-tool": _make_entry()}
        result = format_tool_list(tools, OutputFormat.JSON)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "mock-tool"
        assert parsed[0]["category"] == "structure-prediction"
        assert parsed[0]["gpu"] is True
        assert parsed[0]["version"] == "1.0"
        assert parsed[0]["description"] == "A mock tool for testing."

    def test_json_sorted_by_name(self) -> None:
        tools = {
            "zzz-tool": _make_entry(),
            "aaa-tool": _make_entry(),
        }
        result = format_tool_list(tools, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed[0]["name"] == "aaa-tool"
        assert parsed[1]["name"] == "zzz-tool"

    def test_table_empty(self) -> None:
        result = format_tool_list({}, OutputFormat.TABLE)
        assert "No tools registered." in result

    def test_table_populated(self) -> None:
        tools = {"mock-tool": _make_entry()}
        result = format_tool_list(tools, OutputFormat.TABLE)
        assert "mock-tool" in result
        assert "structure-prediction" in result


# ---------------------------------------------------------------------------
# format_tool_info
# ---------------------------------------------------------------------------


class TestFormatToolInfo:
    def test_json_includes_schema(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["name"] == "mock-tool"
        assert "input_schema" in parsed
        schema = parsed["input_schema"]
        assert "properties" in schema
        assert "sequences" in schema["properties"]

    def test_json_fields(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["category"] == "structure-prediction"
        assert parsed["image_tag"] == "mock-tool:1.0"
        assert parsed["requires_gpu"] is True
        assert parsed["gpu_count"] == 1
        assert parsed["default_timeout"] == 600
        assert parsed["supports_batch"] is False
        assert parsed["version"] == "1.0"

    def test_json_no_notes_when_empty(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert "notes" not in parsed

    def test_json_includes_notes(self) -> None:
        entry = ToolEntry(
            image_tag="mock-tool:1.0",
            category=ToolCategory.STRUCTURE_PREDICTION,
            requires_gpu=True,
            gpu_count=1,
            input_schema=_MockInput,
            output_schema=_MockOutput,
            default_timeout=600,
            supports_batch=False,
            description="A mock tool.",
            version="1.0",
            notes=("Parser may drop residues.", "Use unique chain IDs."),
        )
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["notes"] == ["Parser may drop residues.", "Use unique chain IDs."]

    def test_table_contains_notes(self) -> None:
        entry = ToolEntry(
            image_tag="mock-tool:1.0",
            category=ToolCategory.STRUCTURE_PREDICTION,
            requires_gpu=True,
            gpu_count=1,
            input_schema=_MockInput,
            output_schema=_MockOutput,
            default_timeout=600,
            supports_batch=False,
            description="A mock tool.",
            version="1.0",
            notes=("Parser may drop residues.",),
        )
        result = format_tool_info("mock-tool", entry, OutputFormat.TABLE)
        assert "Notes" in result
        assert "Parser may drop residues." in result

    def test_table_no_notes_when_empty(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.TABLE)
        assert "Notes" not in result

    def test_table_contains_name(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.TABLE)
        assert "mock-tool" in result
        assert "structure-prediction" in result


# ---------------------------------------------------------------------------
# format_workspace_result
# ---------------------------------------------------------------------------


class TestFormatWorkspaceResult:
    def test_json_valid(self) -> None:
        run_result = _make_run_result()
        result = format_workspace_result(run_result, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["exit_code"] == 0
        assert parsed["phase"] == "inference"

    def test_json_includes_error_fields(self) -> None:
        run_result = RunResult(
            status="failure",
            exit_code=1,
            phase="inference",
            error_type="RuntimeError",
            error_message="OOM",
            wall_time_seconds=5.0,
        )
        result = format_workspace_result(run_result, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["error_type"] == "RuntimeError"
        assert parsed["error_message"] == "OOM"

    def test_table_contains_status(self) -> None:
        run_result = _make_run_result()
        result = format_workspace_result(run_result, OutputFormat.TABLE)
        assert "success" in result
        assert "inference" in result


# ---------------------------------------------------------------------------
# format_image_list
# ---------------------------------------------------------------------------


class TestFormatImageList:
    def test_json_empty(self) -> None:
        result = format_image_list([], OutputFormat.JSON)
        assert json.loads(result) == []

    def test_json_populated(self) -> None:
        images = [_make_image_info()]
        result = format_image_list(images, OutputFormat.JSON)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["uri"] == "ghcr.io/briney/autobio-mock:1.0"
        assert parsed[0]["tag"] == "1.0"
        assert parsed[0]["size_mb"] == 500.0
        assert "2026-01-15" in parsed[0]["created"]

    def test_table_empty(self) -> None:
        result = format_image_list([], OutputFormat.TABLE)
        assert "No autobio images found locally." in result

    def test_table_populated(self) -> None:
        images = [_make_image_info()]
        result = format_image_list(images, OutputFormat.TABLE)
        assert "ghcr.io/briney/autobio-mock:1.0" in result
