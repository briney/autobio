"""Tests for autobio.cli.formatters."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from autobio.cli.formatters import (
    OutputFormat,
    format_image_list,
    format_tool_info,
    format_tool_info_catalog,
    format_tool_list,
    format_tool_list_merged,
    format_workspace_result,
)
from autobio.core.catalog import Mode, Tool
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

    def test_json_no_input_format_when_empty(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert "input_format" not in parsed

    def test_json_includes_input_format(self) -> None:
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
            input_format=("Uses YAML format.", "Example: version: 1"),
        )
        result = format_tool_info("mock-tool", entry, OutputFormat.JSON)
        parsed = json.loads(result)
        assert parsed["input_format"] == ["Uses YAML format.", "Example: version: 1"]

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

    def test_table_contains_input_format(self) -> None:
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
            input_format=("Uses YAML format.",),
        )
        result = format_tool_info("mock-tool", entry, OutputFormat.TABLE)
        assert "Input Format" in result
        assert "Uses YAML format." in result

    def test_table_no_input_format_when_empty(self) -> None:
        entry = _make_entry()
        result = format_tool_info("mock-tool", entry, OutputFormat.TABLE)
        assert "Input Format" not in result

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


# ---------------------------------------------------------------------------
# format_tool_info_catalog
# ---------------------------------------------------------------------------


class _InInfo(BaseInput):
    pass


class _OutInfo(BaseOutput):
    pass


def _tool_for_info() -> Tool:
    return Tool(
        name="demo",
        display_name="Demo",
        category=ToolCategory.SCORING,
        description="demo tool",
        version="1.0.0",
        image_tag="demo:1.0.0",
        requires_gpu=False,
        gpu_count=0,
        default_mode="a",
        modes={
            "a": Mode(
                "a",
                "Alpha",
                "alpha mode",
                _InInfo,
                _OutInfo,
                default_timeout=300,
                notes=("First note.", "Second note."),
            ),
            "b": Mode(
                "b",
                "Beta",
                "beta mode",
                _InInfo,
                _OutInfo,
                default_timeout=600,
                category=ToolCategory.SIMULATION,
            ),
        },
        keywords=("demo", "example"),
    )


def test_format_tool_info_catalog_json_shape() -> None:
    parsed = json.loads(format_tool_info_catalog(_tool_for_info(), OutputFormat.JSON))
    assert parsed["name"] == "demo"
    assert parsed["default_mode"] == "a"
    assert parsed["categories"] == ["scoring", "simulation"]  # union, primary first
    assert parsed["keywords"] == ["demo", "example"]
    mode_names = [m["name"] for m in parsed["modes"]]
    assert mode_names == ["a", "b"]
    mode_a = parsed["modes"][0]
    assert mode_a["category"] == "scoring"  # falls back to Tool category
    assert "input_schema" in mode_a and "output_schema" in mode_a
    assert parsed["modes"][1]["category"] == "simulation"  # mode override


def test_format_tool_info_catalog_table_runs() -> None:
    out = format_tool_info_catalog(_tool_for_info(), OutputFormat.TABLE)
    assert "demo" in out and "Alpha" in out


def test_format_tool_info_catalog_json_includes_notes() -> None:
    parsed = json.loads(format_tool_info_catalog(_tool_for_info(), OutputFormat.JSON))
    assert parsed["modes"][0]["notes"] == ["First note.", "Second note."]
    assert parsed["modes"][1]["notes"] == []  # mode "b" has no notes


def test_format_tool_info_catalog_table_includes_notes() -> None:
    out = format_tool_info_catalog(_tool_for_info(), OutputFormat.TABLE)
    assert "First note." in out and "Second note." in out


def test_format_tool_info_catalog_json_includes_tool_notes() -> None:
    """JSON output must include tool-level notes at the top level."""
    tool = Tool(
        name="noted-tool",
        display_name="Noted Tool",
        category=ToolCategory.SCORING,
        description="A tool with notes.",
        version="1.0.0",
        image_tag="noted:1.0.0",
        requires_gpu=False,
        gpu_count=0,
        default_mode="default",
        modes={
            "default": Mode(
                "default",
                "Default",
                "default mode",
                _InInfo,
                _OutInfo,
                default_timeout=300,
            ),
        },
        notes=("Important caveat.", "Read the docs."),
    )
    parsed = json.loads(format_tool_info_catalog(tool, OutputFormat.JSON))
    assert parsed["notes"] == ["Important caveat.", "Read the docs."]


def test_format_tool_info_catalog_table_includes_tool_notes() -> None:
    """TABLE output must include tool-level notes as a row."""
    tool = Tool(
        name="noted-tool",
        display_name="Noted Tool",
        category=ToolCategory.SCORING,
        description="A tool with notes.",
        version="1.0.0",
        image_tag="noted:1.0.0",
        requires_gpu=False,
        gpu_count=0,
        default_mode="default",
        modes={
            "default": Mode(
                "default",
                "Default",
                "default mode",
                _InInfo,
                _OutInfo,
                default_timeout=300,
            ),
        },
        notes=("Critical requirement.", "Handle with care."),
    )
    out = format_tool_info_catalog(tool, OutputFormat.TABLE)
    assert "Critical requirement." in out
    assert "Handle with care." in out


# ---------------------------------------------------------------------------
# format_tool_list_merged
# ---------------------------------------------------------------------------


def _flat_entry() -> ToolEntry:
    return ToolEntry(
        image_tag="prodigy:1.0.0",
        category=ToolCategory.SCORING,
        requires_gpu=False,
        gpu_count=0,
        input_schema=_InInfo,
        output_schema=_OutInfo,
        default_timeout=300,
        supports_batch=False,
        description="legacy tool",
        version="1.0.0",
    )


def test_format_tool_list_merged_json_has_both() -> None:
    rows = json.loads(
        format_tool_list_merged(
            {"prodigy": _flat_entry()}, {"demo": _tool_for_info()}, OutputFormat.JSON
        )
    )
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"prodigy", "demo"}
    assert by_name["demo"]["modes"] == ["a", "b"]
    assert by_name["demo"]["categories"] == ["scoring", "simulation"]
    assert "modes" not in by_name["prodigy"]  # legacy row keeps the old shape
    assert [r["name"] for r in rows] == ["demo", "prodigy"]  # sorted by name


def test_format_tool_list_merged_table_runs() -> None:
    out = format_tool_list_merged(
        {"prodigy": _flat_entry()}, {"demo": _tool_for_info()}, OutputFormat.TABLE
    )
    assert "prodigy" in out and "demo" in out


def test_format_tool_info_catalog_table_skips_schema_computation() -> None:
    """TABLE format must not call model_json_schema() on mode schemas (micro-opt)."""

    class _ExplodingInput(BaseInput):
        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            raise AssertionError("model_json_schema must not be called for TABLE")

    tool = Tool(
        name="demo",
        display_name="Demo",
        category=ToolCategory.SCORING,
        description="demo tool",
        version="1.0.0",
        image_tag="demo:1.0.0",
        requires_gpu=False,
        gpu_count=0,
        default_mode="a",
        modes={
            "a": Mode("a", "Alpha", "alpha mode", _ExplodingInput, _OutInfo, default_timeout=300),
        },
    )
    out = format_tool_info_catalog(tool, OutputFormat.TABLE)
    assert "Mode: a" in out
    with pytest.raises(AssertionError, match="must not be called"):
        format_tool_info_catalog(tool, OutputFormat.JSON)
