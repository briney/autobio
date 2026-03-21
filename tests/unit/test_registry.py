"""Tests for autobio.core.registry."""

from __future__ import annotations

import pytest

from autobio.core.registry import (
    TOOL_REGISTRY,
    ToolCategory,
    ToolEntry,
    get_tool,
    list_tools,
)
from autobio.schemas.base import BaseInput, BaseOutput

# ---------------------------------------------------------------------------
# Helpers — lightweight mock schemas for ToolEntry construction
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


# ---------------------------------------------------------------------------
# ToolCategory
# ---------------------------------------------------------------------------


class TestToolCategory:
    def test_values(self) -> None:
        assert ToolCategory.STRUCTURE_PREDICTION == "structure-prediction"
        assert ToolCategory.EMBEDDING == "embedding"
        assert ToolCategory.INVERSE_FOLDING == "inverse-folding"
        assert ToolCategory.SCORING == "scoring"

    def test_is_str(self) -> None:
        for member in ToolCategory:
            assert isinstance(member, str)

    def test_member_count(self) -> None:
        assert len(ToolCategory) == 4


# ---------------------------------------------------------------------------
# ToolEntry
# ---------------------------------------------------------------------------


class TestToolEntry:
    def test_construction(self) -> None:
        entry = _make_entry()
        assert entry.image_tag == "mock-tool:1.0"
        assert entry.category == ToolCategory.STRUCTURE_PREDICTION
        assert entry.requires_gpu is True
        assert entry.gpu_count == 1
        assert entry.input_schema is _MockInput
        assert entry.output_schema is _MockOutput
        assert entry.default_timeout == 600
        assert entry.supports_batch is False
        assert entry.description == "A mock tool for testing."
        assert entry.version == "1.0"

    def test_category_accepts_strenum(self) -> None:
        entry = _make_entry(category=ToolCategory.EMBEDDING)
        assert entry.category == ToolCategory.EMBEDDING
        assert entry.category == "embedding"


# ---------------------------------------------------------------------------
# Registry helpers (get_tool / list_tools)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore TOOL_REGISTRY around each test."""
    saved = dict(TOOL_REGISTRY)
    TOOL_REGISTRY.clear()
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(saved)


class TestGetTool:
    def test_success(self) -> None:
        entry = _make_entry()
        TOOL_REGISTRY["mock-tool"] = entry
        assert get_tool("mock-tool") is entry

    def test_unknown_tool_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown tool 'nonexistent'"):
            get_tool("nonexistent")

    def test_error_message_lists_available_tools(self) -> None:
        TOOL_REGISTRY["aaa"] = _make_entry()
        TOOL_REGISTRY["zzz"] = _make_entry()
        with pytest.raises(KeyError, match="aaa, zzz"):
            get_tool("missing")

    def test_error_message_shows_none_when_empty(self) -> None:
        with pytest.raises(KeyError, match=r"\(none\)"):
            get_tool("anything")


class TestListTools:
    def test_returns_all_when_no_filter(self) -> None:
        TOOL_REGISTRY["sp"] = _make_entry(category=ToolCategory.STRUCTURE_PREDICTION)
        TOOL_REGISTRY["emb"] = _make_entry(category=ToolCategory.EMBEDDING)
        result = list_tools()
        assert set(result.keys()) == {"sp", "emb"}

    def test_filters_by_category(self) -> None:
        TOOL_REGISTRY["sp"] = _make_entry(category=ToolCategory.STRUCTURE_PREDICTION)
        TOOL_REGISTRY["emb"] = _make_entry(category=ToolCategory.EMBEDDING)
        result = list_tools(category=ToolCategory.EMBEDDING)
        assert list(result.keys()) == ["emb"]

    def test_returns_empty_dict_when_no_match(self) -> None:
        TOOL_REGISTRY["sp"] = _make_entry(category=ToolCategory.STRUCTURE_PREDICTION)
        result = list_tools(category=ToolCategory.SCORING)
        assert result == {}

    def test_empty_registry(self) -> None:
        assert list_tools() == {}
        assert list_tools(category=ToolCategory.EMBEDDING) == {}

    def test_returns_copy(self) -> None:
        TOOL_REGISTRY["sp"] = _make_entry()
        result = list_tools()
        result["injected"] = _make_entry()  # type: ignore[assignment]
        assert "injected" not in TOOL_REGISTRY
