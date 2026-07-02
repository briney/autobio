"""Tests for the Tool/Mode catalog."""

from __future__ import annotations

import pytest

from autobio.core.catalog import (
    CATALOG,
    Mode,
    Tool,
    get_tool,
    list_tools,
    register,
    tool_categories,
)
from autobio.core.registry import ToolCategory
from autobio.schemas.base import BaseInput, BaseOutput


@pytest.fixture(autouse=True)
def _clean_catalog():
    """Snapshot, clear, and restore CATALOG around each test."""
    snapshot = dict(CATALOG)
    CATALOG.clear()
    yield
    CATALOG.clear()
    CATALOG.update(snapshot)


def _mode(name: str, category: ToolCategory | None = None) -> Mode:
    return Mode(
        name=name,
        display_name=name.title(),
        description=f"{name} mode",
        input_schema=BaseInput,
        output_schema=BaseOutput,
        default_timeout=600,
        category=category,
    )


def _tool(name: str = "demo", **overrides) -> Tool:
    kwargs = dict(
        name=name,
        display_name=name.title(),
        category=ToolCategory.EMBEDDING,
        description="demo tool",
        version="1.0.0",
        image_tag=f"{name}:1.0.0",
        requires_gpu=True,
        gpu_count=1,
        modes={"embed": _mode("embed")},
        default_mode="embed",
    )
    kwargs.update(overrides)
    return Tool(**kwargs)


def test_tool_requires_default_mode_to_exist():
    with pytest.raises(ValueError, match="default_mode"):
        _tool(default_mode="missing")


def test_tool_requires_nonempty_modes():
    with pytest.raises(ValueError, match="at least one mode"):
        _tool(modes={}, default_mode="embed")


def test_register_and_get_tool():
    tool = _tool()
    register(tool)
    assert get_tool("demo") is tool


def test_register_rejects_duplicate():
    register(_tool())
    with pytest.raises(ValueError, match="already registered"):
        register(_tool())


def test_get_tool_unknown_lists_available():
    register(_tool("alpha"))
    with pytest.raises(KeyError, match="alpha"):
        get_tool("nope")


def test_list_tools_returns_copy():
    register(_tool())
    result = list_tools()
    result.clear()
    assert "demo" in CATALOG


def test_tool_categories_union_of_primary_and_mode_overrides():
    tool = _tool(
        category=ToolCategory.INVERSE_FOLDING,
        modes={
            "design": _mode("design"),
            "score": _mode("score", category=ToolCategory.SCORING),
        },
        default_mode="design",
    )
    assert tool_categories(tool) == (
        ToolCategory.INVERSE_FOLDING,
        ToolCategory.SCORING,
    )


def test_list_tools_filters_by_mode_override_category():
    tool = _tool(
        category=ToolCategory.INVERSE_FOLDING,
        modes={
            "design": _mode("design"),
            "score": _mode("score", category=ToolCategory.SCORING),
        },
        default_mode="design",
    )
    register(tool)
    assert "demo" in list_tools(category=ToolCategory.SCORING)
    assert "demo" in list_tools(category=ToolCategory.INVERSE_FOLDING)
    assert "demo" not in list_tools(category=ToolCategory.SIMULATION)
