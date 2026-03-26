"""Tool registry — maps tool names to metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autobio.schemas.base import BaseInput, BaseOutput


class ToolCategory(StrEnum):
    """Functional categories for computational biology tools."""

    STRUCTURE_PREDICTION = "structure-prediction"
    EMBEDDING = "embedding"
    INVERSE_FOLDING = "inverse-folding"
    SCORING = "scoring"
    STRUCTURE_DESIGN = "structure-design"
    SIMULATION = "simulation"


@dataclass
class ToolEntry:
    """Metadata for a registered tool.

    Each entry describes one tool available in the registry: its container
    image, category, resource requirements, I/O schemas, and human-readable
    description.
    """

    image_tag: str
    category: ToolCategory
    requires_gpu: bool
    gpu_count: int
    input_schema: type[BaseInput]
    output_schema: type[BaseOutput]
    default_timeout: int
    supports_batch: bool
    description: str
    version: str
    notes: tuple[str, ...] = ()
    input_format: tuple[str, ...] = ()


TOOL_REGISTRY: dict[str, ToolEntry] = {}
"""Global mapping of tool names to their :class:`ToolEntry` metadata.

Empty at import time; populated when specific tool modules are loaded.
"""


def get_tool(name: str) -> ToolEntry:
    """Look up a tool by name.

    Args:
        name: Registered tool name.

    Returns:
        The matching :class:`ToolEntry`.

    Raises:
        KeyError: If the tool is not registered, with a message listing
            available tools.
    """
    try:
        return TOOL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(TOOL_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown tool {name!r}. Available tools: {available}") from None


def list_tools(category: ToolCategory | None = None) -> dict[str, ToolEntry]:
    """Return registered tools, optionally filtered by category.

    Args:
        category: If provided, only tools in this category are returned.

    Returns:
        A dict mapping tool names to their :class:`ToolEntry` metadata.
    """
    if category is None:
        return dict(TOOL_REGISTRY)
    return {name: entry for name, entry in TOOL_REGISTRY.items() if entry.category == category}
