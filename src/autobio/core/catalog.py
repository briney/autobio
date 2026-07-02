"""Tool/Mode catalog — the Tools→Modes registry.

A ``Tool`` is one coherent model or engine (one catalog card). A ``Mode`` is a
named use of a Tool (a task/operation) that owns its own resolved input/output
schemas and execution metadata. This module is the additive successor to
``core.registry``; the flat ``TOOL_REGISTRY`` is removed once all tools migrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from autobio.core.registry import ToolCategory

if TYPE_CHECKING:
    from autobio.schemas.base import BaseInput, BaseOutput


@dataclass(frozen=True)
class Mode:
    """A named use (task/operation) of a Tool.

    ``image_tag`` overrides the owning Tool's image for this mode (used by
    engines whose modes ship as separate container images); ``None`` falls
    back to ``Tool.image_tag``.
    """

    name: str
    display_name: str
    description: str
    input_schema: type[BaseInput]
    output_schema: type[BaseOutput]
    default_timeout: int
    supports_batch: bool = False
    image_tag: str | None = None
    category: ToolCategory | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    """A coherent model or engine exposing one or more :class:`Mode` uses."""

    name: str
    display_name: str
    category: ToolCategory
    description: str
    version: str
    image_tag: str
    requires_gpu: bool
    gpu_count: int
    modes: dict[str, Mode]
    default_mode: str
    keywords: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError(f"Tool {self.name!r} must declare at least one mode.")
        if self.default_mode not in self.modes:
            raise ValueError(
                f"Tool {self.name!r} default_mode {self.default_mode!r} "
                f"is not among its modes: {sorted(self.modes)}."
            )


@dataclass(frozen=True)
class CategoryInfo:
    """Display metadata for a tool category (drives consumer sidebar submenus)."""

    category: ToolCategory
    label: str
    description: str
    order: int
    icon: str | None = None


_CATEGORY_INFO: dict[ToolCategory, CategoryInfo] = {
    ToolCategory.STRUCTURE_PREDICTION: CategoryInfo(
        ToolCategory.STRUCTURE_PREDICTION,
        "Structure Prediction",
        "Predict 3D structures from sequence.",
        order=1,
    ),
    ToolCategory.STRUCTURE_DESIGN: CategoryInfo(
        ToolCategory.STRUCTURE_DESIGN,
        "Structure Design",
        "Generate or design new structures.",
        order=2,
    ),
    ToolCategory.INVERSE_FOLDING: CategoryInfo(
        ToolCategory.INVERSE_FOLDING,
        "Inverse Folding",
        "Design sequences for a target backbone.",
        order=3,
    ),
    ToolCategory.EMBEDDING: CategoryInfo(
        ToolCategory.EMBEDDING,
        "Embeddings",
        "Extract learned representations and likelihoods from sequences.",
        order=4,
    ),
    ToolCategory.SCORING: CategoryInfo(
        ToolCategory.SCORING,
        "Scoring",
        "Score structures, complexes, or mutations.",
        order=5,
    ),
    ToolCategory.SIMULATION: CategoryInfo(
        ToolCategory.SIMULATION,
        "Simulation",
        "Molecular dynamics and physics-based simulation.",
        order=6,
    ),
}


def get_category_info(category: ToolCategory) -> CategoryInfo:
    """Return display metadata for a category."""
    return _CATEGORY_INFO[category]


def list_categories() -> list[CategoryInfo]:
    """Return all category metadata entries, sorted by display order."""
    return sorted(_CATEGORY_INFO.values(), key=lambda c: c.order)


CATALOG: dict[str, Tool] = {}
"""Global mapping of Tool name to :class:`Tool`. Populated when tool modules load."""


def register(tool: Tool) -> None:
    """Register a Tool in the global catalog.

    Args:
        tool: The Tool to register.

    Raises:
        ValueError: If a Tool with the same name is already registered.
    """
    if tool.name in CATALOG:
        raise ValueError(f"Tool {tool.name!r} is already registered.")
    CATALOG[tool.name] = tool


def get_tool(name: str) -> Tool:
    """Look up a Tool by name.

    Args:
        name: Registered Tool name.

    Returns:
        The matching :class:`Tool`.

    Raises:
        KeyError: If the Tool is not registered, listing available Tools.
    """
    try:
        return CATALOG[name]
    except KeyError:
        available = ", ".join(sorted(CATALOG)) or "(none)"
        raise KeyError(f"Unknown tool {name!r}. Available tools: {available}") from None


def tool_categories(tool: Tool) -> tuple[ToolCategory, ...]:
    """Return the insertion-ordered union of a Tool's primary and mode categories."""
    seen: list[ToolCategory] = [tool.category]
    for mode in tool.modes.values():
        if mode.category is not None and mode.category not in seen:
            seen.append(mode.category)
    return tuple(seen)


def list_tools(category: ToolCategory | None = None) -> dict[str, Tool]:
    """Return registered Tools, optionally filtered by category.

    A Tool matches *category* if its primary category or any mode's overriding
    category equals it (so cross-category Tools surface under each submenu).

    Args:
        category: If provided, only Tools spanning this category are returned.

    Returns:
        A copy mapping Tool names to :class:`Tool`.
    """
    if category is None:
        return dict(CATALOG)
    return {name: t for name, t in CATALOG.items() if category in tool_categories(t)}
