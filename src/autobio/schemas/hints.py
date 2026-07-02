"""UI-hint vocabulary for tool input schemas.

Hints ride inside the JSON Schema under a single namespaced ``x-autobio`` object
attached via Pydantic ``Field(json_schema_extra=ui(...))``. They are presentation
only — consumers that don't recognize a hint fall back to type-driven rendering
and treat unknown fields as ``tier="advanced"``. Hints never affect validation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Tier(StrEnum):
    """Whether a field surfaces on the main form or under 'Advanced'."""

    PRIMARY = "primary"
    ADVANCED = "advanced"


class Widget(StrEnum):
    """Preferred UI control for a field (a hint; consumers may override by type)."""

    TOGGLE = "toggle"
    SELECT = "select"
    SLIDER = "slider"
    NUMBER = "number"
    TEXT = "text"
    TEXTAREA = "textarea"
    SEQUENCE = "sequence"
    FILE = "file"


def ui(
    *,
    tier: Tier | str | None = None,
    widget: Widget | str | None = None,
    group: str | None = None,
    order: int | None = None,
    unit: str | None = None,
    step: float | None = None,
    enum_labels: dict[str, str] | None = None,
    flavor: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an ``x-autobio`` hint object for ``Field(json_schema_extra=...)``.

    Only the arguments you pass appear in the result; ``Tier``/``Widget`` enums
    are coerced to their string values.

    Returns:
        ``{"x-autobio": {<provided keys>}}``.
    """
    hint: dict[str, Any] = {}
    if tier is not None:
        hint["tier"] = str(tier)
    if widget is not None:
        hint["widget"] = str(widget)
    if group is not None:
        hint["group"] = group
    if order is not None:
        hint["order"] = order
    if unit is not None:
        hint["unit"] = unit
    if step is not None:
        hint["step"] = step
    if enum_labels is not None:
        hint["enum_labels"] = enum_labels
    if flavor is not None:
        hint["flavor"] = flavor
    return {"x-autobio": hint}
