# Tools→Modes Refactor — Plan 1: Foundation (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the purely-additive foundation for the Tools→Modes refactor — a `Tool`/`Mode` catalog, category taxonomy metadata, the `x-autobio` UI-hint helper, and `SequenceSet` FASTA-accepting input types — without touching the existing registry, runners, or CLI, so every existing test stays green.

**Architecture:** All new code lives in new units (`core/catalog.py`, `schemas/hints.py`, `schemas/sequences.py`) plus additive functions in `utils/sequences.py`. Nothing consumes the foundation yet; the tool-family migration (Plan 2) wires runners and schemas onto it, and the CLI/contracts (Plan 3) surface it. The old `core/registry.py`/`ToolEntry` path is untouched by this plan and removed at the end of Plan 2.

**Tech Stack:** Python 3.11+, Pydantic v2, dataclasses, `pytest`. No new dependencies. FASTA parsing stays pure-Python (no BioPython), matching the existing `utils/sequences.py`.

## Global Constraints

- Python 3.11+; modern syntax (`X | Y` unions, `StrEnum`, `match`). `from __future__ import annotations` at the top of every module.
- Type hints on all signatures; Google-style docstrings on all public classes/functions.
- Max line length 100. Formatter `ruff format`; linter `ruff check`; config in `pyproject.toml`.
- `pathlib.Path` over `os.path`; `str()` only at IO boundaries.
- Absolute imports only; no wildcard imports; no business logic in `__init__.py`.
- Tests: `pytest`, mirror source layout under `tests/unit/`. Use `@pytest.mark.parametrize` for input variation. No Docker/GPU needed for any task in this plan.
- No changes to container execution, the workspace protocol, `result.json`, or GPU allocation.
- Do NOT modify `core/registry.py`, `tools/`, or `cli/` in this plan. Foundation is additive only.
- Canonical `ToolCategory` values (verbatim): `STRUCTURE_PREDICTION="structure-prediction"`, `EMBEDDING="embedding"`, `INVERSE_FOLDING="inverse-folding"`, `SCORING="scoring"`, `STRUCTURE_DESIGN="structure-design"`, `SIMULATION="simulation"`.
- Canonical antibody model (existing, `schemas/antibody.py`): `AntibodySequence(id: str, heavy_chain: str | None = None, light_chain: str | None = None)` with an "at least one chain" `model_validator`.

---

### Task 1: `Mode` and `Tool` dataclasses + catalog registry

**Files:**
- Create: `src/autobio/core/catalog.py`
- Test: `tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `autobio.core.registry.ToolCategory` (existing `StrEnum`); `autobio.schemas.base.BaseInput`/`BaseOutput` (for type-only annotations).
- Produces:
  - `Mode` frozen dataclass: `name: str`, `display_name: str`, `description: str`, `input_schema: type[BaseInput]`, `output_schema: type[BaseOutput]`, `default_timeout: int`, `supports_batch: bool = False`, `category: ToolCategory | None = None`, `notes: tuple[str, ...] = ()`.
  - `Tool` frozen dataclass: `name: str`, `display_name: str`, `category: ToolCategory`, `description: str`, `version: str`, `image_tag: str`, `requires_gpu: bool`, `gpu_count: int`, `modes: dict[str, Mode]`, `default_mode: str`, `keywords: tuple[str, ...] = ()`, `notes: tuple[str, ...] = ()`. `__post_init__` raises `ValueError` if `modes` is empty or `default_mode not in modes`.
  - `CATALOG: dict[str, Tool]` (module-global, empty at import).
  - `register(tool: Tool) -> None` — raises `ValueError` on duplicate `tool.name`.
  - `get_tool(name: str) -> Tool` — raises `KeyError` listing available tools.
  - `list_tools(category: ToolCategory | None = None) -> dict[str, Tool]` — returns a copy; when `category` is given, includes a Tool if its primary `category` OR any mode's overriding `category` equals it.
  - `tool_categories(tool: Tool) -> tuple[ToolCategory, ...]` — insertion-ordered union of the primary category and all mode category overrides.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_catalog.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autobio.core.catalog'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/autobio/core/catalog.py
"""Tool/Mode catalog — the Tools→Modes registry.

A ``Tool`` is one coherent model or engine (one catalog card). A ``Mode`` is a
named use of a Tool (a task/operation) that owns its own resolved input/output
schemas and execution metadata. This module is the additive successor to
``core.registry``; the flat ``TOOL_REGISTRY`` is removed once all tools migrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autobio.core.registry import ToolCategory

if TYPE_CHECKING:
    from autobio.schemas.base import BaseInput, BaseOutput


@dataclass(frozen=True)
class Mode:
    """A named use (task/operation) of a Tool."""

    name: str
    display_name: str
    description: str
    input_schema: type[BaseInput]
    output_schema: type[BaseOutput]
    default_timeout: int
    supports_batch: bool = False
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
```

Note: `field` is imported for parity with later edits; if `ruff check` flags it as unused, remove the `field` import.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_catalog.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint, format, and confirm nothing else broke**

Run: `ruff check --fix src/autobio/core/catalog.py tests/unit/test_catalog.py && ruff format src/autobio/core/catalog.py tests/unit/test_catalog.py && pytest tests/unit/test_catalog.py tests/unit/test_registry.py -q`
Expected: clean lint/format; all tests pass (old `test_registry.py` untouched and green).

- [ ] **Step 6: Commit**

```bash
git add src/autobio/core/catalog.py tests/unit/test_catalog.py
git commit -m "catalog: add Tool/Mode dataclasses and catalog registry"
```

---

### Task 2: Category taxonomy metadata

**Files:**
- Modify: `src/autobio/core/catalog.py`
- Test: `tests/unit/test_catalog.py`

**Interfaces:**
- Consumes: `ToolCategory` (all six members).
- Produces:
  - `CategoryInfo` frozen dataclass: `category: ToolCategory`, `label: str`, `description: str`, `order: int`, `icon: str | None = None`.
  - `get_category_info(category: ToolCategory) -> CategoryInfo`.
  - `list_categories() -> list[CategoryInfo]` — all six, sorted by `order`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_catalog.py

from autobio.core.catalog import CategoryInfo, get_category_info, list_categories  # noqa: E402


def test_list_categories_covers_all_members_sorted_by_order():
    cats = list_categories()
    assert {c.category for c in cats} == set(ToolCategory)
    orders = [c.order for c in cats]
    assert orders == sorted(orders)
    assert len(orders) == len(set(orders))  # orders are unique


def test_get_category_info_returns_labelled_entry():
    info = get_category_info(ToolCategory.EMBEDDING)
    assert isinstance(info, CategoryInfo)
    assert info.category is ToolCategory.EMBEDDING
    assert info.label and info.description
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_catalog.py -k category -v`
Expected: FAIL with `ImportError: cannot import name 'CategoryInfo'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/autobio/core/catalog.py` (after the `Mode`/`Tool` definitions):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_catalog.py -v`
Expected: PASS (all catalog tests)

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/core/catalog.py tests/unit/test_catalog.py
ruff format src/autobio/core/catalog.py tests/unit/test_catalog.py
git add src/autobio/core/catalog.py tests/unit/test_catalog.py
git commit -m "catalog: add category taxonomy display metadata"
```

---

### Task 3: `x-autobio` UI-hint helper

**Files:**
- Create: `src/autobio/schemas/hints.py`
- Test: `tests/unit/test_hints.py`

**Interfaces:**
- Produces:
  - `Tier(StrEnum)`: `PRIMARY="primary"`, `ADVANCED="advanced"`.
  - `Widget(StrEnum)`: `TOGGLE`, `SELECT`, `SLIDER`, `NUMBER`, `TEXT`, `TEXTAREA`, `SEQUENCE`, `FILE` (values = lowercase names).
  - `ui(*, tier=None, widget=None, group=None, order=None, unit=None, step=None, enum_labels=None, flavor=None) -> dict[str, dict[str, object]]` — returns `{"x-autobio": {<only the provided keys>}}`, coercing `Tier`/`Widget` enums to their string values. Intended for `Field(json_schema_extra=ui(...))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_hints.py
"""Tests for the x-autobio UI-hint helper."""

from __future__ import annotations

from pydantic import BaseModel, Field

from autobio.schemas.hints import Tier, Widget, ui


def test_ui_emits_only_provided_keys_under_namespace():
    assert ui(tier=Tier.PRIMARY, widget=Widget.SELECT) == {
        "x-autobio": {"tier": "primary", "widget": "select"}
    }


def test_ui_coerces_enums_and_passes_scalars():
    hint = ui(tier=Tier.ADVANCED, order=2, unit="Å", step=0.5, flavor="antibody")
    assert hint == {
        "x-autobio": {
            "tier": "advanced",
            "order": 2,
            "unit": "Å",
            "step": 0.5,
            "flavor": "antibody",
        }
    }


def test_ui_omits_none_values():
    assert ui(widget=Widget.TOGGLE) == {"x-autobio": {"widget": "toggle"}}


def test_hint_surfaces_in_model_json_schema():
    class M(BaseModel):
        per_position: bool = Field(default=False, json_schema_extra=ui(tier=Tier.PRIMARY, widget=Widget.TOGGLE))

    schema = M.model_json_schema()
    assert schema["properties"]["per_position"]["x-autobio"] == {
        "tier": "primary",
        "widget": "toggle",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autobio.schemas.hints'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/autobio/schemas/hints.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_hints.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/schemas/hints.py tests/unit/test_hints.py
ruff format src/autobio/schemas/hints.py tests/unit/test_hints.py
git add src/autobio/schemas/hints.py tests/unit/test_hints.py
git commit -m "schemas: add x-autobio UI-hint helper (Tier/Widget/ui)"
```

---

### Task 4: Generic FASTA string parsing in `utils/sequences.py`

**Files:**
- Modify: `src/autobio/utils/sequences.py`
- Test: `tests/unit/test_sequences.py`

**Interfaces:**
- Consumes: existing `parse_fasta(path: Path) -> dict[str, str]` (its per-line logic is refactored into the new string parser to stay DRY).
- Produces: `parse_fasta_string(text: str) -> dict[str, str]` — parses FASTA text into an insertion-ordered `{id: sequence}`; raises `ValueError` naming a duplicate id; raises `ValueError` if a sequence line precedes any header. `parse_fasta` is refactored to read the file and delegate to `parse_fasta_string` (behavior unchanged).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_sequences.py

from autobio.utils.sequences import parse_fasta_string  # noqa: E402


def test_parse_fasta_string_basic():
    text = ">a\nMKT\nVLL\n>b\nGGG\n"
    assert parse_fasta_string(text) == {"a": "MKTVLL", "b": "GGG"}


def test_parse_fasta_string_rejects_duplicate_ids():
    import pytest

    with pytest.raises(ValueError, match="[Dd]uplicate.*'a'"):
        parse_fasta_string(">a\nMKT\n>a\nGGG\n")


def test_parse_fasta_string_rejects_sequence_before_header():
    import pytest

    with pytest.raises(ValueError, match="before any header"):
        parse_fasta_string("MKT\n>a\nGGG\n")


def test_parse_fasta_string_ignores_blank_lines():
    text = "\n>a\nMKT\n\n\n>b\nGGG\n\n"
    assert parse_fasta_string(text) == {"a": "MKT", "b": "GGG"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sequences.py -k parse_fasta_string -v`
Expected: FAIL with `ImportError: cannot import name 'parse_fasta_string'`

- [ ] **Step 3: Write minimal implementation**

In `src/autobio/utils/sequences.py`, add `parse_fasta_string` and refactor `parse_fasta` to delegate. Replace the existing `parse_fasta` body with:

```python
def parse_fasta_string(text: str) -> dict[str, str]:
    """Parse FASTA text into an insertion-ordered ``{id: sequence}`` mapping.

    Args:
        text: Raw FASTA content. ``>``-prefixed lines are headers (the leading
            ``>`` is stripped); subsequent non-blank lines are concatenated.

    Returns:
        Mapping of header id to sequence, in first-seen order.

    Raises:
        ValueError: On a duplicate id, or a sequence line before any header.
    """
    sequences: dict[str, str] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].strip()
            if current in sequences:
                raise ValueError(f"Duplicate sequence id {current!r} in FASTA input.")
            sequences[current] = ""
        else:
            if current is None:
                raise ValueError("FASTA sequence data appears before any header line.")
            sequences[current] += line
    return sequences


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into an insertion-ordered ``{id: sequence}`` mapping.

    Args:
        path: Path to a FASTA file.

    Returns:
        Mapping of header id to sequence, in first-seen order.

    Raises:
        ValueError: On a duplicate id, or a sequence line before any header.
    """
    return parse_fasta_string(path.read_text())
```

Note: if the existing `parse_fasta` did not previously raise on duplicate ids, confirm no existing test in `test_sequences.py` asserts the old silent-overwrite behavior; the new duplicate-id error is intended (spec §6.4). If such a test exists, update it to expect the `ValueError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sequences.py -v`
Expected: PASS (new tests plus all pre-existing `test_sequences.py` tests)

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/utils/sequences.py tests/unit/test_sequences.py
ruff format src/autobio/utils/sequences.py tests/unit/test_sequences.py
git add src/autobio/utils/sequences.py tests/unit/test_sequences.py
git commit -m "sequences: add parse_fasta_string and delegate parse_fasta to it"
```

---

### Task 5: Antibody FASTA pairing parser

**Files:**
- Modify: `src/autobio/utils/sequences.py`
- Test: `tests/unit/test_sequences.py`

**Interfaces:**
- Consumes: `parse_fasta_string` (Task 4); `autobio.schemas.antibody.AntibodySequence`; existing `validate_antibody_sequence`.
- Produces:
  - `normalize_chain_token(token: str) -> str` — maps a case-insensitive chain token to `"heavy"` or `"light"`; accepts aliases `heavy`/`h`/`vh` and `light`/`l`/`vl`; raises `ValueError` on an unknown token.
  - `parse_antibody_fasta_string(text: str) -> list[AntibodySequence]` — parses `>{pair_id}|{chain}` records into paired `AntibodySequence` objects. Raises `ValueError` naming the offending record for: a header without a `|` chain tag, an unknown chain token, a duplicate `(pair_id, chain)`, or a sequence failing `validate_antibody_sequence`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_sequences.py
import pytest  # noqa: E402

from autobio.schemas.antibody import AntibodySequence  # noqa: E402
from autobio.utils.sequences import (  # noqa: E402
    normalize_chain_token,
    parse_antibody_fasta_string,
)


@pytest.mark.parametrize(
    "token,expected",
    [("heavy", "heavy"), ("H", "heavy"), ("VH", "heavy"),
     ("light", "light"), ("l", "light"), ("Vl", "light")],
)
def test_normalize_chain_token_aliases(token, expected):
    assert normalize_chain_token(token) == expected


def test_normalize_chain_token_unknown():
    with pytest.raises(ValueError, match="chain token"):
        normalize_chain_token("kappa")


def test_parse_antibody_fasta_pairs_and_unpaired():
    text = (
        ">ab1|heavy\nQVQLVQSG\n"
        ">ab1|light\nDIQMTQSP\n"
        ">ab2|heavy\nEVQLLESG\n"
    )
    result = parse_antibody_fasta_string(text)
    assert result == [
        AntibodySequence(id="ab1", heavy_chain="QVQLVQSG", light_chain="DIQMTQSP"),
        AntibodySequence(id="ab2", heavy_chain="EVQLLESG"),
    ]


def test_parse_antibody_fasta_missing_chain_tag():
    with pytest.raises(ValueError, match="ab1"):
        parse_antibody_fasta_string(">ab1\nQVQLVQSG\n")


def test_parse_antibody_fasta_duplicate_pair_chain():
    text = ">ab1|heavy\nQVQLVQSG\n>ab1|heavy\nEVQLLESG\n"
    with pytest.raises(ValueError, match="ab1.*heavy"):
        parse_antibody_fasta_string(text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sequences.py -k "chain_token or antibody_fasta" -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_chain_token'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/autobio/utils/sequences.py`:

```python
_HEAVY_TOKENS = {"heavy", "h", "vh"}
_LIGHT_TOKENS = {"light", "l", "vl"}


def normalize_chain_token(token: str) -> str:
    """Map a case-insensitive chain token to ``"heavy"`` or ``"light"``.

    Accepts aliases ``heavy``/``h``/``vh`` and ``light``/``l``/``vl``.

    Raises:
        ValueError: If the token is not a recognized chain identifier.
    """
    key = token.strip().lower()
    if key in _HEAVY_TOKENS:
        return "heavy"
    if key in _LIGHT_TOKENS:
        return "light"
    raise ValueError(
        f"Unknown chain token {token!r}. Expected one of "
        f"{sorted(_HEAVY_TOKENS | _LIGHT_TOKENS)}."
    )


def parse_antibody_fasta_string(text: str) -> list[AntibodySequence]:
    """Parse antibody FASTA text into paired :class:`AntibodySequence` objects.

    Headers encode a pair id and a chain: ``>{pair_id}|{chain}``. Records sharing
    a ``pair_id`` are paired into one antibody; a lone record becomes an unpaired
    antibody (that chain only).

    Raises:
        ValueError: For a header without a ``|`` chain tag, an unknown chain
            token, a duplicate ``(pair_id, chain)``, or a non-protein sequence.
    """
    raw = parse_fasta_string(text)
    chains: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for header, seq in raw.items():
        if "|" not in header:
            raise ValueError(
                f"Antibody FASTA header {header!r} is missing a chain tag "
                f"(expected '{{pair_id}}|{{chain}}')."
            )
        pair_id, _, chain_token = header.rpartition("|")
        pair_id = pair_id.strip()
        chain = normalize_chain_token(chain_token)
        if not validate_antibody_sequence(seq):
            raise ValueError(f"Record {header!r}: sequence contains non-protein characters.")
        if pair_id not in chains:
            chains[pair_id] = {}
            order.append(pair_id)
        if chain in chains[pair_id]:
            raise ValueError(f"Duplicate record for pair {pair_id!r} chain {chain!r}.")
        chains[pair_id][chain] = seq

    return [
        AntibodySequence(
            id=pair_id,
            heavy_chain=chains[pair_id].get("heavy"),
            light_chain=chains[pair_id].get("light"),
        )
        for pair_id in order
    ]
```

Add the required import at the top of the module (with the other imports, respecting isort grouping — this is a local import from `autobio.schemas`, so it goes in the local group):

```python
from autobio.schemas.antibody import AntibodySequence
```

Note on import safety: `schemas/antibody.py` imports only `pydantic` and `schemas/base.py` — it does NOT import `utils/sequences`, so this import creates no cycle. Verify with `python -c "import autobio.utils.sequences"` after the edit.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sequences.py -v && python -c "import autobio.utils.sequences"`
Expected: PASS (all sequence tests); import succeeds with no `ImportError`/circular-import error.

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/utils/sequences.py tests/unit/test_sequences.py
ruff format src/autobio/utils/sequences.py tests/unit/test_sequences.py
git add src/autobio/utils/sequences.py tests/unit/test_sequences.py
git commit -m "sequences: add antibody FASTA header-tagged pairing parser"
```

---

### Task 6: Generic `SequenceSet` accepting input type

**Files:**
- Create: `src/autobio/schemas/sequences.py`
- Test: `tests/unit/test_sequence_set.py`

**Interfaces:**
- Consumes: `parse_fasta`, `parse_fasta_string` (Tasks 4).
- Produces:
  - `normalize_generic_sequences(value: object) -> dict[str, str]` — accepts a `dict[str, str]` (returned as `{str: str}`), FASTA text (a `str` starting with `>` or containing a newline), or a FASTA file path (a `str` ending `.fasta`/`.fa`, read from disk). Raises `ValueError` for any other input, or when a path-looking string does not exist.
  - `GenericSequenceSet = Annotated[dict[str, str], BeforeValidator(normalize_generic_sequences)]` — usable as a Pydantic field type; the canonical stored form is `dict[str, str]`, so existing JSON callers are unaffected.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sequence_set.py
"""Tests for SequenceSet accepting input types."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from autobio.schemas.sequences import GenericSequenceSet, normalize_generic_sequences


class _Generic(BaseModel):
    sequences: GenericSequenceSet


def test_generic_accepts_native_dict():
    assert _Generic(sequences={"a": "MKT"}).sequences == {"a": "MKT"}


def test_generic_accepts_fasta_text():
    assert _Generic(sequences=">a\nMKT\n>b\nGGG\n").sequences == {"a": "MKT", "b": "GGG"}


def test_generic_accepts_fasta_file(tmp_path):
    f = tmp_path / "seqs.fasta"
    f.write_text(">a\nMKT\n")
    assert _Generic(sequences=str(f)).sequences == {"a": "MKT"}


def test_generic_rejects_missing_file():
    with pytest.raises((ValidationError, ValueError)):
        _Generic(sequences="/no/such/path.fasta")


def test_generic_rejects_bad_type():
    with pytest.raises((ValidationError, ValueError)):
        _Generic(sequences=12345)


def test_normalize_is_idempotent_on_dict():
    assert normalize_generic_sequences({"a": "MKT"}) == {"a": "MKT"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sequence_set.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'autobio.schemas.sequences'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/autobio/schemas/sequences.py
"""SequenceSet input types — accept structured JSON, FASTA text, or a FASTA file.

Each SequenceSet's field type is the canonical structured form (so existing JSON
callers and agents are unaffected); a ``BeforeValidator`` additionally accepts
FASTA text or a ``.fasta``/``.fa`` file path and normalizes it centrally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from autobio.utils.sequences import parse_fasta, parse_fasta_string

_FASTA_SUFFIXES = (".fasta", ".fa")


def _looks_like_fasta_text(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith(">") or "\n" in value


def _looks_like_fasta_path(value: str) -> bool:
    return value.strip().lower().endswith(_FASTA_SUFFIXES)


def normalize_generic_sequences(value: object) -> dict[str, str]:
    """Normalize a generic sequence input to a ``{id: sequence}`` mapping.

    Accepts a native ``dict[str, str]``, FASTA text, or a ``.fasta``/``.fa`` path.

    Raises:
        ValueError: For an unsupported input type, or a path that does not exist.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        if _looks_like_fasta_text(value):
            return parse_fasta_string(value)
        if _looks_like_fasta_path(value):
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"FASTA file not found: {value!r}.")
            return parse_fasta(path)
        raise ValueError(
            "String sequence input must be FASTA text (starting with '>' or "
            "multi-line) or a path to a .fasta/.fa file."
        )
    raise ValueError(
        f"Unsupported sequence input type {type(value).__name__!r}; expected a "
        "dict, FASTA text, or a FASTA file path."
    )


GenericSequenceSet = Annotated[dict[str, str], BeforeValidator(normalize_generic_sequences)]
"""Field type accepting ``dict[str, str]``, FASTA text, or a FASTA file path."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sequence_set.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/schemas/sequences.py tests/unit/test_sequence_set.py
ruff format src/autobio/schemas/sequences.py tests/unit/test_sequence_set.py
git add src/autobio/schemas/sequences.py tests/unit/test_sequence_set.py
git commit -m "schemas: add GenericSequenceSet accepting structured/FASTA/file input"
```

---

### Task 7: Antibody `SequenceSet` accepting input type

**Files:**
- Modify: `src/autobio/schemas/sequences.py`
- Test: `tests/unit/test_sequence_set.py`

**Interfaces:**
- Consumes: `parse_antibody_fasta_string` (Task 5); `Path` and `_looks_like_fasta_text`/`_looks_like_fasta_path`/`_FASTA_SUFFIXES` (Task 6); `autobio.schemas.antibody.AntibodySequence`.
- Produces:
  - `normalize_antibody_sequences(value: object) -> list[AntibodySequence]` — accepts a `list` of `AntibodySequence` or dicts (coerced to `AntibodySequence`), FASTA text, or a `.fasta`/`.fa` file path. Raises `ValueError` for unsupported input or a missing file.
  - `AntibodySequenceSet = Annotated[list[AntibodySequence], BeforeValidator(normalize_antibody_sequences)]`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_sequence_set.py
from autobio.schemas.antibody import AntibodySequence  # noqa: E402
from autobio.schemas.sequences import AntibodySequenceSet  # noqa: E402


class _Ab(BaseModel):
    sequences: AntibodySequenceSet


def test_antibody_accepts_native_list_of_models():
    ab = AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")
    assert _Ab(sequences=[ab]).sequences == [ab]


def test_antibody_accepts_list_of_dicts():
    got = _Ab(sequences=[{"id": "ab1", "heavy_chain": "QVQLVQSG"}]).sequences
    assert got == [AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")]


def test_antibody_accepts_fasta_text_with_pairing():
    text = ">ab1|heavy\nQVQLVQSG\n>ab1|light\nDIQMTQSP\n"
    assert _Ab(sequences=text).sequences == [
        AntibodySequence(id="ab1", heavy_chain="QVQLVQSG", light_chain="DIQMTQSP")
    ]


def test_antibody_accepts_fasta_file(tmp_path):
    f = tmp_path / "ab.fa"
    f.write_text(">ab1|heavy\nQVQLVQSG\n")
    assert _Ab(sequences=str(f)).sequences == [
        AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")
    ]


def test_antibody_rejects_bad_type():
    with pytest.raises((ValidationError, ValueError)):
        _Ab(sequences=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sequence_set.py -k antibody -v`
Expected: FAIL with `ImportError: cannot import name 'AntibodySequenceSet'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/autobio/schemas/sequences.py`. Add the import (local group, top of file):

```python
from autobio.schemas.antibody import AntibodySequence
from autobio.utils.sequences import parse_antibody_fasta_string, parse_fasta, parse_fasta_string
```

(Merge the `utils.sequences` import with the existing one from Task 6 into a single line as shown.) Then append:

```python
def normalize_antibody_sequences(value: object) -> list[AntibodySequence]:
    """Normalize an antibody sequence input to a list of :class:`AntibodySequence`.

    Accepts a native list of ``AntibodySequence`` (or dicts coerced to them),
    FASTA text with ``>{pair_id}|{chain}`` pairing, or a ``.fasta``/``.fa`` path.

    Raises:
        ValueError: For an unsupported input type, or a path that does not exist.
    """
    if isinstance(value, list):
        out: list[AntibodySequence] = []
        for item in value:
            if isinstance(item, AntibodySequence):
                out.append(item)
            elif isinstance(item, dict):
                out.append(AntibodySequence(**item))
            else:
                raise ValueError(
                    f"Antibody list items must be AntibodySequence or dict, got "
                    f"{type(item).__name__!r}."
                )
        return out
    if isinstance(value, str):
        if _looks_like_fasta_text(value):
            return parse_antibody_fasta_string(value)
        if _looks_like_fasta_path(value):
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"FASTA file not found: {value!r}.")
            return parse_antibody_fasta_string(path.read_text())
        raise ValueError(
            "String antibody input must be FASTA text (starting with '>' or "
            "multi-line) or a path to a .fasta/.fa file."
        )
    raise ValueError(
        f"Unsupported antibody sequence input type {type(value).__name__!r}; expected "
        "a list, FASTA text, or a FASTA file path."
    )


AntibodySequenceSet = Annotated[
    list[AntibodySequence], BeforeValidator(normalize_antibody_sequences)
]
"""Field type accepting a list of AntibodySequence/dicts, FASTA text, or a FASTA file path."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sequence_set.py -v && python -c "import autobio.schemas.sequences"`
Expected: PASS (all SequenceSet tests); clean import (no circular-import error — `schemas/antibody.py` does not import `schemas/sequences.py`).

- [ ] **Step 5: Full foundation regression + lint/format**

Run: `ruff check --fix src/autobio tests/unit && ruff format src/autobio tests/unit && pytest -m "not docker and not gpu" -q && mypy src/`
Expected: clean lint/format; full non-docker/gpu suite passes (foundation additive, nothing pre-existing broken); mypy clean.

- [ ] **Step 6: Commit**

```bash
git add src/autobio/schemas/sequences.py tests/unit/test_sequence_set.py
git commit -m "schemas: add AntibodySequenceSet with FASTA pairing input"
```

---

## Self-Review

**1. Spec coverage (against `docs/superpowers/specs/2026-07-02-autobio-tools-modes-refactor-design.md`, Phase 0 scope):**
- §4 `Tool`/`Mode` data model + registry → Task 1. ✓
- §4.1 category taxonomy metadata → Task 2. ✓
- §7.2 `x-autobio` hint vocabulary (`tier`/`widget`/`group`/`order`/`unit`/`step`/`enum_labels`/`flavor`) + graceful degradation (presentation-only, surfaces in `model_json_schema()`) → Task 3. ✓
- §6.2 FASTA parsing: generic (`>id`, duplicate-id error) → Task 4; antibody `>{pair_id}|{chain}` pairing, chain aliases, record-naming errors → Task 5. ✓
- §6.1 `SequenceSet` `Annotated[Canonical, BeforeValidator]` per flavor, canonical type preserved, accepts structured/FASTA-text/FASTA-file → Tasks 6 & 7. ✓
- Phase 0 constraint "purely additive, existing tests green, don't touch registry/tools/cli" → enforced by Global Constraints and each task's regression step. ✓
- Deferred to later plans (correctly out of this plan): runner mode-dispatch + `self.current_mode` (Plan 2 Task 1), `_CONSUMED_EXTRA_KEYS` promotion, two-class consolidation, `list`/`info`/`run --mode` contracts, README, removal of old `TOOL_REGISTRY`.

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step contains complete test and implementation code. The one conditional note (Task 4, "if such a test exists, update it") is a concrete instruction with the exact expected change, not a placeholder.

**3. Type consistency:** `parse_fasta_string`/`parse_fasta` (Task 4) consumed unchanged by Tasks 5–7. `normalize_chain_token`/`parse_antibody_fasta_string` (Task 5) consumed by Task 7. `_looks_like_fasta_text`/`_looks_like_fasta_path`/`_FASTA_SUFFIXES` defined in Task 6, reused in Task 7 (same module). `AntibodySequence(id, heavy_chain, light_chain)` used consistently with the existing schema. `Tier`/`Widget`/`ui` (Task 3) return the exact `{"x-autobio": {...}}` shape asserted in tests. `Tool`/`Mode`/`get_tool`/`list_tools`/`tool_categories`/`CategoryInfo`/`get_category_info`/`list_categories` names match between Tasks 1–2 and their tests.

---

## Next plans (not in scope here)

- **Plan 2 — Tool-family migration (Phase 1):** base-runner mode dispatch (`run(..., mode=...)` + `self.current_mode`), then migrate families onto the catalog (declare `Tool`+`modes`, split schemas from a shared base, promote `extra` keys with `x-autobio` hints + update `_CONSUMED_EXTRA_KEYS`, swap sequence fields to `SequenceSet`, consolidate `esm_if1`/`antifold` two-class runners), regenerate the authoritative flat→`(tool, mode)` map, and remove the old `TOOL_REGISTRY`/`ToolEntry`. Drafted against the frozen Task 1 APIs.
- **Plan 3 — Contracts/CLI (Phase 2):** `list`/`info`/`run --mode` JSON with `output_schema` and `x-autobio`, category taxonomy in output, `Tool`/`Mode` Python API exports, `info` snapshot tests.
- **Plan 4 — Docs/cleanup (Phase 3):** rewrite README to Tools/Modes, remove flat names, final full-suite pass.
