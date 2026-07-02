# Tools→Modes Plan 3 — Recipe Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the tool-migration recipe with the carry-forward items from Plan 2's final whole-branch review, so the remaining ~11 family migrations start from a fail-fast, non-diverging baseline.

**Architecture:** Four independent, additive hardening changes to the *already-proven* catalog path (freesasa + esm are migrated and green): (1) a shared `ToolRunner._apply_extra` helper that rejects `extra` keys shadowing typed fields, replacing the empty-`_CONSUMED_EXTRA_KEYS` merge loops; (2) a disjointness guard test asserting `CATALOG` and `TOOL_REGISTRY` never share a name; (3) standardize the module-level `<TOOL>_TOOL` constant convention (esm gains `ESM1B_TOOL`/`ESM2_TOOL`); (4) a micro-opt so `format_tool_info_catalog` computes JSON schemas only in the JSON branch. No container, workspace, protocol, or GPU changes.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy. No ML dependencies on the host.

## Global Constraints

- Python 3.11+; modern syntax (`|` unions, `match`, `StrEnum`). Max line length **100**.
- Ruff lint select: `E, W, F, I, UP, B, SIM, TCH`. Formatter: `ruff format` (double quotes).
- Type hints on all signatures; Google-style docstrings on public classes/functions.
- Tests run with **`python -m pytest`** (bare `pytest` may use the wrong env). Reinstall editable (`pip install -e ".[dev]"`) if `src/` edits are not picked up.
- Scope is **autobio core only** — no changes to `containers/`, the workspace/`result.json` protocol, `standardize.*`, or GPU allocation. Container `config.json` output for freesasa/esm must remain byte-compatible (their existing config tests must stay green).
- `AutobioError` is the host-side user-error type (`autobio.core.result`); it is already imported in `src/autobio/tools/base.py`.
- Commit message convention: `<component>: <what changed and why>`.
- Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Fail-fast `extra`-shadowing contract (shared base helper)

**Why:** After migration, `_CONSUMED_EXTRA_KEYS` is empty per tool and the `extra` flat-merge runs **last**, so a caller passing a promoted field's name via `extra={...}` silently overrides the typed field in `config.json` (freesasa) or leaks an inert key (esm). A shared helper that rejects any `extra` key naming a typed field on the active mode's input schema fixes this fail-fast — directly satisfying the standing "dict pass-through needs fail-fast validation" feedback — and becomes the one merge path every future migration uses.

**Files:**
- Modify: `src/autobio/tools/base.py` (add `Any` import; add `_apply_extra` method)
- Modify: `src/autobio/tools/freesasa.py` (drop `_CONSUMED_EXTRA_KEYS`; call `_apply_extra`)
- Modify: `src/autobio/tools/esm.py` (drop `_CONSUMED_EXTRA_KEYS`; call `_apply_extra`)
- Test: `tests/unit/test_tool_runner_modes.py` (base-helper behavior)
- Test: `tests/unit/test_freesasa.py`, `tests/unit/test_esm.py` (per-tool shadow + passthrough)

**Interfaces:**
- Produces: `ToolRunner._apply_extra(self, config: dict[str, Any], input_data: BaseInput) -> None` — mutates `config` in place with accepted `extra` keys; raises `AutobioError` if any `extra` key names a typed field (other than `extra`) on `self.current_mode.input_schema`. Requires `self.current_mode is not None` (catalog tools only).
- Consumes: `self.current_mode` (a `Mode`, set by `run()` before `prepare_workspace`), `Mode.input_schema.model_fields`.

- [ ] **Step 1: Write the failing base-helper tests**

Add to `tests/unit/test_tool_runner_modes.py`. Place `_TypedInput` next to the existing `_Input` class (after line 28), and the tests after `test_resolve_mode_unknown_raises` (after line 108):

```python
class _TypedInput(BaseInput):
    alpha_param: int = 0


def _typed_mode() -> Mode:
    return Mode("alpha", "Alpha", "a", _TypedInput, _Output, default_timeout=111)


def test_apply_extra_passes_through_unknown_keys() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner.current_mode = _typed_mode()
    config: dict[str, object] = {"base": 1}
    runner._apply_extra(config, _TypedInput(extra={"beta_param": 7}))
    assert config == {"base": 1, "beta_param": 7}


def test_apply_extra_rejects_shadowing_typed_field() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner.current_mode = _typed_mode()
    with pytest.raises(AutobioError, match="shadow typed input fields: alpha_param"):
        runner._apply_extra({}, _TypedInput(extra={"alpha_param": 5}))
```

- [ ] **Step 2: Run the base-helper tests to verify they fail**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -k apply_extra -v`
Expected: FAIL with `AttributeError: 'ToolRunner' object has no attribute '_apply_extra'` (or `_CaptureRunner`).

- [ ] **Step 3: Add the `Any` import to `base.py`**

In `src/autobio/tools/base.py`, change the typing import (line 8):

```python
from typing import TYPE_CHECKING, Any
```

- [ ] **Step 4: Implement `_apply_extra` in `ToolRunner`**

In `src/autobio/tools/base.py`, insert this method immediately after `_resolve_mode` (i.e., after the block ending at the current line 198, before `def _image_tag`):

```python
    def _apply_extra(self, config: dict[str, Any], input_data: BaseInput) -> None:
        """Merge ``input_data.extra`` into *config*, rejecting typed-field shadows.

        ``extra`` is the escape hatch for parameters not promoted to typed fields
        on a mode's input schema. A key in ``extra`` that names a typed field
        would silently override (or duplicate) that field in ``config.json``, so
        such keys are rejected fail-fast rather than written.

        Args:
            config: The mapping being assembled for ``config.json``; mutated in
                place with the accepted ``extra`` keys.
            input_data: The validated input whose ``extra`` dict is merged.

        Raises:
            AutobioError: If ``extra`` contains a key that shadows a typed field
                on the active mode's input schema.
        """
        assert self.current_mode is not None
        typed_fields = set(self.current_mode.input_schema.model_fields) - {"extra"}
        shadowed = sorted(key for key in input_data.extra if key in typed_fields)
        if shadowed:
            raise AutobioError(
                "extra must not contain keys that shadow typed input fields: "
                f"{', '.join(shadowed)}. Pass these as typed config fields, not via 'extra'."
            )
        config.update(input_data.extra)
```

- [ ] **Step 5: Run the base-helper tests to verify they pass**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -k apply_extra -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Switch freesasa to `_apply_extra`**

In `src/autobio/tools/freesasa.py`:

Delete the module-level constant and its comment (currently lines 41–42):

```python
# All formerly-consumed keys are now typed fields; nothing is stripped from extra.
_CONSUMED_EXTRA_KEYS: frozenset[str] = frozenset()
```

Replace the merge loop in `prepare_workspace` (currently lines 87–89):

```python
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value
```

with:

```python
        self._apply_extra(config, input_data)
```

- [ ] **Step 7: Switch esm to `_apply_extra`**

In `src/autobio/tools/esm.py`:

Delete the module-level constant and its comment (currently lines 58–59):

```python
# All formerly-consumed keys (checkpoint) are now typed fields; nothing is stripped from extra.
_CONSUMED_EXTRA_KEYS: frozenset[str] = frozenset()
```

Replace the merge loop in `prepare_workspace` (currently lines 93–95):

```python
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value
```

with:

```python
        self._apply_extra(config, input_data)
```

- [ ] **Step 8: Add per-tool shadow + passthrough tests**

Append to `tests/unit/test_freesasa.py`:

```python
def test_extra_shadowing_typed_field_rejected(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    with pytest.raises(AutobioError, match="shadow typed input fields"):
        _written_config(
            runner,
            FreeSASASASAInput(structure_path=_pdb, extra={"probe_radius": 2.0}),
            tmp_path,
        )


def test_extra_unknown_key_passed_through(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    cfg = _written_config(
        runner, FreeSASASASAInput(structure_path=_pdb, extra={"custom_flag": True}), tmp_path
    )
    assert cfg["custom_flag"] is True
```

Append to `tests/unit/test_esm.py`:

```python
def test_extra_shadowing_typed_field_rejected(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    with pytest.raises(AutobioError, match="shadow typed input fields"):
        _written_config(
            runner, ESMEmbedInput(sequences={"s1": "MKT"}, extra={"layer": 5}), tmp_path
        )
```

- [ ] **Step 9: Run the freesasa + esm suites**

Run: `python -m pytest tests/unit/test_freesasa.py tests/unit/test_esm.py tests/unit/test_tool_runner_modes.py -v`
Expected: PASS (all, including the pre-existing `test_sasa_config_unchanged` / `test_esm1b_config_model_name` byte-compat tests).

- [ ] **Step 10: Lint, format, type-check**

Run:
```bash
ruff check --fix src/autobio/tools/base.py src/autobio/tools/freesasa.py src/autobio/tools/esm.py tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py
ruff format src/autobio/tools/base.py src/autobio/tools/freesasa.py src/autobio/tools/esm.py tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py
mypy src/
```
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add src/autobio/tools/base.py src/autobio/tools/freesasa.py src/autobio/tools/esm.py \
    tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py
git commit -m "tools: add fail-fast _apply_extra shadow guard; adopt in freesasa/esm"
```

---

### Task 2: Cross-registry disjointness guard

**Why:** A tool name appearing in both `CATALOG` and `TOOL_REGISTRY` would be listed twice by `autobio list` (`format_tool_list_merged` concatenates both maps) and makes lookups ambiguous — the signature of a half-finished migration where a flat entry was not deleted. A dedicated test that imports all tools and asserts disjointness fails loudly the moment that happens. It lives in its own file because `test_catalog.py`/`test_registry.py` clear their registries via autouse fixtures, hiding the real populated state.

**Files:**
- Create: `tests/unit/test_registry_disjoint.py`

**Interfaces:**
- Consumes: `autobio.core.catalog.CATALOG`, `autobio.core.registry.TOOL_REGISTRY` (both populated by importing `autobio.tools`).

- [ ] **Step 1: Write the disjointness test**

Create `tests/unit/test_registry_disjoint.py`:

```python
"""Guard: the catalog and the legacy flat registry must never share a name.

A tool name in both ``CATALOG`` and ``TOOL_REGISTRY`` would be listed twice by
``autobio list`` and makes runner/metadata lookup ambiguous — it signals a
half-finished migration (a flat entry that was not deleted). This test fails
loudly if that ever happens. It deliberately has no registry-clearing fixture:
it must observe the real, fully-populated registries.
"""

from __future__ import annotations


def test_catalog_and_flat_registry_are_disjoint() -> None:
    import autobio.tools  # noqa: F401 - importing populates both registries
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY

    overlap = set(CATALOG) & set(TOOL_REGISTRY)
    assert not overlap, f"Tools registered in both CATALOG and TOOL_REGISTRY: {sorted(overlap)}"
```

- [ ] **Step 2: Run the test to verify it passes on the current tree**

Run: `python -m pytest tests/unit/test_registry_disjoint.py -v`
Expected: PASS (freesasa/esm are catalog-only; all others are flat-only — currently disjoint).

- [ ] **Step 3: Run it alongside the registry-clearing suites to confirm no ordering flakiness**

Run: `python -m pytest tests/unit/test_catalog.py tests/unit/test_registry.py tests/unit/test_registry_disjoint.py -v`
Expected: PASS (the clearing fixtures snapshot/restore, so the module-level registries are intact when this test runs).

- [ ] **Step 4: Lint, format, type-check**

Run:
```bash
ruff check --fix tests/unit/test_registry_disjoint.py
ruff format tests/unit/test_registry_disjoint.py
mypy src/
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_registry_disjoint.py
git commit -m "test: guard CATALOG/TOOL_REGISTRY disjointness against half-migrations"
```

---

### Task 3: Standardize the `<TOOL>_TOOL` module-constant convention (esm)

**Why:** freesasa exposes `FREESASA_TOOL` (a module-level constant registered via `register(FREESASA_TOOL)`), which CLI-isolation tests re-register after CATALOG-clearing fixtures. esm instead inlines `register(Tool(...))`, so its Tool objects are not importable. Standardizing on the named-constant form now prevents the 11 upcoming migrations from diverging and makes esm's Tools re-registerable.

**Files:**
- Modify: `src/autobio/tools/esm.py` (extract `ESM1B_TOOL` / `ESM2_TOOL` constants)
- Test: `tests/unit/test_esm.py` (assert the constants exist and are the registered objects)

**Interfaces:**
- Produces: module-level `ESM1B_TOOL: Tool` and `ESM2_TOOL: Tool` in `autobio.tools.esm`, each registered so `get_tool("esm1b") is ESM1B_TOOL` and `get_tool("esm2") is ESM2_TOOL`.

- [ ] **Step 1: Write the failing constants test**

Append to `tests/unit/test_esm.py`:

```python
def test_esm_tool_constants_registered() -> None:
    import autobio.tools  # noqa: F401
    from autobio.tools.esm import ESM1B_TOOL, ESM2_TOOL

    assert ESM1B_TOOL.name == "esm1b"
    assert ESM2_TOOL.name == "esm2"
    assert get_tool("esm1b") is ESM1B_TOOL
    assert get_tool("esm2") is ESM2_TOOL
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/test_esm.py::test_esm_tool_constants_registered -v`
Expected: FAIL with `ImportError: cannot import name 'ESM1B_TOOL' from 'autobio.tools.esm'`.

- [ ] **Step 3: Extract the named constants**

In `src/autobio/tools/esm.py`, replace the two inline registration blocks (currently lines 191–248, the `register(Tool(name="esm1b", ...))` and `register(Tool(name="esm2", ...))` calls) with named constants + registration. The Tool field values are unchanged — only the binding-to-a-name and the `register(...)` argument change:

```python
ESM1B_TOOL = Tool(
    name="esm1b",
    display_name="ESM-1b",
    category=ToolCategory.EMBEDDING,
    description=(
        "Extract protein sequence embeddings using ESM-1b (650M params, 33 layers, 1280-dim)."
    ),
    version="1.0.0",
    image_tag="esm:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="embed",
    modes={
        "embed": Mode(
            name="embed",
            display_name="Embeddings",
            description="Extract per-sequence or per-residue embeddings.",
            input_schema=ESMEmbedInput,
            output_schema=EmbeddingOutput,
            default_timeout=600,
            supports_batch=True,
            notes=_ESM_NOTES,
        )
    },
    keywords=("esm", "embedding", "protein language model"),
)
"""The catalog Tool for ESM-1b — exposed for tests that re-register it after
CATALOG-clearing fixtures (e.g. CLI isolation tests)."""

register(ESM1B_TOOL)

ESM2_TOOL = Tool(
    name="esm2",
    display_name="ESM-2",
    category=ToolCategory.EMBEDDING,
    description=(
        "Extract protein sequence embeddings using ESM-2. Default checkpoint 650M "
        "(33 layers, 1280-dim); select 8M/35M/150M/3B/15B via the checkpoint field."
    ),
    version="1.0.0",
    image_tag="esm:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="embed",
    modes={
        "embed": Mode(
            name="embed",
            display_name="Embeddings",
            description="Extract per-sequence or per-residue embeddings.",
            input_schema=ESM2Input,
            output_schema=EmbeddingOutput,
            default_timeout=600,
            supports_batch=True,
            notes=_ESM2_NOTES,
        )
    },
    keywords=("esm", "esm2", "embedding", "protein language model"),
)
"""The catalog Tool for ESM-2 — exposed for tests that re-register it after
CATALOG-clearing fixtures (e.g. CLI isolation tests)."""

register(ESM2_TOOL)
```

- [ ] **Step 4: Run the constants test + full esm suite**

Run: `python -m pytest tests/unit/test_esm.py -v`
Expected: PASS (all, including `test_esm_registered_as_single_mode_tools` and `test_info_snapshot_esm2`).

- [ ] **Step 5: Lint, format, type-check**

Run:
```bash
ruff check --fix src/autobio/tools/esm.py tests/unit/test_esm.py
ruff format src/autobio/tools/esm.py tests/unit/test_esm.py
mypy src/
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autobio/tools/esm.py tests/unit/test_esm.py
git commit -m "esm: extract ESM1B_TOOL/ESM2_TOOL constants for the standard convention"
```

---

### Task 4: Compute `info` JSON schemas lazily (micro-opt)

**Why:** `format_tool_info_catalog` builds the `modes` list — including `model_json_schema()` for every mode's input and output schema — before the format branch, so TABLE output pays for schema generation it never renders. Moving the schema computation into the JSON branch is a pure internal refactor (identical output) that also lets a schema whose `model_json_schema()` is expensive/raising still render as a table.

**Files:**
- Modify: `src/autobio/cli/formatters.py` (restructure `format_tool_info_catalog`)
- Test: `tests/unit/test_formatters.py` (regression: TABLE must not touch mode schemas)

**Interfaces:**
- Consumes: `Tool`, `Mode`, `tool_categories` (already imported in `formatters.py`).
- Produces: no signature change; `format_tool_info_catalog(tool, fmt)` output is byte-identical to today for both JSON and TABLE.

- [ ] **Step 1: Write the failing regression test**

In `tests/unit/test_formatters.py`, add `import pytest` to the imports (after `import json`, line 5), then append at the end of the file:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/unit/test_formatters.py::test_format_tool_info_catalog_table_skips_schema_computation -v`
Expected: FAIL — the TABLE call raises `AssertionError` because schemas are computed before the branch.

- [ ] **Step 3: Restructure `format_tool_info_catalog`**

In `src/autobio/cli/formatters.py`, replace the entire body of `format_tool_info_catalog` (currently the block computing `modes = [...]` at lines 222–234 through the end of the function at line 273) with:

```python
    if fmt == OutputFormat.JSON:
        modes = [
            {
                "name": mode.name,
                "display_name": mode.display_name,
                "description": mode.description,
                "category": (mode.category or tool.category).value,
                "default_timeout": mode.default_timeout,
                "supports_batch": mode.supports_batch,
                "input_schema": mode.input_schema.model_json_schema(),
                "output_schema": mode.output_schema.model_json_schema(),
            }
            for mode in tool.modes.values()
        ]
        data = {
            "name": tool.name,
            "display_name": tool.display_name,
            "category": tool.category.value,
            "categories": [c.value for c in tool_categories(tool)],
            "version": tool.version,
            "image_tag": tool.image_tag,
            "requires_gpu": tool.requires_gpu,
            "gpu_count": tool.gpu_count,
            "description": tool.description,
            "keywords": list(tool.keywords),
            "default_mode": tool.default_mode,
            "modes": modes,
        }
        return json.dumps(data, indent=2)

    table = Table(title=f"Tool: {tool.name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Display Name", tool.display_name)
    table.add_row("Category", tool.category.value)
    table.add_row("Categories", ", ".join(c.value for c in tool_categories(tool)))
    table.add_row("Image", tool.image_tag)
    table.add_row("GPU Required", "yes" if tool.requires_gpu else "no")
    table.add_row("GPU Count", str(tool.gpu_count))
    table.add_row("Version", tool.version)
    table.add_row("Description", tool.description)
    if tool.keywords:
        table.add_row("Keywords", ", ".join(tool.keywords))
    table.add_row("Default Mode", tool.default_mode)
    for mode in tool.modes.values():
        category = (mode.category or tool.category).value
        table.add_row(
            f"Mode: {mode.name}",
            f"{mode.display_name} — {mode.description} "
            f"(category={category}, timeout={mode.default_timeout}s)",
        )
    return _render_table(table)
```

- [ ] **Step 4: Run the formatter suite + the info snapshot tests**

Run: `python -m pytest tests/unit/test_formatters.py tests/unit/test_freesasa.py::test_info_snapshot_freesasa tests/unit/test_esm.py::test_info_snapshot_esm2 -v`
Expected: PASS (JSON output unchanged; the byte-identical snapshot tests confirm no regression).

- [ ] **Step 5: Lint, format, type-check**

Run:
```bash
ruff check --fix src/autobio/cli/formatters.py tests/unit/test_formatters.py
ruff format src/autobio/cli/formatters.py tests/unit/test_formatters.py
mypy src/
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/autobio/cli/formatters.py tests/unit/test_formatters.py
git commit -m "cli: compute info JSON schemas only in the JSON branch (micro-opt)"
```

---

## Final verification (after all tasks)

- [ ] **Full non-docker/gpu suite + lint + type-check**

Run:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
python -m pytest -m "not docker and not gpu"
mypy src/
```
Expected: all green (baseline before this plan was 1317 passed).

---

## Self-Review

**1. Spec coverage** (against the "Carry-forward from Plan 2" list in `docs/superpowers/plans/2026-07-02-autobio-tools-modes-phase1.md`, §"Carry-forward from Plan 2"):
- (1) Harden the `extra` contract (fail-fast on typed-field shadow) → Task 1. ✓
- (2) Cross-registry disjointness guard → Task 2. ✓
- (3) Standardize `<TOOL>_TOOL` module-constant convention → Task 3 (esm). ✓
- (7) Micro-opt: compute `model_json_schema()` only in JSON branch → Task 4. ✓
- Deferred (correctly out of this hardening plan, applied *during* family migrations): (4) add `mode: str | None` to `RunMetadata` — do it when the first genuinely multi-mode engine (rosetta/openmm) migrates; (5) per-family serialization/coercion check — a per-family checklist item; (6) README row reconciliation — batched in the teardown/README plan.

**2. Placeholder scan:** No "TBD"/"similar to Task N"/"handle edge cases". Every code step shows complete code; every test step shows complete tests; commands use the `python -m pytest` form and expected outcomes.

**3. Type consistency:** `_apply_extra(config: dict[str, Any], input_data: BaseInput) -> None` is defined in Task 1 and called with `dict[str, Any]` (freesasa) and `dict[str, object]` (esm) — both compatible with the `dict[str, Any]` parameter under mypy. `ESM1B_TOOL`/`ESM2_TOOL` names in Task 3 match the test in the same task. `_ExplodingInput.model_json_schema` override in Task 4 matches the call site `mode.input_schema.model_json_schema()`. The reused test helpers (`_register_faketool`, `_make_runner`, `_written_config`, `_pdb`, `_OutInfo`) already exist in their target files.

---

## Next plans (not in this hardening plan)

This plan is the recipe-hardening prerequisite the Plan 2 review said to apply **before** the bulk family migrations. It renumbers the provisional plan sequence from the Phase 1 doc: the family migrations become **Plan 4**, teardown becomes **Plan 5**.

- **Plan 4 — remaining family migrations (Phase 1 cont.).** Migrate the rest against the now-hardened recipe, grouped by shape (suggested PR-per-group, simple → complex):
  - **Singletons** (single-mode Tools, esm-shaped): esmfold, chai1, boltz1, boltz2, proteinmpnn, rfd3, antipasti, baddg, stabddg, prodigy, openfold3.
  - **Same-category multi-mode:** rosetta (per-mode `image_tag`; first engine to add `RunMetadata.mode`, carry-forward #4), evoef2, complexa.
  - **Cross-category + two-class consolidation:** esm_if1 (`{design, score}`), antifold (`{design, score}`), ligandmpnn.
  - **Output-schema variance:** openmm (per-mode image + `SimulationOutput`), antibody LMs ×6 (antibody `SequenceSet`, output variance, shared runner).
  - Per family, apply carry-forward #5 (verify promoted-field coercions keep `config.json` byte-compatible) and #6 (collapse migrated flat README rows into one Tool row).
- **Plan 5 — teardown + contract polish.** Once all families are on the catalog: remove `TOOL_REGISTRY`/`ToolEntry`/legacy CLI paths, collapse `get_runner`/CLI to catalog-only, finalize the unified §7 `list` JSON schema, decide the `CATALOG`↔`TOOL_REGISTRY` naming/home, and rewrite the README to Tools/Modes.
