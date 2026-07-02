# Tools→Modes Refactor — Plan 2: Phase 1 (Prove-the-Pattern) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the transition scaffolding that lets catalog **Tools** (with **Modes**) and legacy flat tools coexist, then migrate two self-contained families (**freesasa**, **esm**) onto the catalog end-to-end — proving the migration recipe before mass migration.

**Architecture:** Two registries coexist during Phase 1 — the legacy `TOOL_REGISTRY` (flat name → `ToolEntry`) and the new `CATALOG` (Tool name → `Tool`). A tool is "migrated" iff its name is in `CATALOG`. The base `ToolRunner` gains a dual code path: migrated tools resolve a `Mode` (image/timeout/schema per mode) and expose it as `self.current_mode`; legacy tools keep using `self.entry` unchanged. The CLI (`list`/`info`/`run`) gains a catalog path alongside the legacy path. Migrating a family means declaring its `Tool` + `Mode`s, splitting its input schema from a shared base with typed fields + `x-autobio` hints, rewiring its runner to dispatch on `self.current_mode`, and swapping its registry/runner-map entries. Container-side `config.json` is byte-for-byte unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2, dataclasses, Typer CLI, `pytest`. No new dependencies.

## Global Constraints

- Python 3.11+; `from __future__ import annotations` at the top of every module; modern syntax (`X | Y`, `Literal`, `StrEnum`).
- Type hints on all signatures; Google-style docstrings on all public classes/functions.
- Max line length 100. Formatter `ruff format`; linter `ruff check`; both must be clean. `mypy src/` must be clean.
- Absolute imports only; no wildcard imports; no business logic in `__init__.py`.
- **Test invocation:** use `python -m pytest ...` everywhere. A bare `pytest` on this machine resolves to a different environment that cannot import `autobio` and fails with `ModuleNotFoundError`. `ruff` and `mypy` are fine invoked bare.
- **Container-side is a non-goal.** For each migrated `(tool, mode)`, the runner MUST write the same `config.json` keys/values the container expects today (freesasa still writes `"mode": "bsa"|"sasa"`; esm still writes a resolved `"model_name"`). Tests must assert `config.json` is unchanged from today for equivalent inputs. No changes to `containers/`, the workspace protocol, `result.json`, or GPU allocation.
- **Clean break for migrated tools:** the flat names `freesasa_bsa`/`freesasa_sasa` are REMOVED (replaced by tool `freesasa` + `--mode sasa|bsa`). `esm1b`/`esm2` keep their names but become single-mode Tools. No back-compat shims for removed flat names.
- **`extra` double-write rule:** every `extra` key promoted to a typed field must be dropped from the runner's `extra` flat-merge (via `_CONSUMED_EXTRA_KEYS`, or because it is no longer an `extra` key at all). Promoted keys must never be written to `config.json` twice.
- **Do NOT** migrate any family other than freesasa and esm, and do NOT remove `TOOL_REGISTRY`/`ToolEntry` or the legacy CLI paths — other families remain flat until a later plan.
- Carry-forward decisions from Plan 1 applied here: (a) the sequence `x-autobio` hint is attached at the FIELD level in the migrated schema (not bundled into the `SequenceSet` alias); (b) the registry keeps the name `CATALOG` in `core/catalog.py` (rename/teardown is a later plan).

---

### Task 1: Add per-mode `image_tag` override to `Mode`

Some multi-mode engines migrated later (rosetta, openmm) use a **different container image per mode**. The `Tool`/`Mode` model must express that. Add an optional `image_tag` override on `Mode`; the base runner (Task 2) resolves `mode.image_tag or tool.image_tag`. freesasa/esm leave it `None`.

**Files:**
- Modify: `src/autobio/core/catalog.py`
- Test: `tests/unit/test_catalog.py`

**Interfaces:**
- Produces: `Mode` gains field `image_tag: str | None = None` (keyword-defaulted; existing keyword constructions unaffected).

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_catalog.py (module-top imports already cover Mode)

def test_mode_image_tag_defaults_to_none() -> None:
    assert _mode("embed").image_tag is None


def test_mode_image_tag_override_is_stored() -> None:
    m = Mode(
        name="relax",
        display_name="Relax",
        description="relax",
        input_schema=BaseInput,
        output_schema=BaseOutput,
        default_timeout=600,
        image_tag="rosetta-relax:1.0.0",
    )
    assert m.image_tag == "rosetta-relax:1.0.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_catalog.py -k image_tag -v`
Expected: FAIL — `TypeError: Mode.__init__() got an unexpected keyword argument 'image_tag'` (and the default test fails on the missing attribute).

- [ ] **Step 3: Write minimal implementation**

In `src/autobio/core/catalog.py`, add the field to `Mode` (immediately after `supports_batch`):

```python
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
    image_tag: str | None = None
    category: ToolCategory | None = None
    notes: tuple[str, ...] = ()
```

Update the `Mode` docstring line to note the override:

```python
    """A named use (task/operation) of a Tool.

    ``image_tag`` overrides the owning Tool's image for this mode (used by
    engines whose modes ship as separate container images); ``None`` falls
    back to ``Tool.image_tag``.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_catalog.py -v`
Expected: PASS (all catalog tests, including the two new ones).

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/core/catalog.py tests/unit/test_catalog.py
ruff format src/autobio/core/catalog.py tests/unit/test_catalog.py
git add src/autobio/core/catalog.py tests/unit/test_catalog.py
git commit -m "catalog: add optional per-mode image_tag override to Mode"
```

---

### Task 2: Base `ToolRunner` dual path (catalog + legacy) with `run(mode=...)`

Make `ToolRunner` resolve either a catalog `Tool` (migrated) or a legacy `ToolEntry`. Add a `mode` parameter to `run()`, expose the selected `Mode` as `self.current_mode` before `prepare_workspace`, and resolve image/timeout/gpu/version from whichever source is active. `prepare_workspace`/`parse_output` signatures are UNCHANGED (migrated runners read `self.current_mode`; legacy runners ignore it) — this is what keeps unmigrated runners working.

**Files:**
- Modify: `src/autobio/tools/base.py`
- Test: `tests/unit/test_tool_runner_modes.py` (new)

**Interfaces:**
- Consumes: `autobio.core.catalog.{CATALOG, Mode, Tool, get_tool}` (Task 1); `autobio.core.registry.{TOOL_REGISTRY, ToolEntry}`; `autobio.core.result.AutobioError`.
- Produces:
  - `ToolRunner.__init__(self, tool_name, config)` sets exactly one of `self.tool: Tool | None` / `self.entry: ToolEntry | None`; raises `KeyError` for an unknown name (listing the union of catalog + registry names). Also sets `self.current_mode: Mode | None = None`.
  - `ToolRunner.run(self, input_data, gpu="auto", timeout=None, output_dir=None, mode=None)` — `mode` selects a catalog Mode by name (defaults to `tool.default_mode`); raises `AutobioError` for an unknown mode or when `mode` is passed to a legacy tool.
  - Helpers: `_resolve_mode`, `_image_tag`, `_default_timeout`, `_requires_gpu`, `_gpu_count`, `_tool_version`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tool_runner_modes.py
"""Tests for catalog (Mode-aware) dispatch in ToolRunner, alongside the legacy path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autobio.core.catalog import CATALOG, Mode, Tool, register
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput, BaseOutput
from autobio.tools.base import ToolRunner


@pytest.fixture(autouse=True)
def _clean_catalog():
    snapshot = dict(CATALOG)
    CATALOG.clear()
    yield
    CATALOG.clear()
    CATALOG.update(snapshot)


class _Input(BaseInput):
    pass


class _Output(BaseOutput):
    pass


class _CaptureRunner(ToolRunner):
    """Records the mode active during prepare_workspace; minimal parse_output."""

    captured_mode: str | None = None

    def prepare_workspace(self, input_data, workspace) -> None:
        self.captured_mode = self.current_mode.name if self.current_mode else None

    def parse_output(self, workspace) -> _Output:
        return _Output(
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )


def _register_faketool() -> None:
    register(
        Tool(
            name="faketool",
            display_name="Fake",
            category=ToolCategory.SCORING,
            description="fake",
            version="9.9.9",
            image_tag="fake:1.0.0",
            requires_gpu=False,
            gpu_count=0,
            default_mode="alpha",
            modes={
                "alpha": Mode("alpha", "Alpha", "a", _Input, _Output, default_timeout=111),
                "beta": Mode(
                    "beta", "Beta", "b", _Input, _Output,
                    default_timeout=222, image_tag="fake-beta:1.0.0",
                ),
            },
        )
    )


def _make_runner(tool_name: str) -> _CaptureRunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        return _CaptureRunner(tool_name, AutobioConfig.resolve())


def test_init_resolves_catalog_tool() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    assert runner.tool is not None
    assert runner.entry is None
    assert runner.current_mode is None


def test_init_unknown_name_lists_available() -> None:
    _register_faketool()
    with pytest.raises(KeyError, match="faketool"):
        _make_runner("does-not-exist")


def test_resolve_mode_default_and_explicit() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    assert runner._resolve_mode(None).name == "alpha"
    assert runner._resolve_mode("beta").name == "beta"


def test_resolve_mode_unknown_raises() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    with pytest.raises(AutobioError, match="Unknown mode"):
        runner._resolve_mode("gamma")


def test_image_and_timeout_use_mode_override() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner.current_mode = runner._resolve_mode("beta")
    assert runner._image_tag() == "fake-beta:1.0.0"
    assert runner._default_timeout() == 222
    runner.current_mode = runner._resolve_mode("alpha")
    assert runner._image_tag() == "fake:1.0.0"  # falls back to Tool.image_tag
    assert runner._default_timeout() == 111


def test_run_sets_current_mode_and_mode_metadata(tmp_path, monkeypatch) -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner._gpu.allocate.return_value = []
    monkeypatch.setattr(
        "autobio.core.workspace.Workspace.read_result",
        lambda self: SimpleNamespace(status="success", phase="run", exit_code=0, error_message=None),
    )
    out = runner.run(_Input(), gpu="none", output_dir=tmp_path, mode="beta")
    assert runner.captured_mode == "beta"
    assert out.metadata.tool_version == "9.9.9"
    assert out.metadata.image_uri.endswith("fake-beta:1.0.0")


def test_run_rejects_mode_for_legacy_tool() -> None:
    # 'prodigy' is a legacy flat tool (in TOOL_REGISTRY, not CATALOG) — imported for real.
    import autobio.tools  # noqa: F401 - populate registries
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = _CaptureRunner("prodigy", AutobioConfig.resolve())
    assert runner.entry is not None
    assert runner.tool is None
    with pytest.raises(AutobioError, match="does not support modes"):
        runner.run(_Input(), gpu="none", mode="whatever")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -v`
Expected: FAIL — `run()` has no `mode` parameter / `_resolve_mode` undefined / `self.tool` unset.

- [ ] **Step 3: Write minimal implementation**

Replace the imports and the relevant methods in `src/autobio/tools/base.py`. Update the imports block:

```python
from autobio.core.catalog import CATALOG, Mode, Tool, get_tool
from autobio.core.container import ContainerManager
from autobio.core.gpu import GPUManager
from autobio.core.registry import TOOL_REGISTRY, ToolEntry
from autobio.core.result import AutobioError, ToolExecutionError
from autobio.core.workspace import Workspace
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata
from autobio.utils.logging import get_logger
```

Replace `__init__`:

```python
    def __init__(self, tool_name: str, config: AutobioConfig) -> None:
        self.tool: Tool | None = None
        self.entry: ToolEntry | None = None
        if tool_name in CATALOG:
            self.tool = get_tool(tool_name)
        elif tool_name in TOOL_REGISTRY:
            self.entry = TOOL_REGISTRY[tool_name]
        else:
            available = ", ".join(sorted(set(CATALOG) | set(TOOL_REGISTRY))) or "(none)"
            raise KeyError(f"Unknown tool {tool_name!r}. Available tools: {available}") from None
        self.tool_name = tool_name
        self.config = config
        self.current_mode: Mode | None = None
        self._container = ContainerManager(config)
        self._gpu = GPUManager()
```

Replace `run` (keep the docstring; add the `mode` param and swap `self.entry.*` for the helpers):

```python
    def run(
        self,
        input_data: BaseInput,
        gpu: str | list[int] = "auto",
        timeout: int | None = None,
        output_dir: Path | None = None,
        mode: str | None = None,
    ) -> BaseOutput:
        """Full execution lifecycle. Do NOT override this method.

        For migrated (catalog) tools, *mode* selects a :class:`Mode` by name
        (defaulting to the Tool's ``default_mode``). Legacy tools ignore *mode*
        (passing one raises).

        Raises:
            ToolExecutionError: If the container reports a failure.
            AutobioError: For an unknown mode, or a mode passed to a legacy tool.
        """
        gpu_ids: list[int] = []
        workspace: Workspace | None = None
        start = time.monotonic()

        self.current_mode = self._resolve_mode(mode)

        try:
            workspace = Workspace.create(output_dir)
            self.prepare_workspace(input_data, workspace)
            gpu_ids = self._resolve_gpu(gpu)

            image_uri = f"{self.config.image_prefix}{self._image_tag()}"
            self._container.ensure_image(image_uri)

            effective_timeout = timeout if timeout is not None else self._default_timeout()
            self._container.run(
                image_uri=image_uri,
                workspace=workspace.root,
                gpu_ids=gpu_ids or None,
                timeout=effective_timeout,
            )

            wall_time = time.monotonic() - start
            run_result = workspace.read_result()

            if run_result.status != "success":
                logs = self._read_logs(workspace)
                raise ToolExecutionError(
                    phase=run_result.phase,
                    exit_code=run_result.exit_code,
                    error_message=run_result.error_message or "Unknown error",
                    logs=logs,
                    wall_time=wall_time,
                )

            output = self.parse_output(workspace)
            output.metadata = self._build_metadata(workspace, wall_time, gpu_ids, image_uri)
            return output

        finally:
            if gpu_ids:
                self._gpu.release(gpu_ids)
            if workspace is not None and workspace._is_temp:
                workspace.cleanup()
```

Add the helper methods (place them just after `run`):

```python
    def _resolve_mode(self, mode: str | None) -> Mode | None:
        """Resolve the selected Mode for a migrated tool, or None for a legacy tool."""
        if self.tool is None:
            if mode is not None:
                raise AutobioError(f"Tool {self.tool_name!r} does not support modes.")
            return None
        name = mode if mode is not None else self.tool.default_mode
        try:
            return self.tool.modes[name]
        except KeyError:
            available = ", ".join(sorted(self.tool.modes))
            raise AutobioError(
                f"Unknown mode {name!r} for tool {self.tool_name!r}. Available modes: {available}"
            ) from None

    def _image_tag(self) -> str:
        """Container image tag for the current run (mode override, else tool/entry)."""
        if self.tool is not None:
            assert self.current_mode is not None
            return self.current_mode.image_tag or self.tool.image_tag
        assert self.entry is not None
        return self.entry.image_tag

    def _default_timeout(self) -> int:
        """Default timeout for the current run (per-mode for catalog tools)."""
        if self.current_mode is not None:
            return self.current_mode.default_timeout
        assert self.entry is not None
        return self.entry.default_timeout

    def _requires_gpu(self) -> bool:
        """Whether the active tool requires a GPU."""
        if self.tool is not None:
            return self.tool.requires_gpu
        assert self.entry is not None
        return self.entry.requires_gpu

    def _gpu_count(self) -> int:
        """Number of GPUs the active tool requests under ``gpu='auto'``."""
        if self.tool is not None:
            return self.tool.gpu_count
        assert self.entry is not None
        return self.entry.gpu_count

    def _tool_version(self) -> str:
        """Version string of the active tool."""
        if self.tool is not None:
            return self.tool.version
        assert self.entry is not None
        return self.entry.version
```

In `_resolve_gpu`, replace the two `self.entry.*` reads:

```python
        if gpu == "auto":
            if not self._requires_gpu():
                return []
            return self._gpu.allocate(count=self._gpu_count())
```

In `_build_metadata`, replace `tool_version=self.entry.version` with `tool_version=self._tool_version()`.

- [ ] **Step 4: Run test to verify it passes, and confirm the legacy lifecycle is intact**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py tests/unit/test_tool_runner.py -v`
Expected: PASS — new mode tests pass AND the existing full-lifecycle legacy tests in `test_tool_runner.py` still pass (legacy path unchanged).

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py
ruff format src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py
git add src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py
git commit -m "tools: dual-path ToolRunner (catalog Modes + legacy) with run(mode=...)"
```

> **Note on `get_runner` (`tools/__init__.py`):** no code change is needed. `get_runner(name)` already looks up `TOOL_RUNNERS[name]` and returns `runner_cls(name, config)`; the base `__init__` now routes by name into `CATALOG` or `TOOL_REGISTRY`. Migrating a family just changes which keys live in `TOOL_RUNNERS`. This is verified by tests in Task 6 (freesasa).

---

### Task 3: CLI `info` — catalog path

`autobio info <migrated-tool>` must emit the modes payload (each mode's resolved `input_schema` with `x-autobio` hints + `output_schema`); legacy tools keep the existing payload.

**Files:**
- Modify: `src/autobio/cli/formatters.py`, `src/autobio/cli/info.py`
- Test: `tests/unit/test_formatters.py`

**Interfaces:**
- Consumes: `autobio.core.catalog.{CATALOG, get_tool, tool_categories, Tool}`.
- Produces: `format_tool_info_catalog(tool: Tool, fmt=OutputFormat.TABLE) -> str`. `info_cmd` routes catalog tools to it.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_formatters.py

import json as _json

from autobio.cli.formatters import OutputFormat, format_tool_info_catalog
from autobio.core.catalog import Mode, Tool
from autobio.core.registry import ToolCategory
from autobio.schemas.base import BaseInput, BaseOutput


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
            "a": Mode("a", "Alpha", "alpha mode", _InInfo, _OutInfo, default_timeout=300),
            "b": Mode(
                "b", "Beta", "beta mode", _InInfo, _OutInfo,
                default_timeout=600, category=ToolCategory.SIMULATION,
            ),
        },
        keywords=("demo", "example"),
    )


def test_format_tool_info_catalog_json_shape() -> None:
    parsed = _json.loads(format_tool_info_catalog(_tool_for_info(), OutputFormat.JSON))
    assert parsed["name"] == "demo"
    assert parsed["default_mode"] == "a"
    assert parsed["categories"] == ["scoring", "simulation"]  # union, primary first
    assert parsed["keywords"] == ["demo", "example"]
    mode_names = [m["name"] for m in parsed["modes"]]
    assert mode_names == ["a", "b"]
    mode_a = parsed["modes"][0]
    assert mode_a["category"] == "scoring"          # falls back to Tool category
    assert "input_schema" in mode_a and "output_schema" in mode_a
    assert parsed["modes"][1]["category"] == "simulation"  # mode override


def test_format_tool_info_catalog_table_runs() -> None:
    out = format_tool_info_catalog(_tool_for_info(), OutputFormat.TABLE)
    assert "demo" in out and "Alpha" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_formatters.py -k catalog -v`
Expected: FAIL — `ImportError: cannot import name 'format_tool_info_catalog'`.

- [ ] **Step 3: Write minimal implementation**

In `src/autobio/cli/formatters.py`, add a runtime import near the top (below the existing imports):

```python
from autobio.core.catalog import tool_categories
```

Add the `Tool` type to the `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from autobio.core.catalog import Tool
    from autobio.core.container import ImageInfo
    from autobio.core.registry import ToolEntry
    from autobio.core.result import RunResult
```

Add the function (after `format_tool_info`):

```python
def format_tool_info_catalog(tool: Tool, fmt: OutputFormat = OutputFormat.TABLE) -> str:
    """Format detailed info for a catalog Tool (with its Modes).

    Args:
        tool: The catalog :class:`~autobio.core.catalog.Tool`.
        fmt: Output format.

    Returns:
        Formatted string.
    """
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

    if fmt == OutputFormat.JSON:
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
    for mode in modes:
        table.add_row(
            f"Mode: {mode['name']}",
            f"{mode['display_name']} — {mode['description']} "
            f"(category={mode['category']}, timeout={mode['default_timeout']}s)",
        )
    return _render_table(table)
```

In `src/autobio/cli/info.py`, route catalog tools:

```python
"""`autobio info` — show details for a single tool."""

from __future__ import annotations

from typing import Annotated

import typer

from autobio.cli.formatters import (
    OutputFormat,
    format_tool_info,
    format_tool_info_catalog,
    print_error,
)
from autobio.core.catalog import CATALOG
from autobio.core.catalog import get_tool as get_catalog_tool
from autobio.core.registry import TOOL_REGISTRY
from autobio.core.registry import get_tool as get_registry_tool


def info_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Show detailed information about a tool."""
    if tool in CATALOG:
        typer.echo(format_tool_info_catalog(get_catalog_tool(tool), fmt))
        return
    if tool in TOOL_REGISTRY:
        typer.echo(format_tool_info(tool, get_registry_tool(tool), fmt))
        return
    available = ", ".join(sorted(set(CATALOG) | set(TOOL_REGISTRY))) or "(none)"
    print_error(f"Unknown tool {tool!r}. Available tools: {available}")
    raise typer.Exit(code=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_formatters.py -k catalog -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/cli/formatters.py src/autobio/cli/info.py tests/unit/test_formatters.py
ruff format src/autobio/cli/formatters.py src/autobio/cli/info.py tests/unit/test_formatters.py
git add src/autobio/cli/formatters.py src/autobio/cli/info.py tests/unit/test_formatters.py
git commit -m "cli: info renders catalog Tools with modes + output_schema"
```

---

### Task 4: CLI `list` — merge legacy + catalog tools

`autobio list` must show both remaining flat tools and migrated catalog Tools (each once).

**Files:**
- Modify: `src/autobio/cli/formatters.py`, `src/autobio/cli/list.py`
- Test: `tests/unit/test_formatters.py`

**Interfaces:**
- Consumes: `autobio.core.catalog.{Tool, tool_categories}`, `autobio.core.registry.ToolEntry`.
- Produces: `format_tool_list_merged(flat: dict[str, ToolEntry], tools: dict[str, Tool], fmt=OutputFormat.TABLE) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_formatters.py

from autobio.cli.formatters import format_tool_list_merged
from autobio.core.registry import ToolEntry


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
    rows = _json.loads(
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_formatters.py -k merged -v`
Expected: FAIL — `ImportError: cannot import name 'format_tool_list_merged'`.

- [ ] **Step 3: Write minimal implementation**

In `src/autobio/cli/formatters.py`, add (after `format_tool_list`):

```python
def format_tool_list_merged(
    flat: dict[str, ToolEntry],
    tools: dict[str, Tool],
    fmt: OutputFormat = OutputFormat.TABLE,
) -> str:
    """Format legacy flat tools and catalog Tools together, sorted by name.

    Args:
        flat: Legacy tool name → :class:`ToolEntry` (not yet migrated).
        tools: Catalog tool name → :class:`~autobio.core.catalog.Tool`.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        rows: list[dict[str, object]] = []
        for name, entry in flat.items():
            rows.append(
                {
                    "name": name,
                    "category": entry.category.value,
                    "gpu": entry.requires_gpu,
                    "version": entry.version,
                    "description": entry.description,
                }
            )
        for name, tool in tools.items():
            rows.append(
                {
                    "name": name,
                    "display_name": tool.display_name,
                    "category": tool.category.value,
                    "categories": [c.value for c in tool_categories(tool)],
                    "gpu": tool.requires_gpu,
                    "version": tool.version,
                    "description": tool.description,
                    "modes": list(tool.modes),
                    "keywords": list(tool.keywords),
                }
            )
        rows.sort(key=lambda r: r["name"])
        return json.dumps(rows, indent=2)

    if not flat and not tools:
        return "No tools registered."

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("GPU")
    table.add_column("Version")
    table.add_column("Modes")
    table.add_column("Description")

    combined: list[tuple[str, str, bool, str, str, str]] = []
    for name, entry in flat.items():
        combined.append(
            (name, entry.category.value, entry.requires_gpu, entry.version, "-", entry.description)
        )
    for name, tool in tools.items():
        combined.append(
            (
                name,
                tool.category.value,
                tool.requires_gpu,
                tool.version,
                ", ".join(tool.modes),
                tool.description,
            )
        )
    for name, category, gpu, version, modes, description in sorted(combined, key=lambda r: r[0]):
        table.add_row(name, category, "yes" if gpu else "no", version, modes, description)

    return _render_table(table)
```

In `src/autobio/cli/list.py`:

```python
"""`autobio list` — display registered tools."""

from __future__ import annotations

from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_tool_list_merged
from autobio.core.catalog import list_tools as list_catalog_tools
from autobio.core.registry import ToolCategory
from autobio.core.registry import list_tools as list_registry_tools


def list_tools_cmd(
    category: Annotated[
        ToolCategory | None,
        typer.Option("--category", "-c", help="Filter by tool category."),
    ] = None,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """List available tools."""
    flat = list_registry_tools(category=category)
    tools = list_catalog_tools(category=category)
    typer.echo(format_tool_list_merged(flat, tools, fmt))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_formatters.py -k merged -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/cli/formatters.py src/autobio/cli/list.py tests/unit/test_formatters.py
ruff format src/autobio/cli/formatters.py src/autobio/cli/list.py tests/unit/test_formatters.py
git add src/autobio/cli/formatters.py src/autobio/cli/list.py tests/unit/test_formatters.py
git commit -m "cli: list merges legacy and catalog tools"
```

---

### Task 5: CLI `run --mode` — catalog path

`autobio run <tool> --mode <mode> --config <json>` validates against the selected mode's schema and forwards `mode` to `runner.run`. Legacy tools keep the current path (and reject `--mode`).

**Files:**
- Modify: `src/autobio/cli/run.py`
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `autobio.core.catalog.{CATALOG, get_tool}`, `get_runner`, `AutobioError`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_cli.py
# (test_cli.py already imports CliRunner as `runner`/`CliRunner` and the Typer `app`;
#  reuse those. Adjust the import alias to match the file's existing convention.)

import json as _json
from unittest.mock import MagicMock, patch

from autobio.core.catalog import CATALOG, Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata


class _RunInput(BaseInput):
    pass


class _RunOutput(BaseOutput):
    pass


def _register_run_tool() -> None:
    if "runtool" in CATALOG:
        return
    register(
        Tool(
            name="runtool",
            display_name="RunTool",
            category=ToolCategory.SCORING,
            description="run demo",
            version="1.0.0",
            image_tag="runtool:1.0.0",
            requires_gpu=False,
            gpu_count=0,
            default_mode="a",
            modes={
                "a": Mode("a", "A", "a", _RunInput, _RunOutput, default_timeout=1),
                "b": Mode("b", "B", "b", _RunInput, _RunOutput, default_timeout=1),
            },
        )
    )


def test_run_forwards_mode_for_catalog_tool(tmp_path) -> None:
    from datetime import UTC, datetime

    _register_run_tool()
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")

    mock_runner = MagicMock()
    mock_output = _RunOutput(
        metadata=RunMetadata(
            tool_name="runtool", tool_version="1.0.0", image_uri="runtool:1.0.0",
            wall_time_seconds=0.1, gpu_ids=None, workspace_path=tmp_path,
            timestamp=datetime.now(tz=UTC),
        ),
        raw_output_path=tmp_path,
    )
    mock_runner.run.return_value = mock_output

    with patch("autobio.cli.run.get_runner", return_value=mock_runner):
        result = CliRunner().invoke(
            app, ["run", "runtool", "--mode", "b", "--config", str(cfg), "--gpu", "none"]
        )

    assert result.exit_code == 0, result.output
    assert mock_runner.run.call_args.kwargs["mode"] == "b"


def test_run_rejects_mode_for_legacy_tool(tmp_path) -> None:
    import autobio.tools  # noqa: F401 - populate registries
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    result = CliRunner().invoke(
        app, ["run", "prodigy", "--mode", "x", "--config", str(cfg)]
    )
    assert result.exit_code == 1
    assert "does not support --mode" in result.output
```

> If `test_cli.py` already imports `CliRunner`/`app` at module top, drop the duplicate imports above and reuse them; keep the two test functions.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_cli.py -k "mode" -v`
Expected: FAIL — `run` has no `--mode` option, so `mode` is not forwarded / the legacy rejection message is absent.

- [ ] **Step 3: Write minimal implementation**

Replace `src/autobio/cli/run.py`:

```python
"""`autobio run` — execute a tool inside its container."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer evaluates annotations at runtime
from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_run_result, print_error
from autobio.core.catalog import CATALOG
from autobio.core.catalog import get_tool as get_catalog_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools import get_runner


def run_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to input config JSON file."),
    ],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Mode for a multi-mode tool (defaults to the tool's default)."),
    ] = None,
    gpu: Annotated[
        str,
        typer.Option("--gpu", help="GPU spec: 'auto', 'none', or comma-separated IDs."),
    ] = "auto",
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Maximum wall-clock seconds."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Persist workspace to this directory."),
    ] = None,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Run a tool with the given configuration."""
    try:
        config_data = json.loads(config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print_error(f"Failed to read config file: {exc}")
        raise typer.Exit(code=1) from None

    autobio_config = AutobioConfig.resolve()
    try:
        runner = get_runner(tool, autobio_config)
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    # Choose the input schema (per-mode for catalog tools) and the mode to forward.
    forward_mode: str | None = None
    if tool in CATALOG:
        catalog_tool = get_catalog_tool(tool)
        mode_name = mode if mode is not None else catalog_tool.default_mode
        if mode_name not in catalog_tool.modes:
            available = ", ".join(sorted(catalog_tool.modes))
            print_error(
                f"Unknown mode {mode_name!r} for tool {tool!r}. Available modes: {available}"
            )
            raise typer.Exit(code=1) from None
        input_schema = catalog_tool.modes[mode_name].input_schema
        forward_mode = mode_name
    else:
        if mode is not None:
            print_error(f"Tool {tool!r} does not support --mode.")
            raise typer.Exit(code=1) from None
        input_schema = runner.entry.input_schema

    try:
        input_data = input_schema.model_validate(config_data)
    except Exception as exc:
        print_error(f"Invalid input: {exc}")
        raise typer.Exit(code=1) from None

    try:
        output = runner.run(
            input_data, gpu=gpu, timeout=timeout, output_dir=output_dir, mode=forward_mode
        )
    except AutobioError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    output_dict = output.model_dump(mode="json")
    typer.echo(format_run_result(output_dict, fmt))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_cli.py -k "mode" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, format, commit**

```bash
ruff check --fix src/autobio/cli/run.py tests/unit/test_cli.py
ruff format src/autobio/cli/run.py tests/unit/test_cli.py
git add src/autobio/cli/run.py tests/unit/test_cli.py
git commit -m "cli: run accepts --mode and validates against the mode's schema"
```

---

### Task 6: Migrate the **freesasa** family (multi-mode exemplar)

Convert `freesasa_bsa`/`freesasa_sasa` into one Tool `freesasa` with modes `sasa` (default) + `bsa`. Promote `algorithm`/`probe_radius`/`per_residue` (and BSA's `partner1`/`partner2`) from `extra` to typed fields with `x-autobio` hints; dispatch the runner on `self.current_mode.name`; keep the container `config.json` identical.

**Files:**
- Modify: `src/autobio/schemas/scoring.py` (add freesasa input schemas), `src/autobio/tools/freesasa.py`, `src/autobio/tools/__init__.py`
- Test: `tests/unit/test_freesasa.py` (rewrite for `(tool, mode)`)

**Interfaces:**
- Consumes: `GenericSequenceSet` not needed here; `autobio.schemas.hints.{Tier, Widget, ui}`; `autobio.core.catalog.{Mode, Tool, register, get_tool}`.
- Produces: schemas `FreeSASABaseInput`, `FreeSASASASAInput`, `FreeSASABSAInput`; catalog Tool `freesasa` with modes `sasa`/`bsa`; `TOOL_RUNNERS["freesasa"] = FreeSASARunner` (flat entries removed).

- [ ] **Step 1: Write the failing test**

```python
# Rewrite tests/unit/test_freesasa.py
"""Unit tests for the migrated freesasa Tool (modes: sasa, bsa)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools import get_runner
from autobio.tools.freesasa import FreeSASARunner


@pytest.fixture
def _pdb(tmp_path: Path) -> Path:
    p = tmp_path / "complex.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    return p


def _make_runner(mode_name: str) -> FreeSASARunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = FreeSASARunner("freesasa", AutobioConfig.resolve())
    runner.current_mode = get_tool("freesasa").modes[mode_name]
    return runner


def _written_config(runner: FreeSASARunner, input_data, tmp_path: Path) -> dict:
    from autobio.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


def test_freesasa_registered_as_tool_not_flat() -> None:
    import autobio.tools  # noqa: F401
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY
    from autobio.tools import TOOL_RUNNERS

    assert "freesasa" in CATALOG
    assert set(get_tool("freesasa").modes) == {"sasa", "bsa"}
    assert get_tool("freesasa").default_mode == "sasa"
    assert "freesasa" in TOOL_RUNNERS
    assert "freesasa_bsa" not in TOOL_RUNNERS and "freesasa_sasa" not in TOOL_RUNNERS
    assert "freesasa_bsa" not in TOOL_REGISTRY and "freesasa_sasa" not in TOOL_REGISTRY


def test_get_runner_freesasa_resolves_catalog_tool() -> None:
    import autobio.tools  # noqa: F401
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = get_runner("freesasa", AutobioConfig.resolve())
    assert runner.tool is not None and runner.tool.name == "freesasa"


def test_get_runner_removed_flat_name_raises() -> None:
    import autobio.tools  # noqa: F401
    with pytest.raises(KeyError, match="freesasa_bsa"):
        get_runner("freesasa_bsa", AutobioConfig.resolve())


def test_sasa_config_unchanged(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    cfg = _written_config(runner, FreeSASASASAInput(structure_path=_pdb), tmp_path)
    assert cfg["mode"] == "sasa"
    assert cfg["structure_path"] == f"/workspace/inputs/{_pdb.name}"
    assert cfg["algorithm"] == "LeeRichards"
    assert cfg["probe_radius"] == 1.4
    assert cfg["per_residue"] is False
    assert cfg["output_dir"] == "/workspace/outputs/raw"
    assert "partner1" not in cfg and "partner2" not in cfg


def test_bsa_config_unchanged(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASABSAInput

    runner = _make_runner("bsa")
    cfg = _written_config(
        runner,
        FreeSASABSAInput(structure_path=_pdb, partner1="A,B", partner2="C", algorithm="ShrakeRupley"),
        tmp_path,
    )
    assert cfg["mode"] == "bsa"
    assert cfg["partner1"] == "A,B" and cfg["partner2"] == "C"
    assert cfg["algorithm"] == "ShrakeRupley"


def test_bsa_overlapping_partners_rejected(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASABSAInput

    runner = _make_runner("bsa")
    with pytest.raises(AutobioError, match="overlap"):
        _written_config(
            runner, FreeSASABSAInput(structure_path=_pdb, partner1="A", partner2="A"), tmp_path
        )


def test_info_snapshot_freesasa() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("freesasa"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["sasa", "bsa"]
    sasa = parsed["modes"][0]
    struct = sasa["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    assert "output_schema" in sasa
    bsa = parsed["modes"][1]
    assert "partner1" in bsa["input_schema"]["properties"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_freesasa.py -v`
Expected: FAIL — `FreeSASASASAInput`/`FreeSASABSAInput` do not exist; `freesasa` not in `CATALOG`; flat names still present.

- [ ] **Step 3a: Add the freesasa input schemas**

In `src/autobio/schemas/scoring.py`, update the imports (add `Literal`, the hint helpers, and `Field` is already imported):

```python
from typing import Any, Literal

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
```

Add the freesasa input schemas (after `ScoringInput`):

```python
class FreeSASABaseInput(BaseInput):
    """Shared input for FreeSASA modes (SASA and BSA)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    algorithm: Literal["LeeRichards", "ShrakeRupley"] = Field(
        default="LeeRichards",
        description="SASA computation algorithm.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.ADVANCED, order=10),
    )
    probe_radius: float = Field(
        default=1.4,
        gt=0,
        description="Solvent probe radius.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="Å", step=0.1, order=11),
    )
    per_residue: bool = Field(
        default=False,
        description="Return per-residue values in addition to totals.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=12),
    )


class FreeSASASASAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``sasa`` mode (solvent-accessible surface area)."""


class FreeSASABSAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``bsa`` mode (buried surface area at an interface)."""

    partner1: str = Field(
        description="Comma-separated chain IDs for interface partner 1 (e.g. 'A,B').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    partner2: str = Field(
        description="Comma-separated chain IDs for interface partner 2 (e.g. 'C').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
```

- [ ] **Step 3b: Rewire the runner + registration**

In `src/autobio/tools/freesasa.py`:

Update imports (add catalog + new schemas; keep `ScoringOutput`/`ScoredStructure`):

```python
from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import (
    FreeSASABaseInput,
    FreeSASABSAInput,
    ScoredStructure,
    ScoringOutput,
)
from autobio.tools.base import ToolRunner
```

(Remove the old `from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry` — replace with the `ToolCategory` import above; `TOOL_REGISTRY`/`ToolEntry` are no longer used in this module.)

Set `_CONSUMED_EXTRA_KEYS` to empty (all former extra keys are now typed fields; nothing is consumed from `extra`):

```python
# All formerly-consumed keys are now typed fields; nothing is stripped from extra.
_CONSUMED_EXTRA_KEYS: frozenset[str] = frozenset()
```

Replace `prepare_workspace`:

```python
    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, FreeSASABaseInput)
        assert self.current_mode is not None
        is_bsa = self.current_mode.name == "bsa"

        self._validate_inputs(input_data, is_bsa=is_bsa)

        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        config: dict[str, Any] = {
            "mode": "bsa" if is_bsa else "sasa",
            "structure_path": container_structure_path,
            "algorithm": input_data.algorithm,
            "probe_radius": input_data.probe_radius,
            "per_residue": input_data.per_residue,
            "output_dir": "/workspace/outputs/raw",
        }

        if is_bsa:
            assert isinstance(input_data, FreeSASABSAInput)
            config["partner1"] = input_data.partner1
            config["partner2"] = input_data.partner2

        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)
```

Replace `_validate_inputs` (drop the algorithm/probe_radius/partner-presence checks now enforced by the schema; keep structure existence/suffix + partner content checks):

```python
    @staticmethod
    def _validate_inputs(input_data: FreeSASABaseInput, *, is_bsa: bool) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        suffix = input_data.structure_path.suffix.lower()
        if suffix != ".pdb":
            raise AutobioError(
                f"FreeSASA only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using FreeSASA."
            )

        if is_bsa:
            assert isinstance(input_data, FreeSASABSAInput)
            p1_chains = {c.strip() for c in input_data.partner1.split(",")}
            p2_chains = {c.strip() for c in input_data.partner2.split(",")}
            if not p1_chains or any(c == "" for c in p1_chains):
                raise AutobioError("partner1 contains empty chain IDs.")
            if not p2_chains or any(c == "" for c in p2_chains):
                raise AutobioError("partner2 contains empty chain IDs.")
            overlap = p1_chains & p2_chains
            if overlap:
                raise AutobioError(
                    f"partner1 and partner2 chains must not overlap. "
                    f"Overlapping chains: {', '.join(sorted(overlap))}."
                )
```

`parse_output` is unchanged.

Replace the two `TOOL_REGISTRY["freesasa_bsa"]=...` / `TOOL_REGISTRY["freesasa_sasa"]=...` blocks at the bottom of the file with a single catalog registration (keep the `_BSA_NOTES`/`_SASA_NOTES` tuples and attach them to the modes):

```python
register(
    Tool(
        name="freesasa",
        display_name="FreeSASA",
        category=ToolCategory.SCORING,
        description=(
            "Solvent-accessible surface area (SASA) and buried surface area (BSA) via "
            "FreeSASA. CPU-only, no GPU required."
        ),
        version="2.2.1",
        image_tag="freesasa:2.2.1",
        requires_gpu=False,
        gpu_count=0,
        default_mode="sasa",
        modes={
            "sasa": Mode(
                name="sasa",
                display_name="SASA",
                description="Solvent-accessible surface area of a structure.",
                input_schema=FreeSASASASAInput,
                output_schema=ScoringOutput,
                default_timeout=300,
                notes=_SASA_NOTES,
            ),
            "bsa": Mode(
                name="bsa",
                display_name="BSA",
                description="Buried surface area at a protein-protein interface.",
                input_schema=FreeSASABSAInput,
                output_schema=ScoringOutput,
                default_timeout=300,
                notes=_BSA_NOTES,
            ),
        },
        keywords=("sasa", "bsa", "surface area", "interface", "freesasa"),
    )
)
```

(Delete the now-unused `_BSA_INPUT_FORMAT`/`_SASA_INPUT_FORMAT` tuples — `input_format` is a `ToolEntry` concept and no longer used.)

In `src/autobio/tools/__init__.py`, remove the two flat entries and add the Tool-name entry:

```python
    # (remove) "freesasa_bsa": FreeSASARunner,
    # (remove) "freesasa_sasa": FreeSASARunner,
    "freesasa": FreeSASARunner,
```

- [ ] **Step 4: Run tests to verify they pass, then the full suite**

Run: `python -m pytest tests/unit/test_freesasa.py -v`
Expected: PASS.

Run: `python -m pytest -m "not docker and not gpu" -q`
Expected: PASS (this task changes shared CLI/registry surfaces — confirm nothing else broke). If `tests/integration/test_freesasa_integration.py` references the flat names, update it to use `tool="freesasa"` + `mode=` (integration tests are not run here, but keep them consistent).

- [ ] **Step 5: Lint, format, mypy, commit**

```bash
ruff check --fix src/autobio/schemas/scoring.py src/autobio/tools/freesasa.py src/autobio/tools/__init__.py tests/unit/test_freesasa.py
ruff format src/autobio/schemas/scoring.py src/autobio/tools/freesasa.py src/autobio/tools/__init__.py tests/unit/test_freesasa.py
mypy src/
git add src/autobio/schemas/scoring.py src/autobio/tools/freesasa.py src/autobio/tools/__init__.py tests/unit/test_freesasa.py
git commit -m "freesasa: migrate to catalog Tool with sasa/bsa modes + typed fields"
```

---

### Task 7: Migrate the **esm** family (single-mode × 2 Tools + SequenceSet)

Convert `esm1b`/`esm2` into two single-mode Tools (`embed`). Swap `sequences` to `GenericSequenceSet` with a field-level sequence hint; promote `layer`/`pooling` (and esm2's `checkpoint`) to typed fields with hints. Runner keys model resolution on `self.tool.name`; container `config.json` (resolved `model_name`, `layer`, `pooling`, `hf_cache`) is unchanged.

**Files:**
- Modify: `src/autobio/schemas/embedding.py` (add ESM input schemas), `src/autobio/tools/esm.py`, `src/autobio/tools/__init__.py`
- Test: `tests/unit/test_esm.py` (rewrite)

**Interfaces:**
- Consumes: `autobio.schemas.sequences.GenericSequenceSet`; `autobio.schemas.hints.{Tier, Widget, ui}`; `autobio.core.catalog.{Mode, Tool, register, get_tool}`.
- Produces: schemas `ESMEmbedInput` (esm1b) and `ESM2Input` (esm2, adds `checkpoint`); catalog Tools `esm1b`/`esm2`, each mode `embed`.

- [ ] **Step 1: Write the failing test**

```python
# Rewrite tests/unit/test_esm.py
"""Unit tests for the migrated esm1b / esm2 Tools (mode: embed)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.tools.esm import ESMRunner


def _make_runner(tool_name: str) -> ESMRunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = ESMRunner(tool_name, AutobioConfig.resolve())
    runner.current_mode = get_tool(tool_name).modes["embed"]
    return runner


def _written_config(runner: ESMRunner, input_data, tmp_path: Path) -> dict:
    from autobio.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


def test_esm_registered_as_single_mode_tools() -> None:
    import autobio.tools  # noqa: F401
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY

    assert "esm1b" in CATALOG and "esm2" in CATALOG
    assert set(get_tool("esm1b").modes) == {"embed"}
    assert "esm1b" not in TOOL_REGISTRY and "esm2" not in TOOL_REGISTRY


def test_esm1b_config_model_name(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    cfg = _written_config(runner, ESMEmbedInput(sequences={"s1": "MKT"}), tmp_path)
    assert cfg["model_name"] == "facebook/esm1b_t33_650M_UR50S"
    assert cfg["pooling"] == "mean"
    assert cfg["input_fasta"] == "/workspace/inputs/sequences.fasta"


def test_esm2_checkpoint_resolves_model(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESM2Input

    runner = _make_runner("esm2")
    cfg = _written_config(runner, ESM2Input(sequences={"s1": "MKT"}, checkpoint="150M"), tmp_path)
    assert cfg["model_name"] == "facebook/esm2_t30_150M_UR50D"


def test_esm_accepts_fasta_text(tmp_path: Path) -> None:
    from autobio.schemas.embedding import ESMEmbedInput

    runner = _make_runner("esm1b")
    inp = ESMEmbedInput(sequences=">s1\nMKT\n>s2\nGGG\n")
    assert inp.sequences == {"s1": "MKT", "s2": "GGG"}  # GenericSequenceSet normalized it
    cfg = _written_config(runner, inp, tmp_path)
    assert cfg["model_name"] == "facebook/esm1b_t33_650M_UR50S"


def test_info_snapshot_esm2() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("esm2"), OutputFormat.JSON))
    props = parsed["modes"][0]["input_schema"]["properties"]
    assert props["sequences"]["x-autobio"]["widget"] == "sequence"
    assert props["sequences"]["x-autobio"]["flavor"] == "generic"
    assert props["checkpoint"]["default"] == "650M"
    assert "output_schema" in parsed["modes"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_esm.py -v`
Expected: FAIL — `ESMEmbedInput`/`ESM2Input` do not exist; `esm1b`/`esm2` not in `CATALOG`.

- [ ] **Step 3a: Add the ESM input schemas**

In `src/autobio/schemas/embedding.py`, update imports:

```python
from typing import Literal

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
from autobio.schemas.sequences import GenericSequenceSet
```

Add the ESM input schemas (after `EmbeddingInput`, which stays for now):

```python
class ESMEmbedInput(BaseInput):
    """Input for ESM embedding (esm1b): sequences + layer/pooling."""

    sequences: GenericSequenceSet = Field(
        description="Protein sequences: a dict of id→sequence, FASTA text, or a FASTA file path.",
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="generic", tier=Tier.PRIMARY, order=0),
    )
    layer: int | None = Field(
        default=None,
        description="Model layer to extract embeddings from (None = final layer).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    pooling: Literal["mean", "cls", "per_residue"] = Field(
        default="mean",
        description="Pooling strategy for per-residue embeddings.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.PRIMARY, order=1),
    )


class ESM2Input(ESMEmbedInput):
    """Input for ESM-2 (adds checkpoint size selection)."""

    checkpoint: Literal["8M", "35M", "150M", "650M", "3B", "15B"] = Field(
        default="650M",
        description="ESM-2 checkpoint size.",
        json_schema_extra=ui(
            widget=Widget.SELECT,
            tier=Tier.PRIMARY,
            order=2,
            enum_labels={
                "8M": "8M (t6)", "35M": "35M (t12)", "150M": "150M (t30)",
                "650M": "650M (t33, default)", "3B": "3B (t36)", "15B": "15B (t48)",
            },
        ),
    )
```

- [ ] **Step 3b: Rewire the runner + registration**

In `src/autobio/tools/esm.py`:

Update imports (add catalog + new schemas; keep `EmbeddingOutput`/`SequenceEmbedding`):

```python
from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.embedding import (
    ESM2Input,
    ESMEmbedInput,
    EmbeddingOutput,
    SequenceEmbedding,
)
from autobio.tools.base import ToolRunner
from autobio.utils.sequences import validate_protein_sequence, write_fasta
```

(Remove `from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry` — replace with the `ToolCategory` import above.)

Set `_CONSUMED_EXTRA_KEYS = frozenset()` (checkpoint is now a typed field on `ESM2Input`, not an `extra` key).

Replace `prepare_workspace` (assert on `ESMEmbedInput`; the rest is the same shape — `input_data.sequences` is a normalized `dict[str, str]` after validation):

```python
    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input FASTA into the workspace."""
        assert isinstance(input_data, ESMEmbedInput)

        model_cfg = self._resolve_model_config(input_data)
        self._validate_inputs(input_data, model_cfg)

        write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")

        config: dict[str, object] = {
            "model_name": model_cfg["model_name"],
            "input_fasta": "/workspace/inputs/sequences.fasta",
            "output_dir": "/workspace/outputs/raw",
            "layer": input_data.layer,
            "pooling": input_data.pooling,
            "hf_cache": _HF_CACHE,
        }
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)
```

Replace `_resolve_model_config` (key on `self.tool.name`; read the typed `checkpoint` for esm2):

```python
    def _resolve_model_config(self, input_data: ESMEmbedInput) -> dict[str, str | int]:
        """Return the model config for the current Tool and (esm2) checkpoint."""
        assert self.tool is not None
        if self.tool.name == "esm1b":
            return _ESM1B_CONFIG
        # esm2 — checkpoint is a validated Literal on ESM2Input
        assert isinstance(input_data, ESM2Input)
        checkpoint = input_data.checkpoint
        if checkpoint not in _ESM2_CHECKPOINTS:
            available = ", ".join(sorted(_ESM2_CHECKPOINTS))
            raise AutobioError(
                f"Unknown ESM-2 checkpoint {checkpoint!r}. Available checkpoints: {available}"
            )
        return _ESM2_CHECKPOINTS[checkpoint]
```

Update `_validate_inputs`'s signature type to `ESMEmbedInput` and drop the pooling-membership check (now enforced by the `Literal`); keep the sequence and layer-range checks:

```python
    @staticmethod
    def _validate_inputs(input_data: ESMEmbedInput, model_cfg: dict[str, str | int]) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.sequences:
            raise AutobioError("sequences must be non-empty.")
        for seq_id, seq in input_data.sequences.items():
            if not validate_protein_sequence(seq):
                raise AutobioError(
                    f"Invalid protein sequence for {seq_id!r}: "
                    f"must contain only standard amino acid characters (ACDEFGHIKLMNPQRSTVWY)."
                )
        num_layers = int(model_cfg["num_layers"])
        if input_data.layer is not None and not (0 <= input_data.layer <= num_layers):
            raise AutobioError(
                f"layer must be between 0 and {num_layers} for "
                f"{model_cfg['model_name']}, got {input_data.layer}."
            )
```

`parse_output` is unchanged.

Replace the two `TOOL_REGISTRY[...]` blocks at the bottom with two catalog registrations (keep `_ESM_NOTES`/`_ESM2_NOTES`):

```python
register(
    Tool(
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
)

register(
    Tool(
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
)
```

(Delete the now-unused `_ESM_INPUT_FORMAT` tuple.)

In `src/autobio/tools/__init__.py`: the entries `"esm1b": ESMRunner` and `"esm2": ESMRunner` STAY (names unchanged); no edit needed there for esm.

- [ ] **Step 4: Run tests, then the full suite**

Run: `python -m pytest tests/unit/test_esm.py -v`
Expected: PASS.

Run: `python -m pytest -m "not docker and not gpu" -q`
Expected: PASS (confirm shared surfaces intact). Update `tests/integration/test_esm_integration.py` for `mode="embed"`/typed `checkpoint` if it references the old `extra["checkpoint"]` form (integration not run here).

- [ ] **Step 5: Lint, format, mypy, commit**

```bash
ruff check --fix src/autobio/schemas/embedding.py src/autobio/tools/esm.py tests/unit/test_esm.py
ruff format src/autobio/schemas/embedding.py src/autobio/tools/esm.py tests/unit/test_esm.py
mypy src/
git add src/autobio/schemas/embedding.py src/autobio/tools/esm.py tests/unit/test_esm.py
git commit -m "esm: migrate esm1b/esm2 to single-mode catalog Tools with SequenceSet + typed fields"
```

---

## Self-Review

**1. Spec coverage** (against the Phase-1 scope of `docs/superpowers/specs/2026-07-02-autobio-tools-modes-refactor-design.md` and the Plan-1 carry-forward):
- Per-mode image support (needed by rosetta/openmm later) → Task 1. ✓
- Runner dispatch on `(tool, mode)` instead of tool-name string; `run(..., mode=...)`; container config unchanged → Task 2 + Tasks 6/7. ✓
- `get_runner` addresses migrated Tools → covered (no change needed; verified in Task 6). ✓
- `list`/`info`/`run --mode` contracts with `output_schema` + `x-autobio` in `info` → Tasks 3/4/5 (transition-level; full §7 unified schema polish deferred to Plan 3). ✓
- Promote `extra` keys to typed fields with hints + update `_CONSUMED_EXTRA_KEYS` (no double-write) → Tasks 6/7. ✓
- `SequenceSet` on a real field with a field-level hint → Task 7 (esm). ✓
- Multi-mode schema dispatch + shared base → Task 6 (freesasa base + per-mode subclasses). ✓
- Clean break (flat names removed for migrated tools) → Tasks 6/7 + tests asserting removal. ✓
- Snapshot-style `info` test per representative tool (multi-mode + single-mode/SequenceSet) → Tasks 6/7. ✓
- Deferred (correctly out of this plan): remaining ~11 families, old-registry teardown/rename, cross-category + two-class consolidation, antibody SequenceSet, full §7 unified `list` schema.

**2. Placeholder scan:** No "TBD"/"similar to Task N"/"handle edge cases". Every code step shows complete code; every test step shows complete tests; commands include the `python -m pytest` form and expected outcomes.

**3. Type consistency:** `self.current_mode`/`self.tool`/`self.entry` and the helpers (`_resolve_mode`, `_image_tag`, `_default_timeout`, `_requires_gpu`, `_gpu_count`, `_tool_version`) are defined in Task 2 and consumed unchanged in Tasks 3–7. `format_tool_info_catalog`/`format_tool_list_merged` signatures match between formatters (Tasks 3/4) and their callers (`info.py`/`list.py`). Schema class names (`FreeSASABaseInput`/`FreeSASASASAInput`/`FreeSASABSAInput`, `ESMEmbedInput`/`ESM2Input`) are consistent between the schema modules, runners, registrations, and tests. `Mode.image_tag` (Task 1) is read only in `_image_tag` (Task 2).

---

## Next plans (not in scope here)

- **Plan 3 (Phase 1 cont.) — remaining families:** migrate the rest against this proven pattern, grouped by shape: same-category multi-mode (rosetta [per-mode `image_tag`], evoef2, complexa), cross-category + two-class consolidation (esm_if1, antifold, ligandmpnn), output-schema variance (openmm [per-mode image + SimulationOutput], antibody LMs ×6 [antibody `SequenceSet`, output variance, shared runner]), and remaining singletons (esmfold, chai1, boltz1/2, proteinmpnn, rfd3, antipasti, baddg, stabddg, prodigy, openfold3).
- **Plan 4 — teardown + contract polish:** once all families are on the catalog, remove `TOOL_REGISTRY`/`ToolEntry`/legacy CLI paths, collapse `get_runner`/CLI to catalog-only, finalize the unified §7 `list` JSON schema, decide the `CATALOG`↔`TOOL_REGISTRY` naming/home, and rewrite the README to Tools/Modes.
