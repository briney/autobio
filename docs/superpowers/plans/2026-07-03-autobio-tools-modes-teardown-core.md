# Tools→Modes Teardown — Core Legacy-Registry Removal (Plan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Now that all 28 tools live on the `CATALOG` and `TOOL_REGISTRY` is permanently empty, remove the legacy flat registry (`ToolEntry`, `TOOL_REGISTRY`, registry's `get_tool`/`list_tools`) and the `ToolRunner` dual-path, and migrate the test infrastructure that used the legacy registry to the catalog.

**Architecture:** This is the first of two teardown PRs. This PR removes only the legacy *registry + runner dual-path + CLI legacy branches* and reworks the tests that depended on them. Schema cleanup, carry-forward refactors (`_resolve_container_path` hoist, etc.), and the README/dev-spec rewrite are a SEPARATE follow-up PR — do NOT do them here. `ToolCategory` (the StrEnum) STAYS in `registry.py` — it is used everywhere and is independent of `ToolEntry`/`TOOL_REGISTRY`.

**Tech Stack:** Python 3.11+, Pydantic, Typer CLI, pytest.

**Recon fact sheet (READ IT):** `.superpowers/sdd/recon/teardown.md` — has the exhaustive, line-referenced consumer lists for every symbol removed here (§1, §2, §6). Trust it for "which files/lines"; this plan gives the structure + the tricky bits.

## Global Constraints

- **`ToolCategory` STAYS** in `src/autobio/core/registry.py`. Only `ToolEntry`, `TOOL_REGISTRY`, and registry.py's `get_tool`/`list_tools` are removed. `catalog.py` has its own `get_tool`/`list_tools` (the catalog versions) — those stay.
- **Every commit must be green** (`python -m pytest -m "not docker and not gpu"` exit 0). Because production and tests both reference the legacy symbols, Task 1 migrates ALL consumers (prod + test infra) OFF the legacy registry while the symbols still exist; Task 2 then deletes the now-unused symbols. Do not reorder.
- **No behavior change for catalog tools.** The runner dual-path removal is pure dead-code elimination: for all 28 real tools `self.tool` is always set and `self.entry` always `None` today, so every `self.entry`/`if self.tool is None` branch is unreachable. (Recon §2.)
- **Out of scope (SECOND teardown PR):** unused schema removal; `_resolve_container_path` hoist and other carry-forwards; README/`TOOL_SPEC.md`/`SCHEMA_SPEC.md` rewrite; test-import hygiene (the per-method `import autobio.tools`); the templates/msa latent-bug documentation. Do not touch these here. Do NOT touch `containers/`.
- Env: `python -m pytest` (bare = wrong env); this config omits the "N passed" line — verify via exit code.

---

## Task 1: Remove the runner dual-path + CLI legacy branches; migrate test infra to the catalog

**Files (modify):**
- `src/autobio/tools/base.py`
- `src/autobio/cli/info.py`, `src/autobio/cli/images.py`, `src/autobio/cli/list.py`, `src/autobio/cli/run.py`, `src/autobio/cli/formatters.py`
- `tests/unit/test_tool_runner.py`, `tests/unit/test_tool_runner_modes.py`, `tests/unit/test_cli.py`, `tests/unit/test_formatters.py`

Do NOT delete anything from `registry.py` yet (Task 2). The legacy symbols must still exist so this task's intermediate state stays importable/green.

### base.py (recon §2)
- Remove `from autobio.core.registry import TOOL_REGISTRY, ToolEntry` (keep the catalog import).
- `__init__`: drop `self.entry`; resolve the tool from the catalog only:
  ```python
  def __init__(self, tool_name: str, config: AutobioConfig) -> None:
      if tool_name not in CATALOG:
          available = ", ".join(sorted(CATALOG)) or "(none)"
          raise KeyError(f"Unknown tool {tool_name!r}. Available tools: {available}")
      self.tool: Tool = get_tool(tool_name)
      self.tool_name = tool_name
      self.config = config
      self.current_mode: Mode | None = None
      self._container = ContainerManager(config)
      self._gpu = GPUManager()
  ```
  (`self.tool` is now non-optional `Tool`.)
- `_resolve_mode`: drop the `if self.tool is None:` legacy branch; always resolve a Mode:
  ```python
  def _resolve_mode(self, mode: str | None) -> Mode:
      name = mode if mode is not None else self.tool.default_mode
      try:
          return self.tool.modes[name]
      except KeyError:
          available = ", ".join(sorted(self.tool.modes))
          raise AutobioError(
              f"Unknown mode {name!r} for tool {self.tool_name!r}. Available modes: {available}"
          ) from None
  ```
- `_image_tag`, `_default_timeout`, `_requires_gpu`, `_gpu_count`, `_tool_version`: drop the `assert self.entry is not None; return self.entry.X` tails; make the catalog body unconditional. E.g.:
  ```python
  def _image_tag(self) -> str:
      assert self.current_mode is not None
      return self.current_mode.image_tag or self.tool.image_tag
  def _default_timeout(self) -> int:
      assert self.current_mode is not None
      return self.current_mode.default_timeout
  def _requires_gpu(self) -> bool:
      return self.tool.requires_gpu
  def _gpu_count(self) -> int:
      return self.tool.gpu_count
  def _tool_version(self) -> str:
      return self.tool.version
  ```
- `_build_metadata`: `mode=self.current_mode.name if self.current_mode is not None else None` may stay (harmless) — `current_mode` is always set after `run()`; leave as-is.
- Update the `run()` docstring: remove the "Legacy tools ignore *mode* (passing one raises)" language and the `AutobioError ... mode passed to a legacy tool` note — there are no legacy tools now.

### CLI (recon §1b)
- `cli/info.py`: remove the `TOOL_REGISTRY` fallback branch (the `else`/`get_registry_tool`+`format_tool_info` path). Catalog-only: if the tool isn't in `CATALOG`, error. Drop the now-unused imports (`TOOL_REGISTRY`, `get_registry_tool`, `format_tool_info`).
- `cli/images.py`: remove the `TOOL_REGISTRY` merge for `--all` and single-tool pulls; use `CATALOG` only. Drop `TOOL_REGISTRY`/registry `get_tool` imports.
- `cli/list.py`: remove `list_registry_tools()` (always `{}`) and the merge; call the catalog-only formatter (see below). Drop the `list_registry_tools` import.
- `cli/run.py`: remove the dead `assert runner.entry is not None  # legacy branch` (recon §1b, `cli/run.py:77`).
- `cli/formatters.py`:
  - Delete `format_tool_list` (the `ToolEntry`-based one, lines ~32–77 — no production caller).
  - Delete `format_tool_info` (the `ToolEntry`-based one, line ~157 — only the removed info.py fallback used it).
  - Rename/simplify `format_tool_list_merged(flat: dict[str, ToolEntry], catalog, fmt)` → `format_tool_list(catalog_tools: dict[str, Tool], fmt=...)` (drop the `flat` param entirely; render the catalog only). Update the `cli/list.py` call site to `format_tool_list(tools, fmt)`.
  - Remove the `ToolEntry` import (TYPE_CHECKING). Keep `format_tool_info_catalog` (the `Tool`-based one) untouched.

### Test infra migration (recon §6b)
Because `__init__` is now catalog-only, any test that registered a mock via `TOOL_REGISTRY`/`ToolEntry` will fail to construct its runner. Convert those to a mock **catalog** `Tool`/`Mode`:
- `tests/unit/test_tool_runner.py`: convert the `_register_mock_tool` autouse fixture + `runner` fixture to register a mock `Tool` (with one `Mode`) into `CATALOG` (snapshot/restore `CATALOG` like `test_tool_runner_modes.py`'s `_clean_catalog` does). Re-target `test_tool_runner.py:116` (`assert runner.entry is ...`) → `assert runner.tool is get_tool(_MOCK_TOOL_NAME)` and `test_default_timeout_from_entry` → assert from `runner.current_mode.default_timeout`. Keep ALL lifecycle coverage (`__init__`, `_resolve_gpu`, `_build_metadata`, `_read_logs`, full `run()`, `get_runner`).
- `tests/unit/test_tool_runner_modes.py`: delete `_register_fake_legacy_tool()` (lines ~90–103) and `test_run_rejects_mode_for_legacy_tool()` (lines ~206–215); remove the `TOOL_REGISTRY` snapshot/restore lines from the `_clean_catalog` fixture (lines ~22, 27–28). The other 11 catalog tests stay.
- `tests/unit/test_cli.py`: convert the fake-tool injection mechanism from `TOOL_REGISTRY["mock-tool"] = ToolEntry(...)` (+ the `TOOL_REGISTRY` snapshot/clear/restore autouse fixture) to registering a fake `Tool`/`Mode` into `CATALOG`. Fix the now-wrong comment at ~line 543/546/578 that calls `prodigy` "a legacy flat tool (in TOOL_REGISTRY, not CATALOG)".
- `tests/unit/test_formatters.py`: drop the test classes for the deleted `format_tool_list`/`format_tool_info`; rework the `format_tool_list_merged` tests for the new catalog-only `format_tool_list(catalog_tools, fmt)` signature (build `Tool` objects, not `ToolEntry`). Keep `format_tool_info_catalog` tests.

### Verify + commit
```bash
python -m pytest tests/unit/test_tool_runner.py tests/unit/test_tool_runner_modes.py tests/unit/test_cli.py tests/unit/test_formatters.py -q
python -m pytest -m "not docker and not gpu" -q   # exit 0
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
git add -A && git commit -m "teardown: remove ToolRunner legacy dual-path + CLI legacy branches; migrate test infra to catalog"
```
Note: `TOOL_REGISTRY`/`ToolEntry` still EXIST after this task (in registry.py + the direct registry unit tests + the `assert not in TOOL_REGISTRY` guards) — that's expected; Task 2 deletes them.

---

## Task 2: Delete the legacy symbols + vestigial tests

**Files (modify/delete):**
- `src/autobio/core/registry.py`, `src/autobio/core/__init__.py`
- Delete `tests/unit/test_registry_disjoint.py`
- `tests/unit/test_registry.py`
- Remove the `assert "<name>" not in TOOL_REGISTRY` guard lines from the ~20 tool test files listed in recon §6c.

### registry.py
- Delete `ToolEntry`, `TOOL_REGISTRY`, `get_tool`, `list_tools`. Keep `ToolCategory` and its imports. The file now contains only `ToolCategory`. Update the module docstring from "Tool registry — maps tool names to metadata." to something like "Tool category taxonomy." Remove the now-unused `TYPE_CHECKING`/`dataclass`/`BaseInput`/`BaseOutput` imports that only served `ToolEntry`.

### core/__init__.py
- Drop `TOOL_REGISTRY`, `ToolEntry`, `get_tool`, `list_tools` from the `from autobio.core.registry import ...` line (keep `ToolCategory`) and from `__all__`. No consumer imports `get_tool`/`list_tools` from `autobio.core` (recon confirmed zero), so do not re-add them.

### Tests
- Delete `tests/unit/test_registry_disjoint.py` (vacuous once `TOOL_REGISTRY` is gone).
- `tests/unit/test_registry.py`: keep `TestToolCategory`; delete `TestToolEntry`, `TestGetTool`, `TestListTools` and any now-unused imports (`ToolEntry`, `TOOL_REGISTRY`, `get_tool`, `list_tools`).
- In each file listed in recon §6c, delete the `assert "<flat_name>" not in TOOL_REGISTRY` line(s) (they can't compile once the symbol is gone). Leave the surrounding `import autobio.tools` and other assertions intact (test-import hygiene is the second PR). Confirm none of these files still import `TOOL_REGISTRY` after removal; drop the import if it becomes unused.

### Verify + commit
```bash
grep -rn "TOOL_REGISTRY\|ToolEntry" src/ tests/   # expect ZERO hits
python -m pytest -m "not docker and not gpu" -q   # exit 0
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
git add -A && git commit -m "teardown: delete ToolEntry/TOOL_REGISTRY (keep ToolCategory) + vestigial registry tests"
```
Expected: zero `TOOL_REGISTRY`/`ToolEntry` references in `src/` and `tests/`; `ToolCategory` intact; full suite green.

---

## Self-Review checklist (controller, before dispatch)
- [ ] Task 1 leaves `TOOL_REGISTRY`/`ToolEntry` existing but unused by prod + reworked tests (green).
- [ ] Runner dual-path removal changes zero catalog-tool behavior (recon §2).
- [ ] `format_tool_list_merged`→`format_tool_list` catalog-only; deleted formatters have no remaining callers.
- [ ] Task 2: `grep TOOL_REGISTRY|ToolEntry src/ tests/` → zero; `ToolCategory` kept; docstring updated.
- [ ] Out-of-scope items (schemas, `_resolve_container_path` hoist, README/specs, test-import hygiene, templates/msa) NOT touched.
