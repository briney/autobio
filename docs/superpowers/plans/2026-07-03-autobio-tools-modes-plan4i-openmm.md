# OpenMM Tools→Modes Migration (Plan 4i) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Migrate the three flat OpenMM tools (`openmm_amber_minimize`, `openmm_amber_relax`, `openmm_md_simulate`) into ONE catalog `Tool` (`openmm`) with three `Mode`s served by the existing `OpenMMRunner`, with per-mode image/timeout/schema and cross-category (SCORING primary + `md_simulate`→SIMULATION), while preserving the container `config.json` byte-for-byte per mode.

**Architecture:** This is a **minimal catalog migration** (deliberate scope choice). The runner's proven data-driven config construction (`_VARIANT_CONFIG` + `_CONSUMED_EXTRA_KEYS` filtered merge + host-side validation) is preserved verbatim; only the dispatch key changes from `self.tool_name` to `self.current_mode.name`, and the three `TOOL_REGISTRY`/`ToolEntry` registrations become one `Tool` with three `Mode`s. Full field-promotion (typed fields + `x-autobio` hints for openmm's ~20 params) is an explicitly DEFERRED enhancement — do NOT do it in this plan.

**Tech Stack:** Python 3.11+, Pydantic, pytest. Exemplars: `src/autobio/tools/rosetta.py` (per-mode `Mode.image_tag`), `src/autobio/tools/esm_if1.py` (cross-category `Mode.category` + test shape).

## Global Constraints

- **Byte-compat:** container `config.json` per mode MUST be identical (exact keys, values, and key ORDER) to the pre-branch output. `containers/` is NOT touched. The strongest guard is a full-dict `cfg == expected` test PLUS an explicit `list(cfg.keys()) == list(expected.keys())` key-order assertion, PER MODE. See `.superpowers/sdd/recon/openmm.md` for the exact per-mode key orders.
- **Dispatch on `self.current_mode.name`** (the Mode name: `amber_minimize` / `amber_relax` / `md_simulate`), never the flat tool name.
- **Preserve `_VARIANT_CONFIG` inner dicts byte-identical** — including every `default_*` value and the `protocol` value. Only re-key the OUTER dict from flat names to mode names.
- **Do NOT adopt `_apply_extra`** for openmm. Its params are read from `extra` via `extra.get(param, default)` and written to config under the same key names, so `_apply_extra` would flag them as config-key collisions and raise. Keep the existing `_CONSUMED_EXTRA_KEYS` filtered merge and the `_validate_*` methods verbatim.
- **Per-mode image/timeout/category** via `Mode.image_tag` / `Mode.default_timeout` / `Mode.category` (rosetta + esm_if1 patterns).
- **README is OUT of scope** — `openmm_*` flat-name references in README are deferred to the later teardown plan. Do not touch README.
- Env: run tests with `python -m pytest` (bare `pytest` = wrong env). This pytest config prints dots but omits the "N passed" summary line — verify via exit code.

---

## Task 1: Migrate openmm to a catalog Tool with three modes

**Files:**
- Modify: `src/autobio/tools/openmm.py`
- Modify: `src/autobio/tools/__init__.py` (TOOL_RUNNERS)
- Modify: `tests/unit/test_openmm.py` (unit tests → catalog)
- Modify: `tests/unit/test_openmm_e2e.py`, `tests/unit/test_openmm_relax_e2e.py`, `tests/unit/test_openmm_simulate_e2e.py` (set `current_mode`)

**Interfaces:**
- Consumes: `Mode`, `Tool`, `register`, `get_tool`, `tool_categories` from `autobio.core.catalog`; `ToolCategory` from `autobio.core.registry`; `_image_tag()` in base resolves `current_mode.image_tag or tool.image_tag`; `_apply_extra` is NOT used here.
- Produces: catalog Tool `openmm` with modes `{amber_minimize, amber_relax, md_simulate}`; `TOOL_RUNNERS["openmm"] = OpenMMRunner`; the three `openmm_*` flat names removed from `TOOL_REGISTRY` and `TOOL_RUNNERS`.

### Step 1: Re-point imports in `openmm.py`

Replace the registry import with the catalog import. Change:

```python
from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
```

to:

```python
from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
```

Keep all other imports (`BaseInput, BaseOutput`, scoring/simulation schemas, `ToolRunner`, `AutobioError`, etc.) as-is.

### Step 2: Re-key `_VARIANT_CONFIG` to mode names

Change ONLY the three outer keys; keep every inner dict byte-identical:

- `"openmm_amber_minimize"` → `"amber_minimize"`
- `"openmm_amber_relax"` → `"amber_relax"`
- `"openmm_md_simulate"` → `"md_simulate"`

Do NOT change any inner value (`protocol`, `default_*`, `produces_*`).

### Step 3: Dispatch on `self.current_mode.name`

In `prepare_workspace` and `parse_output`, replace `variant_cfg = _VARIANT_CONFIG[self.tool_name]` with:

```python
assert self.current_mode is not None
variant_cfg = _VARIANT_CONFIG[self.current_mode.name]
```

Leave `_prepare_scoring_workspace`, `_prepare_simulation_workspace`, `_parse_scoring_output`, `_parse_simulation_output`, `_resolve_container_path`, and all `_validate_*` methods UNCHANGED.

### Step 4: Replace the three registrations with one catalog Tool

Delete the three `TOOL_REGISTRY[...] = ToolEntry(...)` blocks and the three `_*_INPUT_FORMAT` constants. KEEP the three `_*_NOTES` tuples. Fold each mode's input-format guidance into its notes by appending the `_*_INPUT_FORMAT` text as trailing note entries (so no guidance is lost), i.e. define per-mode note tuples like:

```python
_MINIMIZE_MODE_NOTES = _AMBER_MINIMIZE_NOTES + _AMBER_MINIMIZE_INPUT_FORMAT
_RELAX_MODE_NOTES = _AMBER_RELAX_NOTES + _AMBER_RELAX_INPUT_FORMAT
_SIMULATE_MODE_NOTES = _MD_SIMULATE_NOTES + _MD_SIMULATE_INPUT_FORMAT
```

(Keep the `_*_INPUT_FORMAT` tuples as module constants used only to build these — that keeps them live, not dead code.)

Then register the Tool:

```python
OPENMM_TOOL = Tool(
    name="openmm",
    display_name="OpenMM",
    category=ToolCategory.SCORING,
    description=(
        "OpenMM molecular mechanics engine (Amber force field). Modes: "
        "amber_minimize (AlphaFold-style energy minimization), amber_relax "
        "(full relaxation with explicit solvent), and md_simulate (production "
        "molecular dynamics with trajectory output)."
    ),
    version="1.1.0",
    image_tag="openmm-amber-minimize:1.1.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="amber_minimize",
    modes={
        "amber_minimize": Mode(
            name="amber_minimize",
            display_name="Amber minimize",
            description=(
                "Amber force-field energy minimization with iterative violation "
                "checking (AlphaFold-style). Produces a refined PDB and energy in kJ/mol."
            ),
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            image_tag="openmm-amber-minimize:1.1.0",
            notes=_MINIMIZE_MODE_NOTES,
        ),
        "amber_relax": Mode(
            name="amber_relax",
            display_name="Amber relax",
            description=(
                "Full relaxation with explicit solvent (default): solvation, "
                "minimization, heating, NVT/NPT equilibration, and short production. "
                "Returns a refined, solvent-stripped structure with energy in kJ/mol."
            ),
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=3600,
            image_tag="openmm-amber-relax:1.1.0",
            notes=_RELAX_MODE_NOTES,
        ),
        "md_simulate": Mode(
            name="md_simulate",
            display_name="MD simulate",
            description=(
                "Production molecular dynamics with the Amber force field and "
                "explicit solvent. Produces a trajectory (DCD/XTC/PDB), energy "
                "time series, and a final protein-only PDB."
            ),
            input_schema=SimulationInput,
            output_schema=SimulationOutput,
            default_timeout=86400,
            image_tag="openmm-md-simulate:1.1.0",
            category=ToolCategory.SIMULATION,
            notes=_SIMULATE_MODE_NOTES,
        ),
    },
    keywords=(
        "openmm", "molecular dynamics", "md", "minimize", "relax",
        "simulation", "amber", "force field", "energy",
    ),
)
"""Catalog Tool for OpenMM (amber_minimize + amber_relax + md_simulate modes)."""

register(OPENMM_TOOL)
```

### Step 5: Update `TOOL_RUNNERS` in `tools/__init__.py`

Remove the three `openmm_amber_minimize`/`openmm_amber_relax`/`openmm_md_simulate` entries and add one:

```python
    "openmm": OpenMMRunner,
```

### Step 6: Rewrite `tests/unit/test_openmm.py` for the catalog

Follow the `tests/unit/test_esm_if1.py` shape (read it). Requirements:

- A runner fixture parametrized/keyed by mode that pins `current_mode`:
  ```python
  runner = OpenMMRunner("openmm", config)
  runner.current_mode = get_tool("openmm").modes[mode_name]
  ```
- **Byte-compat config tests, ONE PER MODE**: build the config via `prepare_workspace`, then assert BOTH full-dict equality (`cfg == expected`) AND `list(cfg.keys()) == list(expected.keys())`. Use the exact key orders in `.superpowers/sdd/recon/openmm.md`. Include: (a) defaults-only for each mode; (b) at least one mode with `extra` overrides of consumed params (e.g. `temperature`, `force_field`) proving they override the default in-place and do NOT duplicate; (c) at least one mode with a NON-consumed extra key proving it flat-merges AFTER the fixed keys; (d) md_simulate with `extra={"n_steps": ...}` proving `n_steps` lands after the default loop and before flat extra.
- **Preserve all existing validation tests** (invalid force_field/restraint_set/water_model/box_shape/ion_type, negative temperature/pressure/box_padding/ion_concentration, invalid timestep, invalid trajectory_format, missing structure) — port them to the new fixture. They must still raise the same `AutobioError` messages.
- **Preserve parse_output tests** for both scoring (minimize/relax) and simulation (md_simulate) branches.
- **Replace `TestOpenMMRegistration`** (which asserted `TOOL_REGISTRY` membership) with catalog assertions:
  - `get_tool("openmm")` exists; `sorted(get_tool("openmm").modes) == ["amber_minimize", "amber_relax", "md_simulate"]`; `default_mode == "amber_minimize"`.
  - Per-mode `input_schema`/`output_schema` correct (ScoringInput/ScoringOutput ×2; SimulationInput/SimulationOutput).
  - Per-mode `image_tag` correct (minimize→openmm-amber-minimize:1.1.0, relax→openmm-amber-relax:1.1.0, md_simulate→openmm-md-simulate:1.1.0) and `Tool.image_tag == "openmm-amber-minimize:1.1.0"`.
  - Per-mode `default_timeout` (600/3600/86400).
  - **Cross-category**: `tool_categories(get_tool("openmm")) == (ToolCategory.SCORING, ToolCategory.SIMULATION)`; `"openmm" in list_tools(ToolCategory.SCORING)` and `"openmm" in list_tools(ToolCategory.SIMULATION)`.
  - `TOOL_RUNNERS["openmm"] is OpenMMRunner`; the three flat names are absent from `TOOL_RUNNERS` and `TOOL_REGISTRY`.
- **`_apply_extra` NOT applicable**: instead add a test proving the consumed-key filter still works — e.g. `extra={"restraint_set": "ca"}` overrides config in place (no duplicate key), and a non-consumed key merges through. (Do NOT test for a "collide" AutobioError — openmm doesn't use `_apply_extra`.)
- **`info` snapshot**: `format_tool_info_catalog(get_tool("openmm"), OutputFormat.JSON)` parses and contains the three modes + Tool/Mode notes (mirror the esm_if1 info test).
- **Full `run()` lifecycle** for one scoring mode and the simulation mode (mock container, like esm_if1's), asserting `metadata.mode == "<mode>"` and the correct output type. Follow esm_if1's `r.run(input_data, gpu="none", output_dir=..., mode=mode_name)` pattern.

### Step 7: Update the three e2e test files

In `test_openmm_e2e.py`, `test_openmm_relax_e2e.py`, `test_openmm_simulate_e2e.py`, the `_make_runner(tool_name, config)` helper builds `OpenMMRunner(tool_name, config)` and callers pass the old flat names, then call `prepare_workspace` directly (bypassing `run()`, so `current_mode` is unset). Update `_make_runner` to accept a MODE name, construct `OpenMMRunner("openmm", config)`, and set `runner.current_mode = get_tool("openmm").modes[mode]`. Update call sites: `"openmm_amber_minimize"`→`"amber_minimize"`, `"openmm_amber_relax"`→`"amber_relax"`, `"openmm_md_simulate"`→`"md_simulate"`. Add the `from autobio.core.catalog import get_tool` import. Do NOT change the assertions on config contents / outputs (byte-compat). These tests may be Docker-gated; the edits here are Docker-free (the helper + prepare_workspace path).

### Step 8: Run checks and commit

```bash
python -m pytest tests/unit/test_openmm.py tests/unit/test_openmm_e2e.py tests/unit/test_openmm_relax_e2e.py tests/unit/test_openmm_simulate_e2e.py tests/unit/test_registry_disjoint.py -q
python -m pytest -m "not docker and not gpu" -q   # exit 0
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
git add -A && git commit -m "openmm: migrate to catalog Tool with amber_minimize/relax/md_simulate modes"
```

Expected: all green; `openmm` in CATALOG with 3 modes; cross-category (SCORING+SIMULATION); three flat names gone from both registries; byte-compat config preserved per mode.

---

## Self-Review checklist (controller, before dispatch)
- [ ] `_VARIANT_CONFIG` inner dicts unchanged (only outer keys re-keyed) → byte-compat by construction.
- [ ] No `_apply_extra` adoption; `_CONSUMED_EXTRA_KEYS` + `_validate_*` preserved.
- [ ] Per-mode image/timeout/category; cross-category union (SCORING, SIMULATION).
- [ ] Byte-compat tests assert full-dict + key-order per mode.
- [ ] README untouched (deferred to teardown).
