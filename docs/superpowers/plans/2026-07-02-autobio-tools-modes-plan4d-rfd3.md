# Tools→Modes Plan 4d — rfd3 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate `rfd3` (RFDiffusion3, `STRUCTURE_DESIGN`) from the legacy flat `TOOL_REGISTRY` to the `Tool`/`Mode` catalog — the last structure_design singleton.

**Architecture:** A single-tool, single-mode migration. Lighter than chai/openfold3: `design_specs`/`input_structures`/`n_batches` are already typed fields on the current schema, and `design_specs` stays an untyped escape-hatch dict (a container-validated tool-native mini-language — contig strings, `select_*` fields, symmetry, etc.). The one substantive change is that rfd3 currently flat-merges its entire `extra` dict via `config.update(input_data.extra)` with no filtering — this becomes `self._apply_extra(config, input_data)` (the hardened base helper). CLI knobs (`step_scale`, `gamma_0`, `diffusion_batch_size`, …) stay in `extra` (now discoverable via the Mode notes, which `info` renders as of Plan 4c). Container-side execution is untouched; `config.json` is byte-for-byte preserved for all normal inputs.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env).
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation, or `containers/rfd3/validate_config.sh` (its `test_rfd3_validate_config.py` test is unaffected).
- **Byte-compat is the success criterion:** `config.json` written by `prepare_workspace` must be identical to pre-migration output for all normal inputs (including the deep-copied `design_specs` with rewritten `input` paths, `n_batches`, `out_dir`, and the merged CLI `extra`). Ship a full-dict `config.json` equality test.
- **Do NOT modify `StructureDesignInput`** in `src/autobio/schemas/structure_design.py` — it is ALSO consumed by the unmigrated `complexa` tool. Add a NEW `RFD3Input(BaseInput)`.
- **`design_specs` stays `dict[str, dict[str, Any]]`** (escape-hatch) — do not attempt to type the per-spec mini-language; real validation lives in the container's `validate_config.sh`.
- Adopt `self._apply_extra(config, input_data)` in place of the unfiltered `config.update(input_data.extra)`. rfd3 has no `_CONSUMED_EXTRA_KEYS` today; none is needed (CLI knobs are genuine `extra` config keys, and none collide with `design_specs`/`n_batches`/`out_dir`).
- Catalog `Tool`/`Mode` have no `input_format` field — drop `_RFD3_INPUT_FORMAT`; fold its contig/`select_*` mini-language guidance into the `design_specs` field description. `_RFD3_NOTES` moves onto the `Mode` (rendered by `info` as of Plan 4c).
- Delete the `TOOL_REGISTRY["rfd3"]` block (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS["rfd3"]` stays (Tool name == flat name).
- Single mode named `generate` (matches current runner; makes the existing `DesignedStructure.evaluation_metrics` docstring "None when mode='generate'" accurate). `supports_batch=True` on the Mode (rfd3 batches natively).
- Merged exemplars: `src/autobio/tools/chai.py` / `src/autobio/tools/boltz.py` (catalog Tool, `_apply_extra`, `<TOOL>_TOOL` constant + `register`). Recon fact sheet: `.superpowers/sdd/recon/sp-b.md` §rfd3.
- Commit convention `<component>: <what>`. Before commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Migrate `rfd3` to the catalog

**Files:** Modify `src/autobio/schemas/structure_design.py` (add `RFD3Input`), `src/autobio/tools/rfd3.py`; Test `tests/unit/test_rfd3.py` (+ `tests/integration/test_rfd3_integration.py`). Read the current `src/autobio/tools/rfd3.py` in full — `prepare_workspace`'s structure-copy + `design_specs` `input`-path-rewrite logic is preserved verbatim (it already reads typed fields).

**Schema module prep** (`structure_design.py`): add `from autobio.schemas.hints import Tier, Widget, ui`. Do NOT touch `StructureDesignInput`/`DesignedStructure`/`StructureDesignOutput`.

**New class** (append):

```python
class RFD3Input(BaseInput):
    """Input for RFDiffusion3 generative structure design (single ``generate`` mode)."""

    design_specs: dict[str, dict[str, Any]] = Field(
        description=(
            "Named design specifications; each value is a dict of RFD3-native "
            "parameters. Common keys: 'input' (target structure filename, must match "
            "an input_structures file), 'contig' (which residues to keep vs design — "
            "chain+range keeps input e.g. 'A1-10', a bare integer/range designs new "
            "e.g. '120-130', '/0' is a chain break), 'length' (e.g. '100' or '80-120'), "
            "the 'select_*' fields (select_fixed_atoms/select_hotspots/select_buried/… "
            "each accept a bool, a contig string, or a {residue_id: atom_names} dict; "
            "atom keywords BKBN/ALL/TIP), 'symmetry' ({'id': 'C3'} — C/D groups), "
            "'partial_t' (noise Å, 5-15), 'ligand', 'unindex', 'infer_ori_strategy' "
            "('com'|'hotspots'), 'is_non_loopy'. See the tool notes for use-case recipes."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    input_structures: list[Path] = Field(
        default_factory=list,
        description=(
            "PDB/mmCIF files referenced by design_specs 'input' values. Each is copied "
            "into the workspace and its 'input' path is rewritten to a container path."
        ),
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=1),
    )
    n_batches: int = Field(
        default=1,
        description="Number of independent design batches per specification.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
```

(No `ge=1` on `n_batches` — the runner's `_validate_inputs` keeps its existing `AutobioError("n_batches must be at least 1, got {n}.")` check, preserving the current error type/message/test.)

**`config.json` contract (byte-compat):** `design_specs` = `copy.deepcopy(input_data.design_specs)` with each `spec["input"]` rewritten host→container via the `filename_map` built from `input_data.input_structures` (unchanged logic); `n_batches` = `input_data.n_batches`; `out_dir` = `"/workspace/outputs/raw"`; then `self._apply_extra(config, input_data)` (replaces `config.update(input_data.extra)`).

**Runner transform** (`rfd3.py`):
- Swap imports: `from autobio.core.registry import ToolCategory` (keep) + `from autobio.core.catalog import Mode, Tool, register`; import `RFD3Input` from `autobio.schemas.structure_design`.
- Delete the `TOOL_REGISTRY["rfd3"] = ToolEntry(...)` block. Drop `_RFD3_INPUT_FORMAT` (its guidance is folded into the `design_specs` description). Keep `_RFD3_NOTES`.
- `prepare_workspace`: change `assert isinstance(input_data, StructureDesignInput)` → `assert isinstance(input_data, RFD3Input)`. Keep the structure-copy + `design_specs` deep-copy + `input`-path-rewrite logic verbatim. Build the config dict as today, but end with `self._apply_extra(config, input_data)` instead of `config.update(input_data.extra)`.
- `_validate_inputs`: change its type hint to `RFD3Input`; keep all checks + `AutobioError` messages unchanged (design_specs non-empty, per-spec dict, n_batches ≥ 1, input files exist, spec `input` cross-reference).
- `parse_output`, `_resolve_container_path`: unchanged (still return `StructureDesignOutput`).

**Tool** (module bottom; reuse `_RFD3_NOTES`):

```python
RFD3_TOOL = Tool(
    name="rfd3",
    display_name="RFdiffusion3",
    category=ToolCategory.STRUCTURE_DESIGN,
    description=(
        "Generate novel protein backbone structures using RFDiffusion3. "
        "Supports unconditioned design, protein binder design, enzyme active "
        "site design, nucleic acid binder design, partial diffusion, and "
        "symmetric design. Provide design specifications via the design_specs "
        "dict — each entry is a named design job with tool-native parameters."
    ),
    version="1.0.0",
    image_tag="rfd3:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="generate",
    modes={
        "generate": Mode(
            name="generate",
            display_name="Generate designs",
            description="Generate novel backbone designs from design specifications.",
            input_schema=RFD3Input,
            output_schema=StructureDesignOutput,
            default_timeout=3600,
            supports_batch=True,
            notes=_RFD3_NOTES,
        )
    },
    keywords=("rfd3", "rfdiffusion", "structure design", "protein design", "binder", "diffusion"),
)
"""Catalog Tool for RFDiffusion3 — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(RFD3_TOOL)
```

- [ ] **Step 1:** Add `RFD3Input` (+ the `hints` import) to `structure_design.py`; run `python -m pytest tests/unit/test_schemas.py -q` (imports clean).
- [ ] **Step 2:** Migrate `rfd3.py` per the transform + contract (imports, delete flat block + `_RFD3_INPUT_FORMAT`, `RFD3Input` isinstance, `_apply_extra`, `RFD3_TOOL` + `register`).
- [ ] **Step 3:** Update `tests/unit/test_rfd3.py`: construct `RFD3Input(...)` instead of `StructureDesignInput(...)` (design_specs/input_structures/n_batches are the same top-level fields; CLI knobs like `step_scale`/`gamma_0`/`low_memory_mode` stay in `extra` — `test_extra_dict_merged` still valid). Convert the registration test to catalog assertions (`get_tool("rfd3")`, modes=={"generate"}, `default_mode=="generate"`, `modes["generate"].supports_batch is True`, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`). Add a full-dict `config.json` equality test (design_specs with a rewritten `input` path + n_batches + out_dir + a merged CLI knob). Add an `info` snapshot (`design_specs` hint `widget=="textarea"`, `output_schema` present, `notes` present — notes now render). Add an extra-shadow-rejection test (`RFD3Input(design_specs={...}, extra={"n_batches": 5})` or `extra={"out_dir": ...}` raises `AutobioError` "collide"). Keep the `_validate_inputs` message-match cases. If `tests/integration/test_rfd3_integration.py` exists, swap its inputs to `RFD3Input` (Docker-free: `--collect-only` + a host-side construction snippet). Do NOT touch `containers/` or `test_rfd3_validate_config.py`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_rfd3.py tests/unit/test_rfd3_validate_config.py tests/unit/test_registry_disjoint.py -v`; then full `python -m pytest -m "not docker and not gpu"`.
- [ ] **Step 5:** `ruff check --fix` + `ruff format` the changed files; `mypy src/`; commit `rfd3: migrate to catalog Tool with _apply_extra (generate mode)`.

---

## Final verification (after the task)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info rfd3` shows the `generate` mode with `x-autobio` hints + `output_schema` + notes; rfd3 absent from `TOOL_REGISTRY` (disjointness guard green). No `structure-design` tool other than the still-flat `complexa` remains… (rfd3 is the only flat structure_design tool; after this it's fully migrated).

## Self-Review

**1. Spec coverage:** rfd3 migrated to a single-mode (`generate`) catalog Tool; `design_specs` kept as escape-hatch dict; unfiltered `config.update(extra)` replaced with `_apply_extra`; `x-autobio` hints added; byte-compat config.json (full-dict test); flat entry removed; `StructureDesignInput` untouched (complexa still uses it); `input_format` dropped into the `design_specs` description; notes on the Mode. RunMetadata.mode still deferred (single-mode).

**2. Placeholder scan:** Full code for `RFD3Input` and `RFD3_TOOL`; runner transform specified by the exact byte-compat contract + the in-repo source + chai/boltz exemplars; no "TBD".

**3. Type consistency:** `RFD3Input` referenced consistently across schema module, runner (`isinstance` + `_validate_inputs`), Tool `input_schema`, tests. Output schema reuses `StructureDesignOutput`. `_apply_extra` unchanged.

## Next plans (Plan 4 continued)
- **mpnn family:** proteinmpnn + ligandmpnn together (shared `MPNNRunner`).
- Same-category multi-mode: rosetta (+ `RunMetadata.mode`, carry-forward #4), evoef2, complexa (complexa also frees `StructureDesignInput` for teardown removal).
- Cross-category two-class consolidation: esm_if1, antifold.
- Output-variance: openmm, antibody LMs ×6.
- Teardown: remove `TOOL_REGISTRY`/`ToolEntry`, now-unused category input schemas, hoist duplicated `_resolve_container_path` onto `ToolRunner`, wire-or-document copied-but-unwired templates/msa; README rewrite.
