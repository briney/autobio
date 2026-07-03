# Tools→Modes Plan 4a — SCORING Singleton Migrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the four `ToolCategory.SCORING` singleton tools (`baddg`, `stabddg`, `antipasti`, `prodigy`) from the legacy flat `TOOL_REGISTRY` to the `Tool`/`Mode` catalog, and strengthen the shared `_apply_extra` guard first — the first PR of the Plan 4 family-migration series.

**Architecture:** Each of these four tools is a single-purpose, single-mode tool with one runner class and one flat name equal to its Tool name. Migration follows the merged exemplars: `src/autobio/tools/esm.py` (single-mode `Tool` via a named `<TOOL>_TOOL` constant + `register(...)`; `_apply_extra`; typed fields promoted off `extra`) and `src/autobio/tools/freesasa.py` (SCORING-category `Tool` with typed fields carrying `x-autobio` hints via `ui()`, host-side `_validate_inputs` raising `AutobioError`, per-mode input-schema classes). Container-side execution is untouched; each tool's `config.json` must remain byte-for-byte identical.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy. No ML dependencies on the host.

## Global Constraints

- Python 3.11+; modern syntax; max line length **100**. Ruff lint select `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google-style docstrings; type hints on all signatures.
- Tests run with **`python -m pytest`** (bare `pytest` = wrong env). Reinstall editable (`pip install -e ".[dev]"`) if `src/` edits aren't picked up.
- **Scope is autobio core only** — NO changes to `containers/`, the workspace/`result.json` protocol, `standardize.*`, or GPU allocation.
- **Byte-compat is the success criterion:** for each tool, the `config.json` written by `prepare_workspace` must be identical (keys, values, types) to the pre-migration output. Each task ships a test asserting the exact config dict.
- **Do NOT modify `ScoringInput`** in `src/autobio/schemas/scoring.py` — it is still consumed by unmigrated tools (rosetta, openmm, evoef2, esm_if1, antifold, ligandmpnn_packer). Add NEW dedicated input classes instead.
- The catalog `Tool`/`Mode` have **no `input_format` field** — the legacy `input_format` tuples are dropped; their content is already conveyed by field `description`s. The legacy `notes` tuples move onto the `Mode`.
- `_apply_extra` (in `src/autobio/tools/base.py`) requires `self.current_mode` to be set; the base `run()` sets it before `prepare_workspace`. Catalog lookup is automatic once a Tool is registered in `CATALOG`.
- `TOOL_RUNNERS` in `src/autobio/tools/__init__.py` is keyed by tool name; since each Tool name here equals its existing flat name (`baddg`/`stabddg`/`antipasti`/`prodigy`), those entries need **no change**.
- The disjointness guard (`tests/unit/test_registry_disjoint.py`) will fail if a flat `TOOL_REGISTRY[...]` entry is left behind — each migration MUST delete the tool's `TOOL_REGISTRY[...] = ToolEntry(...)` block.
- Commit convention: `<component>: <what changed and why>`.
- Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.
- Per-tool reconnaissance fact sheets (authoritative current-state facts) live at `.superpowers/sdd/recon/scoring.md` (baddg, stabddg) and `.superpowers/sdd/recon/affinity.md` (antipasti, prodigy). Read your tool's section before implementing.

---

### Task 1: Strengthen `_apply_extra` to reject runner-derived config-key collisions

**Why:** Plan 3's final review found `_apply_extra` guards only *typed input fields*, so a caller could still pass `extra={"output_dir": ...}` (or `pdb_path`, `checkpoint_path`, `selection`, …) and silently overwrite a container-contract config value via `config.update`. All four tools in this PR write derived config keys that are NOT typed fields — landing this strengthening first means every migration below inherits the strongest guard. This is the recorded first recipe item of Plan 4 and directly serves the standing "dict pass-through needs fail-fast validation" guidance.

**Files:**
- Modify: `src/autobio/tools/base.py` (`_apply_extra`)
- Test: `tests/unit/test_tool_runner_modes.py` (update 1 match + add 1 test)
- Test: `tests/unit/test_freesasa.py`, `tests/unit/test_esm.py` (update the shadow-test `match=` strings)

**Interfaces:**
- Produces: `_apply_extra` now raises `AutobioError` if any `extra` key is a typed field on the active mode's input schema **or** already present in `config` at call time; error message names "typed input fields or runner-derived config keys".

- [ ] **Step 1: Update the base-helper tests (RED)**

In `tests/unit/test_tool_runner_modes.py`: change the existing `test_apply_extra_rejects_shadowing_typed_field` match string from `"shadow typed input fields: alpha_param"` to `"collide with typed input fields.*alpha_param"`, and add a new test after it:

```python
def test_apply_extra_rejects_derived_config_key_collision() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner.current_mode = _typed_mode()
    # "output_dir" is NOT a typed field on _TypedInput, but it is already in config.
    with pytest.raises(AutobioError, match="collide.*output_dir"):
        runner._apply_extra({"output_dir": "/x"}, _TypedInput(extra={"output_dir": "/y"}))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -k apply_extra -v`
Expected: the collision test FAILS (no rejection today), and the renamed match FAILS (message still says "shadow").

- [ ] **Step 3: Strengthen `_apply_extra`**

In `src/autobio/tools/base.py`, replace the body of `_apply_extra` (keep the signature) with:

```python
        assert self.current_mode is not None
        typed_fields = set(self.current_mode.input_schema.model_fields) - {"extra"}
        collisions = sorted(
            key for key in input_data.extra if key in typed_fields or key in config
        )
        if collisions:
            raise AutobioError(
                "extra must not contain keys that collide with typed input fields or "
                f"runner-derived config keys: {', '.join(collisions)}. Pass tool-specific "
                "parameters under new keys; set typed parameters as top-level input fields."
            )
        config.update(input_data.extra)
```

Also update the method's docstring to describe the config-key collision in addition to the typed-field shadow.

- [ ] **Step 4: Update freesasa/esm shadow-test match strings**

In `tests/unit/test_freesasa.py` (`test_extra_shadowing_typed_field_rejected`) and `tests/unit/test_esm.py` (`test_extra_shadowing_typed_field_rejected`): change `match="shadow typed input fields"` to `match="collide with typed input fields"`. (These pass a typed-field key, still rejected by the first clause.)

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py -v`
Expected: PASS (collision + shadow both rejected; passthrough of genuinely-new keys still works).

- [ ] **Step 6: Lint, type-check, commit**

```bash
ruff check --fix src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py
ruff format src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py tests/unit/test_freesasa.py tests/unit/test_esm.py
mypy src/
git add -A && git commit -m "tools: _apply_extra also rejects extra keys colliding with derived config keys"
```

---

## Single-mode flat→catalog migration recipe (Tasks 2–5)

Each of the four tools follows this recipe (the exemplars are `src/autobio/tools/esm.py` and `src/autobio/tools/freesasa.py`, merged on `main`). Per-tool tasks below give the exact new schema class(es), the exact `Tool` object, the byte-compat `config.json` contract, and the validation/tests specifics — this recipe is the shared transform:

1. **Schema:** add the tool's dedicated typed input class(es) to the schema module (full code in each task). Every promoted field carries an `x-autobio` hint via `ui(...)` from `autobio.schemas.hints`. Import `Tier, Widget, ui` in the schema module if not already imported.
2. **Runner imports:** replace `from autobio.core.registry import ToolCategory, ToolEntry` (and the `TOOL_REGISTRY` import) with `from autobio.core.registry import ToolCategory` + `from autobio.core.catalog import Mode, Tool, register`. Keep the `BaseInput` runtime import.
3. **Delete** the module-level `_CONSUMED_EXTRA_KEYS` set and the `TOOL_REGISTRY["<name>"] = ToolEntry(...)` block.
4. **`prepare_workspace`:** `assert isinstance(input_data, <ToolInput>)`; read each promoted parameter from the **typed field** (`input_data.<field>`) instead of `input_data.extra.get(...)`; build the config dict with the **exact keys/values** in the task's contract table; replace the manual extra-merge loop with `self._apply_extra(config, input_data)` as the LAST step; `workspace.write_config(config)`.
5. **`_validate_inputs`:** keep the host-side semantic checks, now reading typed fields, preserving the exact `AutobioError` messages the tests match. Drop only checks that Pydantic now enforces (e.g. "must be a list of strings" once the field is `list[str]`).
6. **`parse_output`:** unchanged.
7. **Register:** add a module-level `<NAME>_TOOL = Tool(...)` constant (full code in each task) with a triple-quoted attribute docstring, then `register(<NAME>_TOOL)` at the bottom of the module.
8. **Tests:** update the tool's `tests/unit/test_<tool>.py` so inputs pass promoted parameters as **top-level fields** (not via `extra`); keep/port the byte-compat config assertions, validation cases (same `AutobioError` matches), and `parse_output` cases; replace the registration test's `TOOL_REGISTRY`/`ToolEntry` assertions with catalog assertions (`get_tool("<name>")` returns the Tool, single expected mode, tool absent from `TOOL_REGISTRY`, still present in `TOOL_RUNNERS`). Add an `info` snapshot assertion (mirroring `test_info_snapshot_freesasa`) checking a promoted field carries its `x-autobio` hint and `output_schema` is present. The `*_e2e.py` tests should be updated to pass promoted params as typed fields; container/`standardize.py` behavior is unchanged.
9. **Verify:** run the tool's unit + e2e suites, `mypy src/`, ruff; confirm `test_registry_disjoint.py` still passes (flat entry removed). Commit.

Each task is independently reviewable and testable. Tasks 2–5 do not depend on each other (separate files); all depend on Task 1.

---

### Task 2: Migrate `baddg` to the catalog

**Files:** Modify `src/autobio/schemas/scoring.py` (add `BAddGInput`), `src/autobio/tools/baddg.py`; Test `tests/unit/test_baddg.py`, `tests/unit/test_baddg_e2e.py`. Fact sheet: `.superpowers/sdd/recon/scoring.md` (baddg section).

**New schema class** (append to `src/autobio/schemas/scoring.py`; `Tier, Widget, ui` are already imported there):

```python
class BAddGInput(BaseInput):
    """Input for BA-ddG binding-ddG prediction (single ``predict`` mode)."""

    structure_path: Path = Field(
        description="Path to the protein complex PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to score, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['YH103H', 'QD30V']); combined effect is predicted."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains: str = Field(
        description="Binding interface as 'binder1_binder2' (e.g. 'ABC_DE' = A,B,C vs D,E).",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    n_folds: int = Field(
        default=3, ge=1, le=3,
        description="Cross-validation folds to average (1-3).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    seed: int = Field(
        default=0, description="Random seed.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    device: str = Field(
        default="auto", description="Compute device ('auto', 'cpu', or 'cuda').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=12),
    )
```

**`config.json` contract (byte-compat — preserve exactly):** `pdb_path` = `f"/workspace/inputs/{structure_path.name}"` (after `shutil.copy2` into `workspace.inputs_dir`); `mutations` = `",".join(input_data.mutations)`; `chains` = `input_data.chains`; `mpnn_checkpoint_path` = `_DEFAULT_MPNN_CHECKPOINT` (existing const); `ddg_checkpoint_path` = `_DEFAULT_DDG_CHECKPOINT` (existing const); `output_dir` = `"/workspace/outputs/raw"`; `n_folds` = `input_data.n_folds`; `seed` = `input_data.seed`; `device` = `input_data.device`; then `self._apply_extra(config, input_data)`.

**Validation to preserve** (`_validate_inputs`, reading typed fields, same messages): structure file exists (`"Input structure file does not exist: {path}"`); `mutations` non-empty (`"BA-ddG requires 'mutations' ..."`); `chains` has exactly one `_` (`"'chains' must be a string with exactly one underscore separator (e.g., 'ABC_DE'), got {chains!r}."`). Drop the now-redundant "must be a list of strings" check (Pydantic enforces `list[str]`).

**Tool object** (module bottom; `_BADDG_NOTES` already defined in the file — reuse it; drop `_BADDG_INPUT_FORMAT`):

```python
BADDG_TOOL = Tool(
    name="baddg",
    display_name="BA-ddG",
    category=ToolCategory.SCORING,
    description=(
        "Predict binding ddG at protein-protein interfaces using BA-ddG, a "
        "Boltzmann-aligned inverse folding model. Returns ddG in kcal/mol."
    ),
    version="1.0.0",
    image_tag="baddg:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict ddG",
            description="Predict binding ddG for mutations in a protein complex.",
            input_schema=BAddGInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_BADDG_NOTES,
        )
    },
    keywords=("baddg", "ddg", "binding affinity", "mutation", "interface"),
)
"""Catalog Tool for BA-ddG — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(BADDG_TOOL)
```

- [ ] **Step 1:** Add `BAddGInput` to `scoring.py`; run `python -m pytest tests/unit/test_schemas.py -q` (schema imports clean).
- [ ] **Step 2:** Migrate `baddg.py` per the recipe (imports, delete `_CONSUMED_EXTRA_KEYS` + `TOOL_REGISTRY` block, typed-field `prepare_workspace` with the contract above, `_apply_extra`, `_validate_inputs` on typed fields, `BADDG_TOOL` + `register`).
- [ ] **Step 3:** Update `tests/unit/test_baddg.py` + `tests/unit/test_baddg_e2e.py`: pass `mutations`/`chains`/`n_folds`/`seed`/`device` as top-level `BAddGInput` fields; keep the byte-compat config assertions (assert the exact contract keys/values); registration test → catalog (`get_tool("baddg")`, modes `{"predict"}`, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`); add an `info` snapshot assertion (`structure_path` hint `widget=="file"`, `output_schema` present). Keep validation-case matches.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_baddg.py tests/unit/test_baddg_e2e.py tests/unit/test_registry_disjoint.py -v`; expected PASS.
- [ ] **Step 5:** `ruff check --fix` + `ruff format` the changed files; `mypy src/`; commit `baddg: migrate to catalog Tool with typed fields (predict mode)`.

---

### Task 3: Migrate `stabddg` to the catalog

**Files:** Modify `src/autobio/schemas/scoring.py` (add `StaBddGInput`), `src/autobio/tools/stabddg.py`; Test `tests/unit/test_stabddg.py`, `tests/unit/test_stabddg_e2e.py`. Fact sheet: `.superpowers/sdd/recon/scoring.md` (stabddg section).

**New schema class** (append to `scoring.py`):

```python
class StaBddGInput(BaseInput):
    """Input for StaB-ddG binding-ddG prediction (single ``predict`` mode)."""

    structure_path: Path = Field(
        description="Path to the protein complex PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to score, StaB-ddG format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['YH103H', 'QD30V'])."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains: str = Field(
        description="Binding interface as 'binder1_binder2' (e.g. 'ABC_DE').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    mc_samples: int = Field(
        default=20, ge=1,
        description="Monte-Carlo samples for variance reduction.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    noise_level: float = Field(
        default=0.1,
        description="Backbone perturbation noise level.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, step=0.05, order=11),
    )
    batch_size: int = Field(
        default=10000, ge=1,
        description="Batch size for scoring.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=12),
    )
    trials: int = Field(
        default=1, ge=1,
        description="Number of independent prediction trials.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=13),
    )
    seed: int = Field(
        default=0, description="Random seed.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=14),
    )
    device: str = Field(
        default="auto", description="Compute device ('auto', 'cpu', or 'cuda').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=15),
    )
```

**`config.json` contract (byte-compat):** `pdb_path` = `f"/workspace/inputs/{structure_path.name}"`; `mutations` = `",".join(input_data.mutations)`; `chains` = `input_data.chains`; `checkpoint_path` = `_DEFAULT_CHECKPOINT` (existing const); `output_dir` = `"/workspace/outputs/raw"`; `mc_samples` = `input_data.mc_samples`; `noise_level` = `input_data.noise_level`; `batch_size` = `input_data.batch_size`; `trials` = `input_data.trials`; `seed` = `input_data.seed`; `device` = `input_data.device`; then `self._apply_extra(config, input_data)`.

**Validation to preserve:** structure exists; `mutations` non-empty (`"StaB-ddG requires 'mutations' ..."`); `chains` exactly one `_` (same message as baddg). Drop the "must be a list of strings" check.

**Tool object** (`_STABDDG_NOTES` reused; drop `_STABDDG_INPUT_FORMAT`):

```python
STABDDG_TOOL = Tool(
    name="stabddg",
    display_name="StaB-ddG",
    category=ToolCategory.SCORING,
    description=(
        "Predict binding ddG at protein-protein interfaces using StaB-ddG, a "
        "ProteinMPNN-based ML method. Returns ddG in kcal/mol."
    ),
    version="1.0.0",
    image_tag="stabddg:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict ddG",
            description="Predict binding ddG for mutations in a protein complex.",
            input_schema=StaBddGInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_STABDDG_NOTES,
        )
    },
    keywords=("stabddg", "ddg", "binding affinity", "mutation", "interface"),
)
"""Catalog Tool for StaB-ddG — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(STABDDG_TOOL)
```

- [ ] **Step 1:** Add `StaBddGInput` to `scoring.py`; import check.
- [ ] **Step 2:** Migrate `stabddg.py` per the recipe + contract above.
- [ ] **Step 3:** Update `tests/unit/test_stabddg.py` + `tests/unit/test_stabddg_e2e.py` (typed-field inputs; byte-compat config assertions incl. all 6 numeric/string params; catalog registration test; `info` snapshot assertion).
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_stabddg.py tests/unit/test_stabddg_e2e.py tests/unit/test_registry_disjoint.py -v`; expected PASS.
- [ ] **Step 5:** ruff + mypy; commit `stabddg: migrate to catalog Tool with typed fields (predict mode)`.

---

### Task 4: Migrate `antipasti` to the catalog

**Files:** Modify `src/autobio/schemas/binding_affinity.py` (hints on `BindingAffinityInput`; add `AntipastiInput`), `src/autobio/tools/antipasti.py`; Test `tests/unit/test_antipasti.py`, `tests/unit/test_antipasti_e2e.py`. Fact sheet: `.superpowers/sdd/recon/affinity.md` (antipasti section).

**Schema changes** (`src/autobio/schemas/binding_affinity.py`): add `from autobio.schemas.hints import Tier, Widget, ui` at the top; add `json_schema_extra` hints to the four existing `BindingAffinityInput` fields — `structure_path` → `ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0)`, `heavy_chain` → `ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1)`, `light_chain` → `ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2)`, `antigen_chains` → `ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=3)` (leave descriptions/types unchanged). Then add:

```python
class AntipastiInput(BindingAffinityInput):
    """Input for ANTIPASTI antibody-antigen affinity prediction."""

    modes: str | int = Field(
        default="all",
        description="Normal modes for the DCCM calculation: 'all', or an integer count.",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=10),
    )
```

**`config.json` contract (byte-compat):** `pdb_path` = `f"/workspace/inputs/{structure_path.name}"`; `heavy_chain` = `input_data.heavy_chain`; `light_chain` = `input_data.light_chain`; `antigen_chains` = `input_data.antigen_chains`; `checkpoint_path` = `_DEFAULT_CHECKPOINT` (existing const); `output_dir` = `"/workspace/outputs/raw"`; `antipasti_dir` = `_ANTIPASTI_DIR` (existing const); `modes` = `input_data.modes`; then `self._apply_extra(config, input_data)`.

**Validation to preserve** (`_validate_inputs`, exact messages): structure exists; `heavy_chain` non-empty (`"heavy_chain must be a non-empty string."`); `light_chain` non-empty; `antigen_chains` non-empty list (`"antigen_chains must be a non-empty list of chain IDs ..."`); each antigen chain non-empty (`"Each antigen chain ID must be a non-empty string."`); cross-field dedup (`"Duplicate chain IDs detected: {all_chains}. ..."`).

**Tool object** (`_ANTIPASTI_NOTES` reused; drop `_ANTIPASTI_INPUT_FORMAT`):

```python
ANTIPASTI_TOOL = Tool(
    name="antipasti",
    display_name="ANTIPASTI",
    category=ToolCategory.SCORING,
    description=(
        "Predict antibody-antigen binding affinity (log10 Kd) from a 3D PDB complex "
        "using ANTIPASTI (Normal Mode Correlation Maps + CNN). CPU-only."
    ),
    version="1.0.0",
    image_tag="antipasti:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict affinity",
            description="Predict antibody-antigen binding affinity (log10 Kd).",
            input_schema=AntipastiInput,
            output_schema=BindingAffinityOutput,
            default_timeout=1800,
            notes=_ANTIPASTI_NOTES,
        )
    },
    keywords=("antipasti", "binding affinity", "antibody", "antigen", "kd"),
)
"""Catalog Tool for ANTIPASTI — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(ANTIPASTI_TOOL)
```

- [ ] **Step 1:** Edit `binding_affinity.py` (hints + `AntipastiInput`); import check.
- [ ] **Step 2:** Migrate `antipasti.py` per the recipe + contract. `modes` moves from `extra.get("modes", "all")` to the typed field default `"all"`.
- [ ] **Step 3:** Update `tests/unit/test_antipasti.py` + `_e2e.py`: `modes` passed as a top-level field (test both default `"all"` and an int like `100`); byte-compat config assertions (incl. `checkpoint_path`, `antipasti_dir`, `modes`); catalog registration test; `info` snapshot (`structure_path` hint `widget=="file"`; `output_schema` present). Keep the chain-validation matches.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_antipasti.py tests/unit/test_antipasti_e2e.py tests/unit/test_registry_disjoint.py -v`; expected PASS.
- [ ] **Step 5:** ruff + mypy; commit `antipasti: migrate to catalog Tool with typed fields (predict mode)`.

---

### Task 5: Migrate `prodigy` to the catalog

**Files:** Modify `src/autobio/schemas/protein_binding_affinity.py` (hints + add `ProdigyInput`), `src/autobio/tools/prodigy.py`; Test `tests/unit/test_prodigy.py`, `tests/unit/test_prodigy_e2e.py`. Fact sheet: `.superpowers/sdd/recon/affinity.md` (prodigy section).

**Schema changes** (`src/autobio/schemas/protein_binding_affinity.py`): add `from autobio.schemas.hints import Tier, Widget, ui`; add hints to the existing `ProteinBindingAffinityInput` fields — `structure_path` → `ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0)`, `chain_selection` → `ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1)`, `temperature` → `ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="°C", order=10)`. Then add:

```python
class ProdigyInput(ProteinBindingAffinityInput):
    """Input for PRODIGY protein-protein affinity prediction."""

    distance_cutoff: float = Field(
        default=5.5, gt=0,
        description="Interatomic contact distance cutoff.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="Å", step=0.1, order=11),
    )
    contact_list: bool = Field(
        default=False,
        description="Include the detailed interface contact list in the output breakdown.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=12),
    )
```

**`config.json` contract (byte-compat — note the `chain_selection`→`selection` remap):** `structure_path` = `f"/workspace/inputs/{structure_path.name}"`; `selection` = `input_data.chain_selection` (may be `None`); `temperature` = `input_data.temperature`; `distance_cutoff` = `input_data.distance_cutoff`; `contact_list` = `input_data.contact_list`; `output_dir` = `"/workspace/outputs/raw"`; then `self._apply_extra(config, input_data)`.

**Validation to preserve:** structure exists; `temperature > -273.15` (`"Temperature must be above absolute zero (-273.15 °C), got {temperature}."`); `chain_selection` if not None must be non-empty after strip (`"chain_selection must be a non-empty string or None."`).

**Tool object** (`_PRODIGY_NOTES` reused; drop `_PRODIGY_INPUT_FORMAT`):

```python
PRODIGY_TOOL = Tool(
    name="prodigy",
    display_name="PRODIGY",
    category=ToolCategory.SCORING,
    description=(
        "Predict protein-protein binding affinity (delta-G and Kd) from a 3D complex "
        "using PRODIGY (interatomic contact counting). CPU-only."
    ),
    version="2.4.0",
    image_tag="prodigy:2.4.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict affinity",
            description="Predict protein-protein binding affinity (delta-G, Kd).",
            input_schema=ProdigyInput,
            output_schema=ProteinBindingAffinityOutput,
            default_timeout=300,
            notes=_PRODIGY_NOTES,
        )
    },
    keywords=("prodigy", "binding affinity", "protein-protein", "delta-g", "kd"),
)
"""Catalog Tool for PRODIGY — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(PRODIGY_TOOL)
```

- [ ] **Step 1:** Edit `protein_binding_affinity.py` (hints + `ProdigyInput`); import check.
- [ ] **Step 2:** Migrate `prodigy.py` per the recipe + contract. `distance_cutoff`/`contact_list` move from `extra.get(...)` to typed field defaults; **preserve the `selection` config key name** (not `chain_selection`).
- [ ] **Step 3:** Update `tests/unit/test_prodigy.py` + `_e2e.py`: `distance_cutoff`/`contact_list` as top-level fields; byte-compat config assertions (assert `cfg["selection"]`, default `temperature`/`distance_cutoff`/`contact_list`); catalog registration test; `info` snapshot (`structure_path` hint `widget=="file"`; `contact_list` hint `widget=="toggle"`; `output_schema` present). Keep temperature/chain_selection validation matches.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_prodigy.py tests/unit/test_prodigy_e2e.py tests/unit/test_registry_disjoint.py -v`; expected PASS.
- [ ] **Step 5:** ruff + mypy; commit `prodigy: migrate to catalog Tool with typed fields (predict mode)`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check src/ tests/` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio list` shows baddg/stabddg/antipasti/prodigy as catalog Tools; `autobio info <tool>` returns the `predict` mode with `x-autobio` hints + `output_schema`; each is absent from `TOOL_REGISTRY` (disjointness guard green).

## Self-Review

**1. Spec coverage:** Task 1 = Plan 3's carry-forward derived-key guard. Tasks 2–5 migrate the four SCORING singletons (baddg, stabddg, antipasti, prodigy) per the design spec (Tools→Modes, typed fields + `x-autobio`, byte-compat config, flat entry removed). SequenceSet is N/A (none of these take raw sequences — structure paths + chain IDs only). RunMetadata.mode (carry-forward #4) is deferred to a genuinely multi-mode engine (rosetta/openmm), not these single-mode tools.

**2. Placeholder scan:** Full code for Task 1 and for every new schema class + `Tool` object; runner transforms specified by exact byte-compat config contracts + the in-repo esm/freesasa exemplars + fact sheets; no "TBD"/"handle edge cases".

**3. Type consistency:** New input classes (`BAddGInput`, `StaBddGInput`, `AntipastiInput`, `ProdigyInput`) are referenced consistently in their schema module, runner, Tool `input_schema`, and tests. Output schemas reuse existing classes (`ScoringOutput`, `BindingAffinityOutput`, `ProteinBindingAffinityOutput`). `_apply_extra` signature is unchanged by Task 1 (behavior strengthened).

## Next plans (Plan 4 continued)

- **Plan 4b — structure-prediction singletons:** esmfold, boltz1, boltz2 (boltz1/boltz2 = two Tools sharing `BoltzRunner`, esm1b/esm2-style; keep `sequences` as `dict[str,str]` — boltz accepts DNA/RNA/SMILES, so `GenericSequenceSet` is unsafe).
- **Plan 4c — complex structure-prediction singletons:** chai1, openfold3 (structural non-scalar extras → typed dict/list fields with `widget: json`; preserve pre-existing behavior incl. known footguns — do not fix bugs in the migration, file them separately).
- **Plan 4d — rfd3 (structure_design):** keep `design_specs` as an escape-hatch dict (container-validated mini-language); adopt `_apply_extra`; single mode.
- **Later groups:** same-category multi-mode (rosetta [+ `RunMetadata.mode`], evoef2, complexa); cross-category + two-class consolidation (esm_if1, antifold); mpnn family (proteinmpnn + ligandmpnn together — shared `MPNNRunner`); output-variance (openmm, antibody LMs ×6). Then teardown/README (final plan).
