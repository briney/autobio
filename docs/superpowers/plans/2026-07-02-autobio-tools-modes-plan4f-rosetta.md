# Tools→Modes Plan 4f — RunMetadata.mode + rosetta Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the deferred `RunMetadata.mode` field (Plan-2 carry-forward #4), then migrate `rosetta` — the first genuinely multi-mode engine — from 4 flat `TOOL_REGISTRY` entries to ONE catalog `Tool` with 4 `Mode`s (`score`/`relax`/`minimize`/`flexddg`), each a separate Docker image.

**Architecture:** Task 1 threads `self.current_mode.name` into `RunMetadata.mode` via the base runner (needed because collapsing `rosetta_flexddg` → `rosetta` + `mode="flexddg"` would otherwise lose the mode signal from metadata). Task 2 is the multi-mode migration (freesasa pattern, scaled up): one `RosettaRunner` dispatching on `self.current_mode.name` via a mode-keyed config table + per-mode input subclasses, per-mode `Mode.image_tag` (4 distinct images) and per-mode `Mode.default_timeout` (600/3600/1800/14400s). `score_function` and `nstruct` (always-written today) promote to typed fields; flexddg adds `mutations`/`chains_to_move`/`resfile`; flexddg power-user knobs (`backrub_trials`, `max_minimization_iter`) stay in `extra`. Container-side execution is untouched; each mode's `config.json` is byte-for-byte preserved.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env).
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation.
- **Byte-compat is the success criterion:** each rosetta mode's `config.json` (keys, values, ORDER) must be identical to the pre-migration flat tool's output. Ship a full-dict `config.json` equality test per mode.
- **Do NOT modify `ScoringInput`** (many consumers) — add new `RosettaBaseInput`/`RosettaRelaxInput`/`RosettaFlexDdgInput` to `scoring.py`.
- Catalog `Mode` supports per-mode `image_tag` (override) and `default_timeout` — use them (rosetta's 4 images + 4 timeouts). `Tool.image_tag` is a required nominal fallback; set it to the default mode's tag (`rosetta-score:1.0.0`) but give EVERY mode an explicit `image_tag` so each resolves distinctly (an existing test asserts 4 distinct image tags).
- Drop the legacy `input_format` tuples; fold essential guidance into field descriptions. `notes` move onto the per-mode `Mode` (rendered by `info`).
- Adopt `self._apply_extra(config, input_data)` (replacing the `_CONSUMED_EXTRA_KEYS` filter loop). Delete `_CONSUMED_EXTRA_KEYS`.
- Delete the 4 `TOOL_REGISTRY[...]` blocks (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS`: replace the 4 flat-name entries (`rosetta_score`/`relax`/`minimize`/`flexddg`) with a single `"rosetta": RosettaRunner` entry (Tool name == the new single flat name). **This is the first migration that changes `TOOL_RUNNERS` keys** (prior migrations kept name==name); the 4 old keys go away and `"rosetta"` is added.
- Merged exemplar: `src/autobio/tools/freesasa.py` (multi-mode Tool, per-mode input subclasses, dispatch on `self.current_mode.name`, `_apply_extra`). Recon fact sheet: `.superpowers/sdd/recon/rosetta.md`.
- Commit convention `<component>: <what>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Add `RunMetadata.mode`

**Why:** Plan-2 carry-forward #4, deferred until the first genuinely multi-mode engine. Rosetta's flat `tool_name` (`"rosetta_flexddg"`) currently carries the mode; collapsing to one `"rosetta"` Tool loses that from `RunMetadata` unless a `mode` field is added. Additive (`default None`), so all existing `RunMetadata` constructions keep working.

**Files:** Modify `src/autobio/schemas/base.py` (`RunMetadata`), `src/autobio/tools/base.py` (`_build_metadata`); Test `tests/unit/test_tool_runner_modes.py`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_tool_runner_modes.py`, extend `test_run_sets_current_mode_and_mode_metadata` (it already runs `faketool` mode `"beta"`): add `assert out.metadata.mode == "beta"`. And add a legacy-tool assertion — in `test_run_rejects_mode_for_legacy_tool` (or a small new test), confirm that a legacy tool's metadata `mode` is `None` (legacy tools never set `current_mode`). If the legacy test doesn't produce a successful run, add:

```python
def test_metadata_mode_none_for_legacy_run(tmp_path, monkeypatch) -> None:
    import autobio.tools  # noqa: F401 - populate registries
    monkeypatch.setattr(
        "autobio.core.workspace.Workspace.read_result",
        lambda self: SimpleNamespace(status="success", phase="run", exit_code=0, error_message=None),
    )
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = _CaptureRunner("prodigy", AutobioConfig.resolve())
    out = runner.run(_Input(), gpu="none", output_dir=tmp_path)
    assert out.metadata.mode is None
```

(`prodigy` is a migrated catalog Tool now — if `_CaptureRunner("prodigy", ...)` resolves a catalog Tool, `current_mode` WILL be set and mode won't be None. Use a genuinely legacy tool if any remain, else construct a `_CaptureRunner` whose `self.tool`/`self.current_mode` are None — i.e. leave `current_mode=None` and assert `_build_metadata(...).mode is None` directly. Simplest robust form:)

```python
def test_build_metadata_mode_none_when_no_current_mode(tmp_path) -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    # current_mode not set (None)
    md = runner._build_metadata_public_shim  # see note
```

Prefer the direct form: after `_register_faketool()` + `_make_runner("faketool")` with `runner.current_mode = None`, call `runner._build_metadata(<workspace>, 0.0, [], "")` and assert `.mode is None`; then set `runner.current_mode = runner._resolve_mode("beta")` and assert `_build_metadata(...).mode == "beta"`. Use a real `Workspace.create(tmp_path)` for the workspace arg.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -k "mode_metadata or metadata_mode" -v`
Expected: FAIL (`RunMetadata` has no `mode` attribute / AttributeError).

- [ ] **Step 3: Add the field**

In `src/autobio/schemas/base.py`, add to `RunMetadata` (right after `tool_name`):

```python
    mode: str | None = None
```

- [ ] **Step 4: Thread it through `_build_metadata`**

In `src/autobio/tools/base.py`, in `_build_metadata`, add to the `RunMetadata(...)` call:

```python
            mode=self.current_mode.name if self.current_mode is not None else None,
```

- [ ] **Step 5: Run to verify pass + full suite**

Run: `python -m pytest tests/unit/test_tool_runner_modes.py -v`, then `python -m pytest -m "not docker and not gpu"` (existing metadata/`test_result`/`test_schemas_base` tests must stay green — the field is additive with a default), then `mypy src/`, ruff.

- [ ] **Step 6: Commit**

```bash
git add src/autobio/schemas/base.py src/autobio/tools/base.py tests/unit/test_tool_runner_modes.py
git commit -m "core: add RunMetadata.mode, set from current mode in run() (carry-forward #4)"
```

---

### Task 2: Migrate `rosetta` to a 4-mode catalog Tool

**Files:** Modify `src/autobio/schemas/scoring.py` (3 new input classes), `src/autobio/tools/rosetta.py`, `src/autobio/tools/__init__.py` (`TOOL_RUNNERS`); Test `tests/unit/test_rosetta.py`, `tests/unit/test_rosetta_e2e.py` (+ integration if present). Read the current `src/autobio/tools/rosetta.py` in full. Fact sheet: `.superpowers/sdd/recon/rosetta.md`. `Any`/`Literal`/`Tier`/`Widget`/`ui` are already imported in `scoring.py`.

**New schema classes** (append to `scoring.py`):

```python
_ROSETTA_SCORE_FUNCTIONS = Literal[
    "ref2015", "ref2015_cart", "beta_nov16", "score12", "talaris2014", "franklin2019"
]


class RosettaBaseInput(BaseInput):
    """Shared input for Rosetta score/minimize modes."""

    structure_path: Path = Field(
        description="Path to the input structure (PDB or mmCIF).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    score_function: _ROSETTA_SCORE_FUNCTIONS = Field(
        default="ref2015",
        description="Rosetta energy score function.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.ADVANCED, order=10),
    )
    nstruct: int = Field(
        default=1, ge=1,
        description="Number of output structures to generate.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )


class RosettaRelaxInput(RosettaBaseInput):
    """Input for Rosetta relax mode (FastRelax; higher nstruct default)."""

    nstruct: int = Field(
        default=5, ge=1,
        description="Number of relaxed structures to generate.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )


class RosettaFlexDdgInput(RosettaBaseInput):
    """Input for Rosetta flex-ddG interface-mutation DDG mode."""

    mutations: list[str] = Field(
        description=(
            "Mutations to score, e.g. ['A42F'] (original-residue-number-new; "
            "'A:42:F' for multi-letter chains)."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains_to_move: str = Field(
        description="Chain ID(s) of the binding partner to perturb at the interface (e.g. 'B').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    nstruct: int = Field(
        default=35, ge=1,
        description="Number of independent backrub samples (use 3 for quick tests).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    resfile: str | None = Field(
        default=None,
        description="Raw Rosetta resfile content (power-user override of the generated mutation list).",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
```

**Runner mode-config table** (`rosetta.py`, replacing `_VARIANT_CONFIG`, keyed by MODE name):

```python
_MODE_CONFIG: dict[str, dict[str, str]] = {
    "score": {"binary": "score_jd2", "protocol": "score"},
    "relax": {"binary": "rosetta_scripts", "protocol": "relax", "xml_path": "/opt/tool/xml/relax.xml"},
    "minimize": {
        "binary": "rosetta_scripts", "protocol": "minimize", "xml_path": "/opt/tool/xml/minimize.xml"
    },
    "flexddg": {"binary": "rosetta_scripts", "protocol": "flexddg"},
}
```

**`config.json` contract per mode (byte-compat — exact keys IN ORDER):**
- **All modes** start: `binary` = `_MODE_CONFIG[mode]["binary"]`; `protocol` = `_MODE_CONFIG[mode]["protocol"]`; `structure_path` = `/workspace/inputs/{name}`; `database_path` = `_ROSETTA_DB`; `score_function` = `input_data.score_function`; `out_dir` = `"/workspace/outputs/raw"`; then `nstruct` = `input_data.nstruct`.
- **relax, minimize** then add `xml_path` = `_MODE_CONFIG[mode]["xml_path"]`.
- **flexddg** then adds (via the preserved mutation logic): `mutations` = `input_data.mutations`; `chains_to_move` = `input_data.chains_to_move`; then if `input_data.resfile`: write it to `inputs/mutations.resfile` and set `resfile_path` = `"/workspace/inputs/mutations.resfile"`, ELSE set `mutation_list` = `input_data.mutations`.
- **All modes** end with `self._apply_extra(config, input_data)` (merges remaining `extra` — e.g. flexddg's `backrub_trials`/`max_minimization_iter`, score's `ex1`/`ex2`).

Note: `score`/`minimize` share `RosettaBaseInput` (nstruct default 1); `relax` uses `RosettaRelaxInput` (nstruct 5); `flexddg` uses `RosettaFlexDdgInput` (nstruct 35). All promoted fields (`score_function`, `nstruct`, and flexddg's `mutations`/`chains_to_move`) were ALWAYS written to config in the flat tools, so always-writing them is byte-compat. `backrub_trials`/`max_minimization_iter` stay in `extra` (only written when passed — byte-compat). The `chains` alias for `chains_to_move` is DROPPED (clean schema break; pass `chains_to_move`).

**Runner transform** (`rosetta.py`):
- Swap imports: keep `ToolCategory`; add `from autobio.core.catalog import Mode, Tool, register`; import the 3 input classes from `autobio.schemas.scoring`. Delete the 4 `TOOL_REGISTRY[...]` blocks, `_VARIANT_CONFIG`, `_CONSUMED_EXTRA_KEYS`. Drop the `_*_INPUT_FORMAT` tuples. Keep `_ROSETTA_DB`, `_MUTATION_PATTERN_HELP`, the four `_*_NOTES`.
- `prepare_workspace`: `assert isinstance(input_data, RosettaBaseInput)`; `assert self.current_mode is not None`; `mode = self.current_mode.name`; `mode_cfg = _MODE_CONFIG[mode]`; validate; copy structure; build config per the contract (reading `input_data.score_function`/`input_data.nstruct`); `if "xml_path" in mode_cfg: config["xml_path"] = mode_cfg["xml_path"]`; `if mode == "flexddg": self._prepare_mutations(input_data, workspace, config)`; `self._apply_extra(config, input_data)`.
- `_prepare_mutations`: retype the param to `RosettaFlexDdgInput` (or assert isinstance inside); read `input_data.mutations`/`input_data.chains_to_move`/`input_data.resfile` (typed fields, not `extra`); preserve the `resfile_path` vs `mutation_list` either/or and the exact config keys.
- `_validate_inputs`: keep the structure-exists check (all modes); for `mode == "flexddg"`, keep an empty-`mutations` check (reworded accurately: `f"Rosetta flex-ddG requires at least one mutation. {_MUTATION_PATTERN_HELP}"`) and an empty-`chains_to_move` check (message unchanged in spirit). Drop the "must be a list of strings" check (Pydantic enforces). Gate the chains check on `mode == "flexddg"` (not on a `requires_mutations` flag). `mutations`/`chains_to_move` being required typed fields means Pydantic already rejects their absence.
- `parse_output`, `_resolve_container_path`: unchanged.

**Tool** (module bottom; reuse the four `_*_NOTES`):

```python
ROSETTA_TOOL = Tool(
    name="rosetta",
    display_name="Rosetta",
    category=ToolCategory.SCORING,
    description=(
        "Rosetta structure scoring, refinement, and interface DDG prediction. Modes: "
        "score (energy), relax (FastRelax), minimize (gradient minimization), and "
        "flexddg (flex-ddG interface mutation DDG)."
    ),
    version="1.0.0",
    image_tag="rosetta-score:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="score",
    modes={
        "score": Mode(
            name="score", display_name="Score",
            description="Score a structure with Rosetta's energy function.",
            input_schema=RosettaBaseInput, output_schema=ScoringOutput,
            default_timeout=600, image_tag="rosetta-score:1.0.0", notes=_SCORE_NOTES,
        ),
        "relax": Mode(
            name="relax", display_name="Relax",
            description="FastRelax a structure to a low-energy conformation.",
            input_schema=RosettaRelaxInput, output_schema=ScoringOutput,
            default_timeout=3600, image_tag="rosetta-relax:1.0.0", notes=_RELAX_NOTES,
        ),
        "minimize": Mode(
            name="minimize", display_name="Minimize",
            description="Gradient-based energy minimization of a structure.",
            input_schema=RosettaBaseInput, output_schema=ScoringOutput,
            default_timeout=1800, image_tag="rosetta-minimize:1.0.0", notes=_MINIMIZE_NOTES,
        ),
        "flexddg": Mode(
            name="flexddg", display_name="Flex-ddG",
            description="Predict interface binding DDG with backrub sampling.",
            input_schema=RosettaFlexDdgInput, output_schema=ScoringOutput,
            default_timeout=14400, image_tag="rosetta-flexddg:1.0.0", notes=_FLEXDDG_NOTES,
        ),
    },
    keywords=("rosetta", "scoring", "relax", "minimize", "flexddg", "ddg", "energy"),
)
"""Catalog Tool for Rosetta (score/relax/minimize/flexddg modes)."""

register(ROSETTA_TOOL)
```

**`TOOL_RUNNERS`** (`src/autobio/tools/__init__.py`): remove the four `"rosetta_score"/"rosetta_relax"/"rosetta_minimize"/"rosetta_flexddg": RosettaRunner` entries; add a single `"rosetta": RosettaRunner`.

- [ ] **Step 1:** Add the 3 input classes to `scoring.py`; `python -m pytest tests/unit/test_schemas.py -q`.
- [ ] **Step 2:** Migrate `rosetta.py` per the transform + per-mode contract; update `TOOL_RUNNERS` in `tools/__init__.py`.
- [ ] **Step 3:** Update `tests/unit/test_rosetta.py` + `test_rosetta_e2e.py`: parametrize over `(mode)` instead of the 4 flat names; construct the per-mode input classes with typed fields (score_function/nstruct as fields; flexddg mutations/chains_to_move/resfile as fields; backrub_trials etc. stay in `extra`); catalog registration test (`get_tool("rosetta")`, modes=={"score","relax","minimize","flexddg"}, `default_mode=="score"`, all 4 flat names absent from `TOOL_REGISTRY`, `"rosetta"` in `TOOL_RUNNERS` and the 4 old keys gone); replace `test_each_tool_has_unique_image_tag` with an assertion that the 4 modes' resolved `image_tag`s are 4 distinct values (`rosetta-{mode}:1.0.0`); a full-dict `config.json` equality test PER mode (score/relax/minimize/flexddg — pin key order incl. xml_path for relax/minimize and the flexddg mutation keys + resfile-vs-mutation_list branch); an `info` snapshot (per-mode notes render; a promoted hint; `output_schema`); assert `run(...).metadata.mode` equals the mode; an extra-shadow-rejection test (`score_function`/`nstruct` or a config key via `extra` raises). Update the `e2e` per-mode classes' input construction. If `tests/integration/test_rosetta_integration.py` exists, swap inputs. Do NOT touch `containers/` or `test_rosetta_standardize.py`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_rosetta.py tests/unit/test_rosetta_e2e.py tests/unit/test_rosetta_standardize.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `rosetta: migrate to catalog Tool with 4 modes (per-mode image/timeout, flexddg typed fields)`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info rosetta` shows 4 modes with per-mode notes + `x-autobio` hints + `output_schema`; `run` metadata carries `mode`; the 4 flat `rosetta_*` names are gone from `TOOL_REGISTRY` and `TOOL_RUNNERS` (only `"rosetta"` remains); disjointness guard green.

## Self-Review

**1. Spec coverage:** `RunMetadata.mode` added + threaded (carry-forward #4). rosetta → one Tool, 4 modes, per-mode `image_tag` + `default_timeout`; per-mode input subclasses (base for score/minimize, relax + flexddg subclasses); `score_function`/`nstruct` promoted (byte-compat, always-written); flexddg `mutations`/`chains_to_move`/`resfile` typed; power-user knobs stay in `extra` via `_apply_extra`; `ScoringInput` untouched; flat entries + `input_format` removed; `TOOL_RUNNERS` collapsed to `"rosetta"`; byte-compat full-dict config test per mode.

**2. Placeholder scan:** Full code for `RunMetadata.mode`, the 3 input classes, and `ROSETTA_TOOL`; runner transform specified by exact per-mode byte-compat contracts + in-repo source + freesasa exemplar; no "TBD".

**3. Type consistency:** `RosettaBaseInput`/`RosettaRelaxInput`/`RosettaFlexDdgInput` referenced consistently across schema, runner (`isinstance` + `_prepare_mutations`), Tool mode `input_schema`, tests. `_apply_extra` unchanged. `RunMetadata.mode` default None keeps all existing constructions valid.

## Next plans (Plan 4 continued)
- **evoef2 + complexa** (next PR, 4g): evoef2 (repair/binding/build_mutant — per-mode subclasses, freesasa style, same image); complexa (protein_binder/ligand_binder/ame — one shared `ComplexaInput` with `design_specs` escape-hatch, per-mode variant/ckpt lookup; frees `StructureDesignInput` for teardown). Recon fact sheets already at `.superpowers/sdd/recon/evoef2.md` + `complexa.md`.
- Then: two-class consolidation (esm_if1, antifold — one Tool w/ {design,score} modes each); output-variance (openmm, antibody LMs ×6); teardown (remove `TOOL_REGISTRY`/`ToolEntry` + now-unused category schemas incl. `StructureDesignInput`; hoist duplicated `_resolve_container_path`; wire-or-document copied-but-unwired templates/msa; dead-key cleanups) + README rewrite.
