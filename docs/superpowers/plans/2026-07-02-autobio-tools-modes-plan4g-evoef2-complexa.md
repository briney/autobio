# Tools→Modes Plan 4g — evoef2 + complexa Migrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the remaining same-category multi-mode engines — `evoef2` (3 modes) and `complexa` (3 modes) — from flat `TOOL_REGISTRY` entries to catalog `Tool`s, following the proven rosetta multi-mode pattern.

**Architecture:** Both collapse N flat names → one `Tool` with N `Mode`s, one runner dispatching on `self.current_mode.name`, `TOOL_RUNNERS` keys collapsed. evoef2 uses **per-mode input subclasses** (disjoint fields: repair/binding/build_mutant) off a shared `EvoEF2BaseInput` (freesasa pattern), same image + uniform 600s timeout. complexa uses **one shared `ComplexaInput`** (design_specs escape-hatch, rfd3 pattern) with a mode-keyed checkpoint lookup table, same image + uniform 43200s timeout. Both adopt `self._apply_extra`. Container-side execution is untouched; each mode's `config.json` is byte-for-byte preserved.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env).
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation.
- **Byte-compat is the success criterion:** each mode's `config.json` (keys AND ORDER) must match the pre-migration flat tool exactly. Ship a full-dict `config.json` equality test **with an explicit `list(cfg.keys()) == list(expected.keys())` key-order assertion** per mode (the rosetta test shape — reuse it).
- **Do NOT modify `ScoringInput`** (evoef2, many consumers) or `StructureDesignInput` (complexa — after this migration it has zero production consumers and is removed in teardown, NOT here). Add new dedicated input classes.
- Adopt `self._apply_extra(config, input_data)` in both runners (replacing evoef2's `_CONSUMED_EXTRA_KEYS` filter loop and complexa's empty-consumed-keys manual loop). Delete the consumed-keys sets.
- `mutations` config key (evoef2 build_mutant) is consumed by `containers/evoef2/standardize.py` — keep its name (`mutations`) + shape (`list[str]`) exactly.
- Catalog `Mode` has no `input_format` field — drop the `_*_INPUT_FORMAT` tuples; fold their guidance into field descriptions and/or the per-mode `Mode.notes`. Reword `notes` that reference `extra['...']` for now-typed fields (evoef2 repair/split_chains/mutations) to describe them as top-level fields. `notes` render in `info` (as of Plan 4c).
- Delete the flat `TOOL_REGISTRY[...]` blocks (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS`: replace the flat-name entries with a single `"evoef2"` / `"complexa"` (rosetta precedent — first key-collapsing migrations were rosetta; do the same here).
- Merged exemplars: `src/autobio/tools/rosetta.py` (multi-mode, per-mode config table, key collapse, byte-compat test shape), `src/autobio/tools/freesasa.py` (per-mode input subclasses), `src/autobio/tools/rfd3.py` (design_specs escape-hatch + copy/rewrite). Recon: `.superpowers/sdd/recon/evoef2.md`, `.superpowers/sdd/recon/complexa.md`.
- Commit convention `<component>: <what>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Migrate `evoef2` to a 3-mode catalog Tool

**Files:** Modify `src/autobio/schemas/scoring.py` (3 new input classes), `src/autobio/tools/evoef2.py`, `src/autobio/tools/__init__.py` (`TOOL_RUNNERS`); Test `tests/unit/test_evoef2_e2e.py` (+ integration if present). Read the current `src/autobio/tools/evoef2.py` in full. Fact sheet: `.superpowers/sdd/recon/evoef2.md`. `Any`/`Literal`/`Tier`/`Widget`/`ui` already imported in `scoring.py`.

**New schema classes** (append to `scoring.py`):

```python
class EvoEF2BaseInput(BaseInput):
    """Shared input for EvoEF2 modes (PDB only)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure (EvoEF2 supports PDB only).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )


class EvoEF2RepairInput(EvoEF2BaseInput):
    """Input for EvoEF2 repair mode (rebuild side chains, optimize hydrogen positions)."""


class EvoEF2BindingInput(EvoEF2BaseInput):
    """Input for EvoEF2 protein-protein binding-energy mode."""

    repair: bool = Field(
        default=True,
        description="Auto-repair the structure before scoring.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=10),
    )
    split_chains: str | None = Field(
        default=None,
        description=(
            "Chain grouping 'group1,group2' (exactly one comma), e.g. 'A,BC' = chain A "
            "vs chains B+C. None uses EvoEF2's default grouping."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=11),
    )


class EvoEF2BuildMutantInput(EvoEF2BaseInput):
    """Input for EvoEF2 build-mutant mode."""

    mutations: list[str] = Field(
        description=(
            "Mutations to introduce, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['EA63Q', 'KB42A']); applied simultaneously."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
```

**`config.json` contract per mode (byte-compat — exact order):**
- **All modes:** `command` = `_MODE_COMMAND[mode]` (repair→`"RepairStructure"`, binding→`"ComputeBinding"`, build_mutant→`"BuildMutant"`); `structure_path` = `/workspace/inputs/{name}`; `evoef2_bin` = existing const; `out_dir` = `"/workspace/outputs/raw"`.
- **binding** then: `repair` = `input_data.repair` (always written); `split_chains` = `input_data.split_chains` (ONLY if truthy).
- **build_mutant** then: `mutations` = `input_data.mutations`; `mutant_file` = `"/workspace/inputs/individual_list.txt"` (after `_write_mutation_file` side-effects the file).
- **All modes** end with `self._apply_extra(config, input_data)`.

**Runner transform** (`evoef2.py`):
- Swap imports (catalog); delete the 3 `TOOL_REGISTRY[...]` blocks, `_VARIANT_CONFIG`, `_CONSUMED_EXTRA_KEYS`, the `_*_INPUT_FORMAT` tuples. Add a small `_MODE_COMMAND` map (mode name → EvoEF2 command). Keep `_EVOEF2_BIN` const, `_MUTATION_RE`, `_MUTATION_FORMAT_HELP`, `_write_mutation_file`, `_resolve_container_path`, the three `_*_NOTES` (reworded, see below).
- `prepare_workspace`: `assert isinstance(input_data, EvoEF2BaseInput)`; `mode = self.current_mode.name`; validate; copy structure; build config per contract; `if mode == "binding":` write `repair`/`split_chains` (reading `input_data.repair`/`input_data.split_chains` — narrow with `assert isinstance(input_data, EvoEF2BindingInput)`); `if mode == "build_mutant":` write mutations + mutant_file (narrow to `EvoEF2BuildMutantInput`); `self._apply_extra(config, input_data)`.
- `_validate_inputs`: keep structure-exists + PDB-suffix (all modes); for `mode == "build_mutant"`: empty-`mutations` → `f"EvoEF2 build_mutant requires at least one mutation. {_MUTATION_FORMAT_HELP}"` (accurate wording; drop the "must be a list of strings" check — Pydantic enforces); keep the per-mutation `_MUTATION_RE` loop. For `mode == "binding"`: keep the `split_chains` comma-count==1 check (message unchanged). Gate both on `self.current_mode.name`.
- `parse_output`, `_resolve_container_path`: unchanged.
- **Reword `_*_NOTES`:** `_BINDING_NOTES`/`_BUILD_MUTANT_NOTES` reference `extra['repair']`/`extra['split_chains']`/`extra['mutations']` — reword to "the repair field"/"the split_chains field"/"the mutations field" (now typed). Fold `_*_INPUT_FORMAT` guidance into the field descriptions above (already done) or drop.

**Tool** (reuse reworded `_REPAIR_NOTES`/`_BINDING_NOTES`/`_BUILD_MUTANT_NOTES`):

```python
EVOEF2_TOOL = Tool(
    name="evoef2",
    display_name="EvoEF2",
    category=ToolCategory.SCORING,
    description=(
        "EvoEF2 physics-based protein structure repair, binding-energy scoring, and "
        "mutant building. Modes: repair, binding, build_mutant."
    ),
    version="1.0.0",
    image_tag="evoef2:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="repair",
    modes={
        "repair": Mode(
            name="repair", display_name="Repair",
            description="Rebuild incomplete side chains and optimize hydrogen positions.",
            input_schema=EvoEF2RepairInput, output_schema=ScoringOutput,
            default_timeout=600, notes=_REPAIR_NOTES,
        ),
        "binding": Mode(
            name="binding", display_name="Binding energy",
            description="Compute protein-protein binding energy (auto-repairs by default).",
            input_schema=EvoEF2BindingInput, output_schema=ScoringOutput,
            default_timeout=600, notes=_BINDING_NOTES,
        ),
        "build_mutant": Mode(
            name="build_mutant", display_name="Build mutant",
            description="Introduce amino-acid substitutions and optimize the local environment.",
            input_schema=EvoEF2BuildMutantInput, output_schema=ScoringOutput,
            default_timeout=600, notes=_BUILD_MUTANT_NOTES,
        ),
    },
    keywords=("evoef2", "scoring", "repair", "binding energy", "mutant", "ddg"),
)
"""Catalog Tool for EvoEF2 (repair/binding/build_mutant modes)."""

register(EVOEF2_TOOL)
```

**`TOOL_RUNNERS`:** remove `evoef2_repair`/`evoef2_binding`/`evoef2_build_mutant`; add `"evoef2": EvoEF2Runner`.

- [ ] **Step 1:** Add the 3 input classes to `scoring.py`; `python -m pytest tests/unit/test_schemas.py -q`.
- [ ] **Step 2:** Migrate `evoef2.py` per the transform + per-mode contract; update `TOOL_RUNNERS`.
- [ ] **Step 3:** Update `tests/unit/test_evoef2_e2e.py` (+ integration): construct per-mode input classes; `_make_runner` sets `runner.current_mode = get_tool("evoef2").modes[mode]` (freesasa pattern); catalog registration test (`get_tool("evoef2")`, modes=={"repair","binding","build_mutant"}, default_mode "repair", 3 flat names absent from `TOOL_REGISTRY`, `"evoef2"` in `TOOL_RUNNERS` + 3 old keys gone); full-dict `config.json` equality + key-order test PER mode (incl. binding's split_chains-present-and-absent, build_mutant's mutant_file); `info` snapshot (per-mode notes + hints + output_schema); extra-shadow-rejection test; reworded empty-mutations message test. Integration tests → `get_runner("evoef2").run(..., mode=...)`. Do NOT touch `containers/`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_evoef2_e2e.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `evoef2: migrate to catalog Tool with 3 modes (repair/binding/build_mutant)`.

---

### Task 2: Migrate `complexa` to a 3-mode catalog Tool

**Files:** Modify `src/autobio/schemas/structure_design.py` (add `ComplexaInput`), `src/autobio/tools/complexa.py`, `src/autobio/tools/__init__.py` (`TOOL_RUNNERS`); Test `tests/unit/test_complexa.py` (+ integration if present). Read the current `src/autobio/tools/complexa.py` in full. Fact sheet: `.superpowers/sdd/recon/complexa.md`. `Any`/`Tier`/`Widget`/`ui` already imported in `structure_design.py` (from the rfd3 migration).

**New schema class** (append to `structure_design.py`; mirrors `RFD3Input`):

```python
class ComplexaInput(BaseInput):
    """Input for Proteina-Complexa binder/scaffold design (protein_binder/ligand_binder/ame modes)."""

    design_specs: dict[str, dict[str, Any]] = Field(
        description=(
            "Named design specifications; each value is a dict of Complexa-native keys — "
            "common: 'input'/'target_input' (target structure filename), 'hotspot_residues', "
            "'binder_length', 'binder_center', 'pdb_id'; ligand_binder adds 'ligand'/'smiles'/"
            "'ligand_chain'/'ligand_only'/'use_bonds_from_file'; ame adds 'motif_residues'/"
            "'contig_atoms'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    input_structures: list[Path] = Field(
        default_factory=list,
        description=(
            "PDB/mmCIF files referenced by design_specs 'input' values; copied into the "
            "workspace with 'input' paths rewritten to container paths."
        ),
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=1),
    )
    n_batches: int = Field(
        default=1,
        description="Number of independent design batches per specification.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
```

**Runner mode-config table** (`complexa.py`, replacing `_VARIANT_CONFIG`, keyed by MODE name; `variant` is dropped — it equals the mode name):

```python
_MODE_CONFIG: dict[str, dict[str, str]] = {
    "protein_binder": {
        "pipeline_config": "search_binder_local_pipeline",
        "ckpt_name": "complexa.ckpt",
        "ae_ckpt_name": "complexa_ae.ckpt",
    },
    "ligand_binder": {
        "pipeline_config": "search_ligand_binder_local_pipeline",
        "ckpt_name": "complexa_ligand.ckpt",
        "ae_ckpt_name": "complexa_ligand_ae.ckpt",
    },
    "ame": {
        "pipeline_config": "search_ame_local_pipeline",
        "ckpt_name": "complexa_ame.ckpt",
        "ae_ckpt_name": "complexa_ame_ae.ckpt",
    },
}
```

**`config.json` contract (byte-compat — exact order, all 3 modes identical shape):** `variant` = `self.current_mode.name`; `pipeline_config` = `_MODE_CONFIG[mode]["pipeline_config"]`; `ckpt_name` = `_MODE_CONFIG[mode]["ckpt_name"]`; `ae_ckpt_name` = `_MODE_CONFIG[mode]["ae_ckpt_name"]`; `weights_dir` = `_WEIGHTS_DIR` const; `design_specs` = deep-copied `input_data.design_specs` with `"input"` values rewritten to `/workspace/inputs/{name}`; `n_batches` = `input_data.n_batches`; `out_dir` = `"/workspace/outputs/raw"`; then `self._apply_extra(config, input_data)`.

**Runner transform** (`complexa.py`):
- Swap imports (catalog); delete the 3 `TOOL_REGISTRY[...]` blocks, `_VARIANT_CONFIG`, `_CONSUMED_EXTRA_KEYS`, and the dead `_VALID_SPEC_KEYS` set. Drop the `_*_INPUT_FORMAT` tuples. Keep `_WEIGHTS_DIR`, `_resolve_container_path`, the copy+rewrite logic, and the three `_*_NOTES`.
- `prepare_workspace`: `assert isinstance(input_data, ComplexaInput)`; `mode = self.current_mode.name`; `mode_cfg = _MODE_CONFIG[mode]`; validate; copy input_structures + deep-copy design_specs + rewrite `input` paths (verbatim logic); build config per the contract (`config["variant"] = mode`); `self._apply_extra(config, input_data)`.
- `_validate_inputs`: retype to `ComplexaInput`; keep all checks (design_specs non-empty, dict values, n_batches>=1, input files exist, spec-input cross-ref) + messages verbatim.
- `parse_output`, `_resolve_container_path`: unchanged (still `StructureDesignOutput`; `evaluation_metrics` handling preserved).
- Add a brief comment noting the naming collision: catalog `Mode` (protein_binder/ligand_binder/ame) vs the Complexa pipeline's own `config["mode"]` (generate/design) which flows through `extra` and is consumed container-side only.

**Tool** (reuse the three `_*_NOTES`; fold `_*_INPUT_FORMAT` guidance into notes/descriptions):

```python
COMPLEXA_TOOL = Tool(
    name="complexa",
    display_name="Proteina-Complexa",
    category=ToolCategory.STRUCTURE_DESIGN,
    description=(
        "Design novel protein binders and scaffolds with Proteina-Complexa (flow-matching "
        "sequence+structure generation). Modes: protein_binder, ligand_binder, ame (atomistic "
        "motif extension). Provide targets/hotspots/length constraints via design_specs."
    ),
    version="2.0.0",
    image_tag="complexa:2.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="protein_binder",
    modes={
        "protein_binder": Mode(
            name="protein_binder", display_name="Protein binder",
            description="Design binders for a protein target.",
            input_schema=ComplexaInput, output_schema=StructureDesignOutput,
            default_timeout=43200, supports_batch=True, notes=_COMPLEXA_NOTES,
        ),
        "ligand_binder": Mode(
            name="ligand_binder", display_name="Ligand binder",
            description="Design binders for a small-molecule ligand target.",
            input_schema=ComplexaInput, output_schema=StructureDesignOutput,
            default_timeout=43200, supports_batch=True, notes=_COMPLEXA_LIGAND_NOTES,
        ),
        "ame": Mode(
            name="ame", display_name="AME (motif scaffolding)",
            description="Scaffold a functional motif into a complete protein.",
            input_schema=ComplexaInput, output_schema=StructureDesignOutput,
            default_timeout=43200, supports_batch=True, notes=_COMPLEXA_AME_NOTES,
        ),
    },
    keywords=("complexa", "proteina", "binder design", "scaffold", "ligand", "motif", "ame"),
)
"""Catalog Tool for Proteina-Complexa (protein_binder/ligand_binder/ame modes)."""

register(COMPLEXA_TOOL)
```

**`TOOL_RUNNERS`:** remove `complexa`/`complexa_ligand`/`complexa_ame`; add `"complexa": ComplexaRunner`.

- [ ] **Step 1:** Add `ComplexaInput` to `structure_design.py`; import check.
- [ ] **Step 2:** Migrate `complexa.py` per the transform + contract; update `TOOL_RUNNERS`.
- [ ] **Step 3:** Update `tests/unit/test_complexa.py` (+ integration): construct `ComplexaInput`; parametrize registration/config tests over the 3 mode names (was 3 flat names); `_make_runner` sets `current_mode`; catalog registration test (`get_tool("complexa")`, modes=={"protein_binder","ligand_binder","ame"}, default_mode, 3 flat names absent from `TOOL_REGISTRY`, `"complexa"` in `TOOL_RUNNERS` + 3 old keys gone); full-dict `config.json` equality + key-order test per mode (variant/pipeline_config/ckpt_name/ae_ckpt_name differ per mode; design_specs input-rewrite; a merged extra key like `mode="design"` or `batch_size`); `info` snapshot (per-mode notes + hints + output_schema); extra-shadow-rejection test. Integration tests → `get_runner("complexa").run(..., mode=...)`. Do NOT touch `containers/` or `test_complexa_standardize.py`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_complexa.py tests/unit/test_complexa_standardize.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `complexa: migrate to catalog Tool with 3 modes (protein_binder/ligand_binder/ame)`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info evoef2`/`autobio info complexa` show their 3 modes with per-mode notes + `x-autobio` hints + `output_schema`; all 6 flat names gone from `TOOL_REGISTRY` + `TOOL_RUNNERS` (only `"evoef2"`/`"complexa"` remain); disjointness guard green. `StructureDesignInput` now has zero production consumers (removal deferred to teardown).

## Self-Review

**1. Spec coverage:** evoef2 → 3-mode Tool (per-mode input subclasses off `EvoEF2BaseInput`); complexa → 3-mode Tool (one shared `ComplexaInput`, design_specs escape-hatch, mode-keyed checkpoint table). Both: `self.current_mode.name` dispatch, `_apply_extra`, byte-compat config (full-dict + key-order test per mode), flat entries + `input_format` removed, notes on Modes (reworded for typed fields), `TOOL_RUNNERS` collapsed. `ScoringInput`/`StructureDesignInput` untouched. `mutations` container-contract key preserved. RunMetadata.mode auto-carries (from 4f). complexa frees `StructureDesignInput` for teardown.

**2. Placeholder scan:** Full code for all new input classes and both `Tool` objects; runner transforms specified by exact per-mode byte-compat contracts + in-repo source + rosetta/freesasa/rfd3 exemplars; no "TBD".

**3. Type consistency:** evoef2 input classes referenced consistently (schema, runner `isinstance` narrowing, Tool modes, tests); complexa `ComplexaInput` likewise. Output schemas reuse `ScoringOutput`/`StructureDesignOutput`. `_apply_extra` unchanged.

## Next plans (Plan 4 continued)
- **Two-class consolidation:** esm_if1, antifold (each currently 2 flat names + 2 runner classes → ONE Tool w/ `{design, score}` modes + ONE runner; frees `InverseFoldingInput`/`ScoringInput` further).
- **Output-variance:** openmm (per-mode image + `SimulationOutput`), antibody LMs ×6 (antibody `SequenceSet`, output variance, shared runner).
- **Teardown:** remove `TOOL_REGISTRY`/`ToolEntry` + now-unused category input schemas (`StructurePredictionInput`, `StructureDesignInput`, and any others freed); hoist duplicated `_resolve_container_path` onto `ToolRunner`; dead-code cleanups (`_ROSETTA_BIN`, evoef2 `out_dir`, complexa mode-doc); wire-or-document copied-but-unwired templates/msa; README rewrite with migration notes (mode-based invocation, dropped aliases, extra-shadowing).
