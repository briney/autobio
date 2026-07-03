# Tools→Modes Plan 4b — Structure-Prediction Singleton Migrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate `esmfold`, `boltz1`, and `boltz2` from the legacy flat `TOOL_REGISTRY` to the `Tool`/`Mode` catalog. Second PR of the Plan 4 family-migration series.

**Architecture:** `esmfold` is a trivial single-mode migration (single-chain protein, zero consumed `extra` keys) that adopts `GenericSequenceSet` for FASTA input. `boltz1`/`boltz2` are two Tools sharing one `BoltzRunner` (esm1b/esm2 pattern), distinguished only by `config["model"]`; their migration promotes the runner-consumed structural `extra` keys (`entity_types`, `boltz_yaml`, `msa_paths`, `constraints`, `properties`, `modifications`) **and** `use_msa_server` to typed fields, so the remaining `extra` (CLI knobs) merges uniformly through the hardened `_apply_extra`. Container-side execution is untouched; each tool's `config.json` (and boltz's generated `input.yaml`) is byte-for-byte preserved.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML (boltz), pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare `pytest` = wrong env). Reinstall editable if `src/` edits aren't picked up.
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation.
- **Byte-compat is the success criterion:** each tool's `config.json` (and, for boltz, the generated `input.yaml`) must be identical to pre-migration output. Ship a full-dict `config.json` equality test per tool; for boltz, also assert the generated YAML is unchanged for a representative input.
- **Do NOT modify `StructurePredictionInput`** in `src/autobio/schemas/structure_prediction.py` — it is still consumed by unmigrated `chai1`/`openfold3`. Add NEW dedicated input classes.
- **boltz `sequences` stays `dict[str, str]`** (NOT `GenericSequenceSet`) — boltz accepts DNA/RNA/SMILES/placeholder values, so protein-FASTA normalization is unsafe. `esmfold` DOES adopt `GenericSequenceSet` (single-chain protein).
- Catalog `Tool`/`Mode` have no `input_format` field — drop the legacy `input_format` tuples; fold their essential guidance into field `description`s / Mode `notes`. Legacy `notes` move onto the `Mode`.
- Each migration MUST delete the tool's `TOOL_REGISTRY[...] = ToolEntry(...)` block (and `_CONSUMED_EXTRA_KEYS`), or `tests/unit/test_registry_disjoint.py` fails. `TOOL_RUNNERS` entries (`esmfold`/`boltz1`/`boltz2`) stay — the Tool names equal the existing flat names.
- Use the merged exemplars: `src/autobio/tools/esm.py` (two Tools one runner via `self.tool.name`; `_apply_extra`; named `<TOOL>_TOOL` constants; `GenericSequenceSet` on `ESMEmbedInput` in `schemas/embedding.py`) and `src/autobio/tools/freesasa.py` (typed fields + `ui()` hints + host-side `_validate_inputs`).
- Commit convention `<component>: <what changed and why>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Migrate `esmfold` to the catalog

**Files:** Modify `src/autobio/schemas/structure_prediction.py` (imports + `ESMFoldInput`), `src/autobio/tools/esmfold.py`; Test `tests/unit/test_esmfold.py` (+ `tests/integration/test_esmfold_integration.py` if present).

**Schema module prep** (`structure_prediction.py`): add `from typing import Any`; `from autobio.schemas.hints import Tier, Widget, ui`; `from autobio.schemas.sequences import GenericSequenceSet  # noqa: TC001 - needed at runtime`. Do NOT touch `StructurePredictionInput`/`PredictedStructure`/`ConfidenceMetrics`/`StructurePredictionOutput`.

**New class** (append):

```python
class ESMFoldInput(BaseInput):
    """Input for ESMFold single-sequence structure prediction (single ``predict`` mode)."""

    sequences: GenericSequenceSet = Field(
        description=(
            "A single protein sequence: a dict of id→sequence (one chain), "
            "FASTA text, or a FASTA file path."
        ),
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="generic", tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        description="Number of models. ESMFold is deterministic; must be 1.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Template structures. ESMFold does not use templates; must be None/empty.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=11),
    )
```

**`config.json` contract (byte-compat):** `model_name` = `_MODEL_NAME` const; `input_fasta` = `"/workspace/inputs/sequences.fasta"`; `output_dir` = `"/workspace/outputs/raw"`; `hf_cache` = `_HF_CACHE` const; then `self._apply_extra(config, input_data)` (replaces the current unfiltered `config.update(input_data.extra)`). Sequences are still written via `write_fasta(input_data.sequences, workspace.inputs_dir / "sequences.fasta")`.

**Validation to preserve** (`_validate_inputs`, reading typed fields, same messages): sequences non-empty (`"sequences must be non-empty."`); single-chain (`"ESMFold is single-chain only. Received ... chains ..."`); each sequence valid protein (`"Invalid protein sequence for ...: must contain only standard amino acid characters (ACDEFGHIKLMNPQRSTVWY)."`); templates None/empty (`"ESMFold does not use templates. ..."`); `num_models == 1` (`"ESMFold is deterministic ... num_models must be 1, got {num_models}."`).

**Tool** (`_ESMFOLD_NOTES` reused; drop `_ESMFOLD_INPUT_FORMAT` — its content is in the `sequences` description):

```python
ESMFOLD_TOOL = Tool(
    name="esmfold",
    display_name="ESMFold",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict protein structure from a single sequence using ESMFold. "
        "No MSA or templates needed — direct sequence-to-structure prediction."
    ),
    version="1.0.0",
    image_tag="esmfold:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a single-chain protein structure from sequence.",
            input_schema=ESMFoldInput,
            output_schema=StructurePredictionOutput,
            default_timeout=600,
            notes=_ESMFOLD_NOTES,
        )
    },
    keywords=("esmfold", "structure prediction", "protein folding", "single sequence"),
)
"""Catalog Tool for ESMFold — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(ESMFOLD_TOOL)
```

**Runner:** swap imports (`from autobio.core.catalog import Mode, Tool, register`; keep `ToolCategory`); delete the `TOOL_REGISTRY["esmfold"]` block; `assert isinstance(input_data, ESMFoldInput)`; build config per the contract; `self._apply_extra(config, input_data)`; keep `parse_output` and `_resolve_container_path` unchanged.

- [ ] **Step 1:** Add imports + `ESMFoldInput` to `structure_prediction.py`; `python -m pytest tests/unit/test_schemas.py -q`.
- [ ] **Step 2:** Migrate `esmfold.py` per the contract above.
- [ ] **Step 3:** Update `tests/unit/test_esmfold.py`: construct `ESMFoldInput`; add a FASTA-text acceptance test (`sequences=">A\nMKT\n"` normalizes to `{"A": "MKT"}`); catalog registration test (`get_tool("esmfold")`, modes=={"predict"}, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`); `info` snapshot (`sequences` hint `widget=="sequence"`/`flavor=="generic"`, `output_schema` present); full-dict `config.json` equality test. Keep all `_validate_inputs` message-match cases. If `tests/integration/test_esmfold_integration.py` exists, swap its input construction to `ESMFoldInput` (Docker-free edit; verify via `--collect-only` + a host-side construction snippet).
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_esmfold.py tests/unit/test_registry_disjoint.py -v`; then full `python -m pytest -m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `esmfold: migrate to catalog Tool with GenericSequenceSet input (predict mode)`.

---

### Task 2: Migrate `boltz1` and `boltz2` to the catalog

**Files:** Modify `src/autobio/schemas/structure_prediction.py` (add `BoltzInput`), `src/autobio/tools/boltz.py`; Test `tests/unit/test_boltz.py` (+ `tests/integration/test_boltz_integration.py` if present). Fact sheet: `.superpowers/sdd/recon/sp-b.md` (boltz section). Read the current `src/autobio/tools/boltz.py` in full — the YAML-generation logic (`_build_boltz_yaml`) is preserved verbatim except that it reads typed fields instead of `input_data.extra`.

**Design (byte-compat rationale):** boltz's runner *consumes* several `extra` keys to build `input.yaml` (they are NOT written to `config.json`). The hardened `_apply_extra` merges *all* non-colliding `extra` into `config.json`, so those consumed keys must leave `extra` — promote them to typed fields. `use_msa_server` is also promoted because the current code documents `extra["use_msa_server"]=False` as an override, which the strengthened guard would now reject (it collides with the `config["use_msa_server"]` key). The remaining `extra` (CLI knobs: `sampling_steps`, `recycling_steps`, `step_scale`, `output_format`, `seed`, `write_full_pae`, `write_full_pde`, `write_embeddings`, `max_parallel_samples`, and boltz2's `sampling_steps_affinity`/`diffusion_samples_affinity`/`method`) stays in `extra` and flat-merges to `config.json` via `_apply_extra` — byte-compat preserved (each is written only when the caller supplies it). Both boltz1 and boltz2 share `BoltzInput` (their differences are CLI knobs, which stay in `extra`).

**New class** (append to `structure_prediction.py`; `Any`/`Tier`/`Widget`/`ui` imported in Task 1):

```python
class BoltzInput(BaseInput):
    """Input for Boltz-1 / Boltz-2 structure prediction (shared by both Tools)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence. Values may be protein/DNA/RNA; for "
            "ligand chains the value is ignored when SMILES/CCD is given via "
            "entity_types. May be empty only when boltz_yaml is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1, ge=1,
        description="Number of structures to generate (maps to diffusion_samples).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Template structures (PDB/mmCIF) copied into the workspace.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=11),
    )
    entity_types: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain entity type: 'protein'/'dna'/'rna', or a dict "
            "{'smiles': 'CC...'} / {'ccd': 'ATP'} for ligands. Default 'protein'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=13),
    )
    msa_paths: list[str] | None = Field(
        default=None,
        description="Pre-computed MSA file paths (filename starts with the chain ID, e.g. 'A.a3m').",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    constraints: list[Any] | None = Field(
        default=None,
        description="Boltz YAML 'constraints' section (bond/pocket/contact entries).",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    properties: list[Any] | None = Field(
        default=None,
        description="Boltz YAML 'properties' section (Boltz-2).",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=16),
    )
    modifications: list[Any] | None = Field(
        default=None,
        description="Boltz YAML 'modifications' section.",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=17),
    )
    boltz_yaml: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Raw Boltz YAML (dict or string) — bypasses automatic YAML generation "
            "for full control over sequences/constraints/modifications/properties."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=18),
    )
```

**`config.json` contract (byte-compat):** `model` = `_MODEL_CONFIG[self.tool_name]`; `input_path` = `"/workspace/inputs/input.yaml"`; `output_dir` = `"/workspace/outputs/raw"`; `cache_dir` = `_BOLTZ_CACHE`; `use_msa_server` = `input_data.use_msa_server`; then, if `input_data.num_models > 1`, `config["diffusion_samples"] = input_data.num_models`; then `self._apply_extra(config, input_data)` (merges the remaining CLI-knob `extra`). This is identical to today's output for the same inputs.

**Runner transform** (`boltz.py`):
- Swap imports to catalog; delete `TOOL_REGISTRY["boltz1"]`/`["boltz2"]` blocks and the `_CONSUMED_EXTRA_KEYS` set.
- `prepare_workspace`: `assert isinstance(input_data, BoltzInput)`. Replace every `input_data.extra.get("<key>")` / `"<key>" in input_data.extra` for the promoted keys with the typed field: `input_data.boltz_yaml is not None` (was `"boltz_yaml" in extra`), `input_data.entity_types`, `input_data.msa_paths`, `input_data.constraints`/`properties`/`modifications`, `input_data.use_msa_server`. Build config per the contract; end with `self._apply_extra(config, input_data)` (replacing the `_CONSUMED_EXTRA_KEYS` filter loop).
- `_build_boltz_yaml`: preserve the logic verbatim, reading `input_data.entity_types` / `input_data.constraints` / `input_data.properties` / `input_data.modifications` / `input_data.msa_paths` / `input_data.templates` instead of `input_data.extra[...]`. The generated YAML must be identical for the same inputs.
- `_validate_inputs`: `has_boltz_yaml = input_data.boltz_yaml is not None`; keep the "sequences non-empty or boltz_yaml" check (same message), the template/MSA existence checks, and the `entity_types` unknown-chain check — all reading typed fields.
- `parse_output`, `_resolve_container_path`: unchanged.
- Keep `_MODEL_CONFIG` and `_BOLTZ_CACHE`.

**Tools** (both use `BoltzInput` + `BoltzRunner`; reuse `_BOLTZ1_NOTES`/`_BOLTZ2_NOTES`; drop `_BOLTZ_INPUT_FORMAT` — fold its entity-type/constraint/YAML examples into the `entity_types`/`constraints`/`boltz_yaml` field descriptions if not already conveyed, else drop):

```python
BOLTZ1_TOOL = Tool(
    name="boltz1",
    display_name="Boltz-1",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures using Boltz-1. Supports proteins, DNA, "
        "RNA, and ligand complexes with template-based and ab initio prediction."
    ),
    version="1.0.0",
    image_tag="boltz:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a biomolecular complex structure.",
            input_schema=BoltzInput,
            output_schema=StructurePredictionOutput,
            default_timeout=3600,
            notes=_BOLTZ1_NOTES,
        )
    },
    keywords=("boltz", "boltz1", "structure prediction", "complex", "docking"),
)
"""Catalog Tool for Boltz-1."""

register(BOLTZ1_TOOL)

BOLTZ2_TOOL = Tool(
    name="boltz2",
    display_name="Boltz-2",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures and binding affinity using Boltz-2. "
        "Supports proteins, DNA, RNA, and ligand complexes; affinity approaches "
        "FEP accuracy at ~1000x speed."
    ),
    version="1.0.0",
    image_tag="boltz:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure + affinity",
            description="Predict a complex structure and (protein-ligand) binding affinity.",
            input_schema=BoltzInput,
            output_schema=StructurePredictionOutput,
            default_timeout=7200,
            notes=_BOLTZ2_NOTES,
        )
    },
    keywords=("boltz", "boltz2", "structure prediction", "affinity", "complex"),
)
"""Catalog Tool for Boltz-2."""

register(BOLTZ2_TOOL)
```

- [ ] **Step 1:** Add `BoltzInput` to `structure_prediction.py`; import check (`python -m pytest tests/unit/test_schemas.py -q`).
- [ ] **Step 2:** Migrate `boltz.py` per the transform + contract, preserving `_build_boltz_yaml` behavior exactly (reads typed fields).
- [ ] **Step 3:** Update `tests/unit/test_boltz.py`: construct `BoltzInput` passing `entity_types`/`msa_paths`/`constraints`/`boltz_yaml`/`use_msa_server` as **top-level fields** (not `extra`); keep CLI-knob params (`sampling_steps`, etc.) in `extra` and assert they still flat-merge to config. Add: catalog registration test for BOTH boltz1 and boltz2 (`get_tool(...)`, modes=={"predict"}, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`); a test that `get_runner("boltz1")` and `get_runner("boltz2")` write `config["model"]` = `"boltz1"`/`"boltz2"` respectively; a full-dict `config.json` equality test; a **generated-YAML equality test** for a representative multi-entity input (protein + ligand via `entity_types`) proving `input.yaml` is unchanged; an `info` snapshot (a promoted field carries its hint; `output_schema` present); an extra-shadow-rejection test (passing `entity_types` or `use_msa_server` via `extra` now raises). If `tests/integration/test_boltz_integration.py` exists, swap input construction to `BoltzInput` (Docker-free; verify `--collect-only` + construction snippet).
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_boltz.py tests/unit/test_registry_disjoint.py -v`; then full `python -m pytest -m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `boltz: migrate boltz1/boltz2 to catalog Tools; promote structural extra keys to typed fields`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio list` shows esmfold/boltz1/boltz2 as catalog Tools; `autobio info <tool>` returns the `predict` mode with `x-autobio` hints + `output_schema`; all three absent from `TOOL_REGISTRY` (disjointness guard green). Integration tests (if edited) import cleanly.

## Self-Review

**1. Spec coverage:** esmfold + boltz1 + boltz2 migrated to Tools/Modes with typed fields + `x-autobio`; byte-compat config.json (+ boltz YAML) with full-dict/YAML equality tests; flat entries removed; `StructurePredictionInput` untouched (chai1/openfold3 unaffected); esmfold adopts `GenericSequenceSet` (FASTA), boltz keeps `dict[str,str]` (DNA/RNA/SMILES). RunMetadata.mode still deferred (single-mode tools). boltz1/boltz2 = two Tools one runner ([[feedback-model-tools]]).

**2. Placeholder scan:** Full code for both new schema classes and all three `Tool` objects; runner transforms specified by exact byte-compat contracts + the in-repo boltz/esmfold source + esm/freesasa exemplars; no "TBD"/"handle edge cases".

**3. Type consistency:** `ESMFoldInput`/`BoltzInput` referenced consistently across schema module, runners, Tool `input_schema`, tests. `GenericSequenceSet` imported in `structure_prediction.py` (matches `embedding.py` usage). Output schema reuses `StructurePredictionOutput`. `_apply_extra` unchanged. `self.tool_name`/`_MODEL_CONFIG` dispatch preserved for boltz1/boltz2.

## Next plans (Plan 4 continued)
- **4c — complex structure-prediction singletons:** chai1, openfold3 (structural non-scalar extras → typed dict/list fields with `widget:json`; preserve pre-existing footguns, file bugs separately).
- **4d — rfd3 (structure_design):** keep `design_specs` as an escape-hatch dict; adopt `_apply_extra`; single mode.
- **Later:** mpnn family (proteinmpnn + ligandmpnn together — shared `MPNNRunner`); same-category multi-mode (rosetta [+`RunMetadata.mode`], evoef2, complexa); cross-category two-class consolidation (esm_if1, antifold); output-variance (openmm, antibody LMs ×6); then teardown + README.
