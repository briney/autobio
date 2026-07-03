# Tools→Modes Plan 4c — chai1 + openfold3 Migrations (and `notes` in `info`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Restore `notes` rendering to `autobio info` (the cross-cutting gap found in Plan 4b), then migrate the two complex structure-prediction tools `chai1` and `openfold3` from the flat `TOOL_REGISTRY` to the `Tool`/`Mode` catalog.

**Architecture:** Task 1 is a small formatter change (`format_tool_info_catalog` emits `Tool.notes` + per-`Mode` `notes`) so the guidance every migrated tool moved onto `Mode.notes` becomes visible again. Tasks 2–3 follow the merged boltz pattern: each tool's runner *consumes* structural `extra` keys to build its native input file (chai1 → FASTA, openfold3 → query JSON), so those keys — plus the constant-default boolean config knobs that callers override (`use_msa_server`, `use_esm_embeddings` / `use_templates`, `pae_enabled`) — are promoted to typed fields, leaving only plain CLI scalars in `extra` (flat-merged via the hardened `_apply_extra`). Promoting the raw-override keys (`chai_fasta` / `query_json`) to typed fields with `is not None`/truthiness semantics also resolves two pre-existing footguns (empty-FASTA / `"null"` data-loss) — a correct consequence of typing that matches the standing fail-fast guidance. Container-side execution is untouched; each tool's `config.json` and its generated input file are byte-for-byte preserved for all normal inputs.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env). Reinstall editable if `src/` edits aren't picked up.
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation.
- **Byte-compat is the success criterion:** each tool's `config.json` AND its generated native input file (chai1 `input.fasta`, openfold3 `query.json`) must be identical to pre-migration output for all normal inputs. Ship a full-dict `config.json` equality test and a generated-input-file equality test per tool.
- **Do NOT modify `StructurePredictionInput`** in `src/autobio/schemas/structure_prediction.py` — no flat structure-prediction tools remain after this plan, but leave it for the teardown plan to remove. Add NEW dedicated input classes (`Chai1Input`, `OpenFold3Input`).
- **chai1/openfold3 `sequences` stays `dict[str, str]`** (NOT `GenericSequenceSet`) — values are overloaded (protein/DNA/RNA sequences OR SMILES for ligand chains), so protein-FASTA normalization is unsafe.
- **Preserve the pre-existing "templates/msa copied but not wired into the generated input" behavior** (chai1 templates; openfold3 templates + `msa_paths`) — keep the host-side existence validation and the copy, do NOT wire them into the FASTA/query. These are pre-existing no-ops; file a separate follow-up, do not "fix" them here.
- **Footgun resolution (intentional, documented):** `chai_fasta`/`query_json` become typed fields; use a single consistent truthiness/`is not None` check in BOTH `_validate_inputs` and `prepare_workspace` so `chai_fasta=None`/`query_json=None` with empty `sequences` fails fast (or builds from sequences) instead of producing an empty FASTA / the literal `"null"`. This changes behavior ONLY for those degenerate, untested inputs; all normal inputs are byte-identical.
- Catalog `Tool`/`Mode` have no `input_format` field — drop the legacy `_*_INPUT_FORMAT` tuples; fold the essential format guidance (entity-type syntax, constraint/CSV columns, query-JSON shape, doc URLs) into the relevant field `description`s. Legacy `notes` move onto the `Mode` (and, after Task 1, are rendered by `info`).
- Each migration deletes its `TOOL_REGISTRY[...]` block + `_CONSUMED_EXTRA_KEYS` (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS` entries (`chai1`/`openfold3`) stay (Tool names == flat names).
- Merged exemplar to follow: `src/autobio/tools/boltz.py` (structural extra keys promoted to typed fields; `_apply_extra` for remaining CLI knobs; byte-compat config + generated input file). Also `src/autobio/tools/esmfold.py`.
- Commit convention `<component>: <what>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.
- Recon fact sheet (authoritative current-state): `.superpowers/sdd/recon/sp-a.md` (chai1 §2, openfold3 §3). Read your tool's section.

---

### Task 1: Render `Tool.notes` and `Mode.notes` in `format_tool_info_catalog`

**Why:** Plan 4b's final review found the catalog `info` formatter emits neither `Tool.notes` nor `Mode.notes`, so guidance every migrated tool moved onto `Mode.notes` is invisible in `autobio info` (the legacy flat formatter rendered notes). The user chose to restore notes. This is additive (existing `info`-snapshot tests check specific keys, not exact key sets) and benefits all already-merged tools.

**Files:** Modify `src/autobio/cli/formatters.py` (`format_tool_info_catalog`); Test `tests/unit/test_formatters.py`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_formatters.py`, extend the `_tool_for_info()` helper's `"a"` mode to carry `notes=("First note.", "Second note.")` (add `notes=(...)` to that `Mode(...)`), and add:

```python
def test_format_tool_info_catalog_json_includes_notes() -> None:
    parsed = json.loads(format_tool_info_catalog(_tool_for_info(), OutputFormat.JSON))
    assert parsed["modes"][0]["notes"] == ["First note.", "Second note."]
    assert parsed["modes"][1]["notes"] == []  # mode "b" has no notes


def test_format_tool_info_catalog_table_includes_notes() -> None:
    out = format_tool_info_catalog(_tool_for_info(), OutputFormat.TABLE)
    assert "First note." in out and "Second note." in out
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_formatters.py -k notes -v`
Expected: FAIL (no `notes` key in mode dict; note text absent from table).

- [ ] **Step 3: Emit notes in both branches**

In `format_tool_info_catalog` (`src/autobio/cli/formatters.py`): add `"notes": list(mode.notes),` to each mode dict in the JSON `modes` comprehension (after `"supports_batch"`), and add `"notes": list(tool.notes),` to the `data` dict (after `"keywords"`). In the TABLE branch, after each `Mode: {mode.name}` row, render the mode's notes when present:

```python
        if mode.notes:
            table.add_row("", "\n".join(f"- {n}" for n in mode.notes))
```

and, after the `Default Mode` row (before the modes loop), render tool-level notes when present:

```python
    if tool.notes:
        table.add_row("Notes", "\n".join(f"- {n}" for n in tool.notes))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/unit/test_formatters.py -v`
Expected: PASS (including the pre-existing `test_format_tool_info_catalog_json_shape`/`_table_runs`, which are additive-compatible).

- [ ] **Step 5: Full suite + lint/type**

Run: `python -m pytest -m "not docker and not gpu"` (existing `info`-snapshot tests for freesasa/esm/etc. must stay green — the added `notes` key is additive); then `ruff check --fix src/autobio/cli/formatters.py tests/unit/test_formatters.py`, `ruff format ...`, `mypy src/`.

- [ ] **Step 6: Commit**

```bash
git add src/autobio/cli/formatters.py tests/unit/test_formatters.py
git commit -m "cli: render Tool/Mode notes in catalog info output (JSON + TABLE)"
```

---

### Task 2: Migrate `chai1` to the catalog

**Files:** Modify `src/autobio/schemas/structure_prediction.py` (add `Chai1Input`), `src/autobio/tools/chai.py`; Test `tests/unit/test_chai.py` (+ `tests/integration/test_chai_integration.py`). Fact sheet: `.superpowers/sdd/recon/sp-a.md` §2. Read the current `src/autobio/tools/chai.py` in full — `_build_fasta` is preserved behaviorally (reads typed fields).

**New class** (append to `structure_prediction.py`; `Any`/`Tier`/`Widget`/`ui` are imported there from Plan 4b):

```python
class Chai1Input(BaseInput):
    """Input for Chai-1 biomolecular structure prediction (single ``predict`` mode)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence (protein/DNA/RNA). For ligand chains "
            "(entity_types = 'ligand' or {'smiles': ...}) the value is a SMILES string. "
            "May be empty only when chai_fasta is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1, ge=1,
        description="Number of structures to generate (maps to num_diffn_samples).",
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
            "Per-chain entity type: 'protein'/'dna'/'rna'/'ligand', or a dict "
            "{'smiles': 'CC...'} for ligands. Default 'protein'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    constraints: str | None = Field(
        default=None,
        description=(
            "Restraints/covalent bonds as CSV content (or a file path). Columns: "
            "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
            "min_distance_angstrom,max_distance_angstrom,comment,restraint_id. "
            "connection_type is 'contact', 'pocket', or 'covalent'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=13),
    )
    msa_directory: str | None = Field(
        default=None,
        description="Path to a directory of pre-computed MSA .aligned.pqt files.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    chai_fasta: str | None = Field(
        default=None,
        description=(
            "Raw Chai-1 FASTA content (headers '>entity_type|name=chain_id'), "
            "bypassing automatic FASTA generation from sequences."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=16),
    )
    use_esm_embeddings: bool = Field(
        default=False,
        description="Enable ESM protein language model embeddings (extra compute).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=17),
    )
```

**`config.json` contract (byte-compat):** `fasta_path` = `"/workspace/inputs/input.fasta"`; `output_dir` = `"/workspace/outputs/raw"`; `downloads_dir` = `_CHAI_DOWNLOADS_DIR`; `use_msa_server` = `input_data.use_msa_server`; `use_esm_embeddings` = `input_data.use_esm_embeddings`; `num_diffn_samples` = `input_data.num_models`; then `constraint_path` = `"/workspace/inputs/restraints.csv"` **iff** `input_data.constraints` (truthy); then `msa_directory` = `"/workspace/inputs/msa"` **iff** `input_data.msa_directory` (truthy); then `self._apply_extra(config, input_data)` (merges remaining CLI scalars: `num_trunk_recycles`, `num_diffn_timesteps`, `seed`, `low_memory`, `use_templates_server`).

**Runner transform** (`chai.py`):
- Swap imports to catalog; delete `TOOL_REGISTRY["chai1"]` + `_CONSUMED_EXTRA_KEYS`; add `CHAI1_TOOL` + `register`. Drop `_CHAI_INPUT_FORMAT` (fold key syntax into the field descriptions above); reuse `_CHAI_NOTES` on the Mode.
- `prepare_workspace`: `assert isinstance(input_data, Chai1Input)`. FASTA: `if input_data.chai_fasta:` use it verbatim, else `self._build_fasta(input_data)`. Constraints/MSA/templates staging read typed fields (`input_data.constraints`, `input_data.msa_directory`, `input_data.templates`). Build config per the contract; end with `self._apply_extra(config, input_data)`.
- `_build_fasta`: preserve exactly, reading `input_data.entity_types` (was `extra.get("entity_types", {})`); keep `sorted(input_data.sequences)` iteration order.
- `_validate_inputs`: `has_chai_fasta = bool(input_data.chai_fasta)` (truthiness, consistent with prepare — resolves the footgun); read `input_data.constraints`/`msa_directory`/`entity_types`/`templates`; preserve all messages.
- `parse_output`, `_resolve_container_path`: unchanged.

**Tool** (reuse `_CHAI_NOTES`):

```python
CHAI1_TOOL = Tool(
    name="chai1",
    display_name="Chai-1",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures using Chai-1. Supports proteins, DNA, RNA, "
        "ligands, and glycans with restraints and covalent bonds."
    ),
    version="1.0.0",
    image_tag="chai:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a biomolecular complex structure.",
            input_schema=Chai1Input,
            output_schema=StructurePredictionOutput,
            default_timeout=3600,
            notes=_CHAI_NOTES,
        )
    },
    keywords=("chai", "chai1", "structure prediction", "complex", "ligand", "glycan"),
)
"""Catalog Tool for Chai-1."""

register(CHAI1_TOOL)
```

- [ ] **Step 1:** Add `Chai1Input` to `structure_prediction.py`; import check.
- [ ] **Step 2:** Migrate `chai.py` per the transform + contract, preserving `_build_fasta` behavior.
- [ ] **Step 3:** Update `tests/unit/test_chai.py`: pass `entity_types`/`constraints`/`msa_directory`/`chai_fasta`/`use_msa_server`/`use_esm_embeddings` as TOP-LEVEL `Chai1Input` fields; keep CLI scalars in `extra` and assert they flat-merge. Add: catalog registration test; full-dict `config.json` equality test; a generated-`input.fasta` equality test for a protein+ligand (`entity_types`) input; an `info` snapshot (a promoted field's hint + `output_schema`; and, since Task 1 landed, assert `notes` present); an extra-shadow-rejection test (`entity_types`/`use_msa_server` via `extra` now raises). Add a footgun-resolution test: `Chai1Input(sequences={}, chai_fasta=None)` raises the "sequences must be non-empty" `AutobioError` (no empty FASTA). If `tests/integration/test_chai_integration.py` exists, swap its inputs to `Chai1Input` (Docker-free: `--collect-only` + host construction snippet). Do NOT touch `containers/`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_chai.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `chai1: migrate to catalog Tool; promote structural extra keys to typed fields`.

---

### Task 3: Migrate `openfold3` to the catalog

**Files:** Modify `src/autobio/schemas/structure_prediction.py` (add `OpenFold3Input`), `src/autobio/tools/openfold3.py`; Test `tests/unit/test_openfold3.py` (+ `tests/integration/test_openfold3_integration.py`). Fact sheet: `.superpowers/sdd/recon/sp-a.md` §3. Read the current `src/autobio/tools/openfold3.py` in full — `_build_query_json` is preserved behaviorally (reads typed fields; keeps dict-insertion iteration order, NOT sorted).

**New class** (append to `structure_prediction.py`):

```python
class OpenFold3Input(BaseInput):
    """Input for OpenFold3 biomolecular structure prediction (single ``predict`` mode)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence (protein/DNA/RNA). For ligand chains "
            "(entity_types = 'ligand'/{'smiles': ...}/{'ccd': ...}) the value is a "
            "SMILES string. May be empty only when query_json is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1, ge=1,
        description="Number of structures to generate (maps to num_diffusion_samples).",
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
            "Per-chain molecule type: 'protein'/'dna'/'rna'/'ligand', or a dict "
            "{'smiles': 'CC...'} / {'ccd': 'ATP'} for ligands. Default 'protein'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    non_canonical_residues: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain non-canonical residues as {chain_id: {position: CCD_code}}, "
            "e.g. {'A': {'3': 'MHO', '5': 'SEP'}}."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=13),
    )
    msa_paths: list[str] | None = Field(
        default=None,
        description="Pre-computed MSA file paths (copied into the workspace).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    query_json: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Raw OpenFold3 query JSON (dict or string) — bypasses automatic query "
            "generation. See https://openfold-3.readthedocs.io/en/latest/input_format_reference.html."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=16),
    )
    use_templates: bool = Field(
        default=True,
        description="Enable template-based prediction.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=17),
    )
    pae_enabled: bool = Field(
        default=True,
        description="Enable the PAE head (produces pTM/ipTM; higher memory).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=18),
    )
```

**`config.json` contract (byte-compat):** `query_json_path` = `"/workspace/inputs/query.json"`; `output_dir` = `"/workspace/outputs/raw"`; `checkpoint_path` = `_CHECKPOINT_PATH`; `use_msa_server` = `input_data.use_msa_server`; `use_templates` = `input_data.use_templates`; `pae_enabled` = `input_data.pae_enabled`; `num_diffusion_samples` = `input_data.num_models`; then `self._apply_extra(config, input_data)` (merges remaining CLI scalars: `num_model_seeds`, `seed`, `output_format`, `low_memory`, `msa_server_url`, `num_devices`).

**Runner transform** (`openfold3.py`):
- Swap imports to catalog; delete `TOOL_REGISTRY["openfold3"]` + `_CONSUMED_EXTRA_KEYS`; add `OPENFOLD3_TOOL` + `register`. Drop `_OPENFOLD3_INPUT_FORMAT` (fold into field descriptions); reuse `_OPENFOLD3_NOTES` on the Mode.
- `prepare_workspace`: `assert isinstance(input_data, OpenFold3Input)`. Query JSON: `if input_data.query_json is not None:` use it (str → verbatim; dict → `json.dumps(..., indent=2)`), else `self._build_query_json(input_data)`. Templates/`msa_paths` staging read typed fields. Build config per contract; end with `self._apply_extra(config, input_data)`.
- `_build_query_json`: preserve exactly, reading `input_data.entity_types` / `input_data.non_canonical_residues` (was `extra.get(...)`); **keep dict-insertion iteration order (NOT sorted)**.
- `_validate_inputs`: `has_query_json = input_data.query_json is not None` (consistent with prepare — resolves the `"null"` footgun); read typed fields; preserve all messages.
- `parse_output`, `_resolve_container_path`: unchanged.

**Tool** (reuse `_OPENFOLD3_NOTES`):

```python
OPENFOLD3_TOOL = Tool(
    name="openfold3",
    display_name="OpenFold3",
    category=ToolCategory.STRUCTURE_PREDICTION,
    description=(
        "Predict biomolecular structures using OpenFold3 (open-source AlphaFold3). "
        "Supports proteins, DNA, RNA, ligands, and non-canonical residues with "
        "MSA-based and template-based prediction."
    ),
    version="1.0.0",
    image_tag="openfold3:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict structure",
            description="Predict a biomolecular complex structure.",
            input_schema=OpenFold3Input,
            output_schema=StructurePredictionOutput,
            default_timeout=3600,
            notes=_OPENFOLD3_NOTES,
        )
    },
    keywords=("openfold3", "alphafold3", "structure prediction", "complex", "ligand"),
)
"""Catalog Tool for OpenFold3."""

register(OPENFOLD3_TOOL)
```

- [ ] **Step 1:** Add `OpenFold3Input` to `structure_prediction.py`; import check.
- [ ] **Step 2:** Migrate `openfold3.py` per the transform + contract, preserving `_build_query_json` behavior (insertion order).
- [ ] **Step 3:** Update `tests/unit/test_openfold3.py`: pass `entity_types`/`non_canonical_residues`/`msa_paths`/`query_json`/`use_msa_server`/`use_templates`/`pae_enabled` as TOP-LEVEL fields; keep CLI scalars in `extra` and assert they flat-merge. Add: catalog registration test; full-dict `config.json` equality test; a generated-`query.json` equality test for a protein+ligand (`entity_types`) + `non_canonical_residues` input; an `info` snapshot (promoted hint + `output_schema` + `notes` present); an extra-shadow-rejection test. Add a footgun-resolution test: `OpenFold3Input(sequences={"A": "MKT"}, query_json=None)` builds `query.json` FROM the sequences (not the literal `"null"`). If `tests/integration/test_openfold3_integration.py` exists, swap its inputs to `OpenFold3Input` (Docker-free verify). Do NOT touch `containers/`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_openfold3.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `openfold3: migrate to catalog Tool; promote structural extra keys to typed fields`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info <tool>` now shows `notes`; chai1/openfold3 are catalog Tools with the `predict` mode + `x-autobio` hints + `output_schema` + notes; both absent from `TOOL_REGISTRY` (disjointness guard green). No `structure_prediction`-category tool remains in `TOOL_REGISTRY`.

## Self-Review

**1. Spec coverage:** notes restored to catalog `info` (Task 1); chai1 + openfold3 migrated to Tools/Modes with typed fields + `x-autobio`; byte-compat config.json + generated input file (equality tests); flat entries removed; `StructurePredictionInput` untouched; `sequences` kept `dict[str,str]`; templates/msa "copied-not-wired" behavior preserved; chai_fasta/query_json footguns resolved via consistent truthiness/`is not None` (documented). RunMetadata.mode still deferred (single-mode).

**2. Placeholder scan:** Full code for the formatter change and both new schema classes + Tool objects; runner transforms specified by exact byte-compat contracts + the in-repo source + the merged boltz exemplar; no "TBD".

**3. Type consistency:** `Chai1Input`/`OpenFold3Input` referenced consistently across schema module, runners, Tool `input_schema`, tests. Output schema reuses `StructurePredictionOutput`. `_apply_extra` unchanged. `self.tool_name` dispatch not needed (single-tool runners).

## Next plans (Plan 4 continued)
- **4d — rfd3 (structure_design):** keep `design_specs` as an escape-hatch dict; adopt `_apply_extra` (rfd3 currently uses unfiltered `config.update(extra)`); single mode.
- **Later:** mpnn family (proteinmpnn + ligandmpnn together — shared `MPNNRunner`); same-category multi-mode (rosetta [+`RunMetadata.mode`], evoef2, complexa); cross-category two-class consolidation (esm_if1, antifold); output-variance (openmm, antibody LMs ×6); then teardown (remove `TOOL_REGISTRY`/`ToolEntry`, `StructurePredictionInput`, hoist duplicated `_resolve_container_path` onto `ToolRunner`) + README rewrite.
