# Tools→Modes Plan 4e — mpnn Family Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the mpnn family from the flat `TOOL_REGISTRY` to the `Tool`/`Mode` catalog: `proteinmpnn` + `ligandmpnn` (two Tools sharing `MPNNRunner`, inverse-folding), and `ligandmpnn_build_mutant` (`LigandMPNNPackerRunner`, a SCORING sidechain-packing mutation tool).

**Architecture:** Task 1 is a two-Tools-one-runner migration (esm1b/esm2 / boltz1/boltz2 pattern): `proteinmpnn`/`ligandmpnn` differ only by `_MODEL_CONFIG[self.tool_name]` (model_type + checkpoint) and share one new `MPNNInput`. Task 2 is a single SCORING migration (baddg/stabddg pattern) with a new `LigandMPNNPackerInput`. Both runners currently merge `extra` weakly (`MPNNRunner` does an *unfiltered* `config.update(extra)`; the packer uses a hand-rolled `_CONSUMED_EXTRA_KEYS` filter) → both adopt the hardened `self._apply_extra(config, input_data)`. Container-side execution is untouched; each tool's `config.json` is byte-for-byte preserved.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env).
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation.
- **Byte-compat is the success criterion:** each tool's `config.json` from `prepare_workspace` must be identical to pre-migration output for all normal inputs. Ship a full-dict `config.json` equality test per tool.
- **Do NOT modify `InverseFoldingInput`** (`src/autobio/schemas/inverse_folding.py`) — it is also consumed by the unmigrated `esm_if1`/`antifold` tools. Add a NEW `MPNNInput`. **Do NOT modify `ScoringInput`** (`src/autobio/schemas/scoring.py`) — many consumers. Add a NEW `LigandMPNNPackerInput`.
- Adopt `self._apply_extra(config, input_data)` in both runners (replacing `MPNNRunner`'s unfiltered `config.update(input_data.extra)` and the packer's `_CONSUMED_EXTRA_KEYS` filter loop). LigandMPNN/CLI knobs not promoted to typed fields stay in `extra` and flat-merge (byte-compat: only written when passed).
- Catalog `Tool`/`Mode` have no `input_format` field — drop `_*_INPUT_FORMAT`; fold essential guidance into field descriptions. `notes` move onto the `Mode` (rendered by `info` as of Plan 4c).
- Each migration deletes its `TOOL_REGISTRY[...]` block (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS` entries (`proteinmpnn`/`ligandmpnn`/`ligandmpnn_build_mutant`) stay (Tool names == flat names).
- Merged exemplars: `src/autobio/tools/boltz.py` / `src/autobio/tools/esm.py` (two Tools one runner via `self.tool_name`); `src/autobio/tools/baddg.py` / `stabddg.py` (SCORING typed-field + `_apply_extra` + accurate mutation messages).
- Commit convention `<component>: <what>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Migrate `proteinmpnn` + `ligandmpnn` to the catalog

**Files:** Modify `src/autobio/schemas/inverse_folding.py` (add `MPNNInput`), `src/autobio/tools/mpnn.py`; Test `tests/unit/test_mpnn.py` (+ integration if present). Read the current `src/autobio/tools/mpnn.py` in full.

**Schema module prep** (`inverse_folding.py`): add `from autobio.schemas.hints import Tier, Widget, ui`. Do NOT touch `InverseFoldingInput`/`DesignedSequence`/`InverseFoldingOutput`.

**New class** (append):

```python
class MPNNInput(BaseInput):
    """Input for ProteinMPNN / LigandMPNN inverse folding (single ``design`` mode)."""

    structure_path: Path = Field(
        description="Path to the input backbone structure (PDB or mmCIF).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    num_sequences: int = Field(
        default=1,
        description="Number of designed sequences to generate.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.PRIMARY, order=1),
    )
    chains_to_design: list[str] | None = Field(
        default=None,
        description="Chain IDs to redesign. None designs all chains.",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature. Lower values produce more conserved designs.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, step=0.05, order=10),
    )
    fixed_positions: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "Positions to keep fixed (not redesigned), as a mapping of chain ID to "
            "1-based residue indices. Mutually exclusive with chains_to_design "
            "(fixed_positions takes precedence)."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=11),
    )
```

**`config.json` contract (byte-compat):** `model_type` = `_MODEL_CONFIG[self.tool_name]["model_type"]`; `checkpoint_path` = `f"{_CHECKPOINT_DIR}/{_MODEL_CONFIG[self.tool_name]['checkpoint']}"`; `is_legacy_weights` = `True`; `structure_path` = `f"/workspace/inputs/{name}"`; `number_of_batches` = `input_data.num_sequences`; `temperature` = `input_data.temperature`; then `designed_chains` = `",".join(input_data.chains_to_design)` **iff** `chains_to_design is not None`; then if `fixed_positions is not None`: set `fixed_residues` = `",".join(f"{chain}{pos}" ...)` and `config.pop("designed_chains", None)` (mutually exclusive, unchanged logic); then `self._apply_extra(config, input_data)` (replaces `config.update(input_data.extra)`).

**Runner transform** (`mpnn.py`):
- Swap imports: keep `ToolCategory`; add `from autobio.core.catalog import Mode, Tool, register`; import `MPNNInput` from `autobio.schemas.inverse_folding`.
- Delete BOTH `TOOL_REGISTRY[...]` blocks. Keep `_MODEL_CONFIG`, `_CHECKPOINT_DIR`, `_MPNN_NOTES`.
- `prepare_workspace`: `assert isinstance(input_data, MPNNInput)` (was `InverseFoldingInput`); keep the model_cfg lookup, structure copy, config build, and designed_chains/fixed_residues mutual-exclusivity logic verbatim; end with `self._apply_extra(config, input_data)`.
- `parse_output`: unchanged (still returns `InverseFoldingOutput`).

**Tools** (module bottom; define the ligand note as a constant and reuse `_MPNN_NOTES`):

```python
_LIGANDMPNN_LIGAND_NOTE = (
    "For protein-ligand complexes, the foundry parser separates non-polymer "
    "residues (ligands, ions) into synthetic chain IDs. Calcium ions (atom "
    "name 'CA') may be miscounted as protein residues by external PDB parsers, "
    "but the foundry parser handles them correctly."
)

PROTEINMPNN_TOOL = Tool(
    name="proteinmpnn",
    display_name="ProteinMPNN",
    category=ToolCategory.INVERSE_FOLDING,
    description="Design protein sequences for given backbone structures using ProteinMPNN.",
    version="1.0.0",
    image_tag="mpnn:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences for a backbone structure.",
            input_schema=MPNNInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_MPNN_NOTES,
        )
    },
    keywords=("proteinmpnn", "inverse folding", "sequence design", "mpnn"),
)
"""Catalog Tool for ProteinMPNN."""

register(PROTEINMPNN_TOOL)

LIGANDMPNN_TOOL = Tool(
    name="ligandmpnn",
    display_name="LigandMPNN",
    category=ToolCategory.INVERSE_FOLDING,
    description="Design protein sequences with ligand awareness using LigandMPNN.",
    version="1.0.0",
    image_tag="mpnn:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences (ligand-aware) for a backbone structure.",
            input_schema=MPNNInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_MPNN_NOTES + (_LIGANDMPNN_LIGAND_NOTE,),
        )
    },
    keywords=("ligandmpnn", "inverse folding", "sequence design", "ligand", "mpnn"),
)
"""Catalog Tool for LigandMPNN."""

register(LIGANDMPNN_TOOL)
```

- [ ] **Step 1:** Add `MPNNInput` (+ hints import) to `inverse_folding.py`; `python -m pytest tests/unit/test_schemas.py -q`.
- [ ] **Step 2:** Migrate `mpnn.py` per the transform + contract.
- [ ] **Step 3:** Update `tests/unit/test_mpnn.py`: construct `MPNNInput(...)` (structure_path/num_sequences/chains_to_design/temperature/fixed_positions are the same top-level fields); LigandMPNN CLI knobs (`omit`/`bias`/`seed`/`batch_size`/`temperature_per_residue`/`atomize_side_chains`/…) stay in `extra` and still flat-merge (assert). Add catalog registration tests for BOTH proteinmpnn and ligandmpnn (`get_tool(...)`, modes=={"design"}, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`); a `config["model_type"]`/`checkpoint_path` proteinmpnn-vs-ligandmpnn test; a full-dict `config.json` equality test (incl. designed_chains and the fixed_positions→fixed_residues mutual-exclusivity case); an `info` snapshot (a promoted hint + `output_schema` + `notes` present; ligandmpnn notes include the ligand note); an extra-shadow-rejection test (`temperature`/`num_sequences` via `extra`, or a config-key like `designed_chains`, now raises). If `tests/integration/test_mpnn_integration.py` exists, swap inputs to `MPNNInput` (Docker-free: `--collect-only` + host construction snippet). Do NOT touch `containers/`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_mpnn.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `mpnn: migrate proteinmpnn/ligandmpnn to catalog Tools with _apply_extra (design mode)`.

---

### Task 2: Migrate `ligandmpnn_build_mutant` to the catalog

**Files:** Modify `src/autobio/schemas/scoring.py` (add `LigandMPNNPackerInput`), `src/autobio/tools/ligandmpnn_packer.py`; Test `tests/unit/test_ligandmpnn_packer_e2e.py` (+ any unit test / integration if present). Read the current `src/autobio/tools/ligandmpnn_packer.py` in full. `Tier`/`Widget`/`ui` are already imported in `scoring.py` (from Plan 4a).

**New class** (append to `scoring.py`):

```python
class LigandMPNNPackerInput(BaseInput):
    """Input for LigandMPNN sidechain-packing mutant building (single ``build_mutant`` mode)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to introduce, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['EA63Q', 'KB42A']); applied simultaneously."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    num_packs: int = Field(
        default=4, ge=1,
        description="Number of packed structures to produce.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    num_denoising_steps: int = Field(
        default=3, ge=1,
        description="Denoising steps during sidechain packing.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    num_samples: int = Field(
        default=16, ge=1,
        description="Samples drawn per pack.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=12),
    )
    repack_everything: bool = Field(
        default=True,
        description="Repack all sidechains (not only mutated residues).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=13),
    )
    pack_with_ligand_context: bool = Field(
        default=True,
        description="Use bound ligands (HETATM) as context during packing.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=14),
    )
```

**`config.json` contract (byte-compat):** `structure_path` = `f"/workspace/inputs/{name}"`; `mutations` = `input_data.mutations`; `checkpoint_sc` = `_CHECKPOINT_SC`; `checkpoint_bb` = `_CHECKPOINT_BB`; `num_packs` = `input_data.num_packs`; `num_denoising_steps` = `input_data.num_denoising_steps`; `num_samples` = `input_data.num_samples`; `repack_everything` = `input_data.repack_everything`; `pack_with_ligand_context` = `input_data.pack_with_ligand_context`; then `self._apply_extra(config, input_data)` (merges remaining `extra`, e.g. `seed`).

**Runner transform** (`ligandmpnn_packer.py`):
- Swap imports: keep `ToolCategory`; add `from autobio.core.catalog import Mode, Tool, register`; import `LigandMPNNPackerInput` from `autobio.schemas.scoring`. Delete `TOOL_REGISTRY["ligandmpnn_build_mutant"]` + `_CONSUMED_EXTRA_KEYS`. Drop `_INPUT_FORMAT`; reuse `_NOTES` on the Mode. Keep `_CHECKPOINT_SC`/`_CHECKPOINT_BB`/`_DEFAULT_*`/`_MUTATION_RE`/`_MUTATION_FORMAT_HELP`/`_resolve_container_path`.
- `prepare_workspace`: `assert isinstance(input_data, LigandMPNNPackerInput)`; read `mutations = input_data.mutations` and the packing params from typed fields; build config per the contract; end with `self._apply_extra(config, input_data)` (replaces the manual `_CONSUMED_EXTRA_KEYS` loop).
- `_validate_inputs`: retype to `LigandMPNNPackerInput`; keep the structure-exists + PDB-suffix checks; keep the per-mutation `_MUTATION_RE` format check. Change the empty-`mutations` message from `"Tool {tool_name!r} requires 'mutations' in the extra dict. ..."` to an accurate `"LigandMPNN packer requires at least one mutation. {_MUTATION_FORMAT_HELP}"` (mutations is now a typed field). Drop the "must be a list of strings" check (Pydantic enforces `list[str]`).
- `parse_output`: unchanged (returns `ScoringOutput`).

**Tool** (module bottom; reuse `_NOTES`):

```python
LIGANDMPNN_PACKER_TOOL = Tool(
    name="ligandmpnn_build_mutant",
    display_name="LigandMPNN Build Mutant",
    category=ToolCategory.SCORING,
    description=(
        "Build mutant protein structures by introducing amino acid substitutions and "
        "repacking sidechains with LigandMPNN's neural-network sidechain packing model. "
        "Predicts chi angles as mixtures of von Mises distributions, producing full-atom "
        "PDB structures with confidence scores."
    ),
    version="1.0.0",
    image_tag="ligandmpnn-packer:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="build_mutant",
    modes={
        "build_mutant": Mode(
            name="build_mutant",
            display_name="Build mutant",
            description="Introduce mutations and repack sidechains into full-atom structures.",
            input_schema=LigandMPNNPackerInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_NOTES,
        )
    },
    keywords=("ligandmpnn", "mutant", "sidechain packing", "repack", "mutation"),
)
"""Catalog Tool for the LigandMPNN sidechain packer."""

register(LIGANDMPNN_PACKER_TOOL)
```

- [ ] **Step 1:** Add `LigandMPNNPackerInput` to `scoring.py`; import check.
- [ ] **Step 2:** Migrate `ligandmpnn_packer.py` per the transform + contract.
- [ ] **Step 3:** Update the packer's test(s) (`tests/unit/test_ligandmpnn_packer_e2e.py`, and any `test_ligandmpnn_packer.py` if present): pass `mutations`/`num_packs`/`num_denoising_steps`/`num_samples`/`repack_everything`/`pack_with_ligand_context` as TOP-LEVEL `LigandMPNNPackerInput` fields; keep `seed` (and any other knob) in `extra` and assert it flat-merges. Catalog registration test (`get_tool("ligandmpnn_build_mutant")`, modes=={"build_mutant"}, absent from `TOOL_REGISTRY`, present in `TOOL_RUNNERS`); full-dict `config.json` equality test; `info` snapshot (a promoted hint + `output_schema` + `notes`); extra-shadow-rejection test (`mutations`/`num_packs` via `extra` now raises). Update the empty-`mutations` validation test to the new accurate message ("requires at least one mutation"). Keep the PDB-suffix + mutation-format-regex cases. If an integration test exists, swap to `LigandMPNNPackerInput` (Docker-free verify). Do NOT touch `containers/`.
- [ ] **Step 4:** Run `python -m pytest tests/unit/test_ligandmpnn_packer_e2e.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 5:** ruff + mypy; commit `ligandmpnn_build_mutant: migrate to catalog Tool with typed fields (build_mutant mode)`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info <tool>` shows the mode + `x-autobio` hints + `output_schema` + notes for all three; all three absent from `TOOL_REGISTRY` (disjointness guard green). No `inverse-folding` tool other than the still-flat `esm_if1`/`antifold` remains flat.

## Self-Review

**1. Spec coverage:** proteinmpnn/ligandmpnn migrated as two Tools sharing `MPNNRunner` + `MPNNInput` ([[feedback-model-tools]]); ligandmpnn_build_mutant migrated as a SCORING Tool with `LigandMPNNPackerInput`; both runners adopt `_apply_extra`; byte-compat config.json (full-dict tests); flat entries removed; `InverseFoldingInput`/`ScoringInput` untouched (still used by esm_if1/antifold and many scoring tools); typed fields + `x-autobio` hints; `input_format` dropped; notes on the Modes; accurate mutation message. RunMetadata.mode still deferred (all single-mode).

**2. Placeholder scan:** Full code for both new schema classes and all three `Tool` objects; runner transforms specified by exact byte-compat contracts + in-repo source + boltz/baddg exemplars; no "TBD".

**3. Type consistency:** `MPNNInput`/`LigandMPNNPackerInput` referenced consistently across schema modules, runners (`isinstance` + `_validate_inputs`), Tool `input_schema`, tests. Output schemas reuse `InverseFoldingOutput`/`ScoringOutput`. `_apply_extra` unchanged. `self.tool_name` dispatch preserved for proteinmpnn/ligandmpnn.

## Next plans (Plan 4 continued)
- Same-category multi-mode: rosetta (+ `RunMetadata.mode`, carry-forward #4), evoef2, complexa (complexa frees `StructureDesignInput` for teardown).
- Cross-category two-class consolidation: esm_if1, antifold (one Tool with `{design, score}` modes each, one runner — frees `InverseFoldingInput`/`ScoringInput` further).
- Output-variance: openmm, antibody LMs ×6.
- Teardown: remove `TOOL_REGISTRY`/`ToolEntry` + now-unused category input schemas, hoist duplicated `_resolve_container_path` onto `ToolRunner`, wire-or-document copied-but-unwired templates/msa; README rewrite.
