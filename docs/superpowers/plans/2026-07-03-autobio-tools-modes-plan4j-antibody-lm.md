# Antibody LM Tools→Modes Migration (Plan 4j) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Migrate the twelve flat antibody-LM tools (6 models × `{X, X_pll}`) into SIX catalog `Tool`s (one per model: `currab`, `ft_esm`, `balm_paired`, `balm_unpaired`, `ablang2`, `antiberta2`), each exposing `{embedding, pll}` `Mode`s served by the shared `AntibodyLMRunner`, with per-mode output/timeout; and modernize the input to native FASTA via `AntibodySequenceSet` with `x-autobio` UI hints.

**Architecture:** Task 1 breaks an import cycle and modernizes `AntibodyInput` (sequences → `AntibodySequenceSet`, add hints) at the schema layer, leaving the runner untouched. Task 2 does the Tools→Modes migration: 6 Tools via a small factory, runner dispatch on `self.current_mode.name`, promote `per_position` to a typed field + adopt `_apply_extra`, and rewrite the tests. Container `config.json` (and the generated `sequences.json`) are preserved byte-for-byte per mode.

**Tech Stack:** Python 3.11+, Pydantic, pytest. Exemplars: `src/autobio/schemas/embedding.py` (`ESMEmbedInput`: `GenericSequenceSet` + `ui()` hints), `src/autobio/tools/esm.py` (shared-runner catalog Tools), `src/autobio/tools/openmm.py` (per-mode dispatch + byte-compat test shape).

## Global Constraints

- **Byte-compat:** container `config.json` per mode AND the generated `inputs/sequences.json` MUST be byte-identical to pre-branch (keys, values, key ORDER). `containers/` is NOT touched. Guard with full-dict `cfg == expected` + `list(cfg.keys()) == list(expected.keys())` per mode (see `.superpowers/sdd/recon/antibody_lm.md` for the exact key order).
- **Dispatch on `self.current_mode.name`** (mode names are exactly `"embedding"` and `"pll"` — equal to the config `mode` value, so no mapping).
- **Behavior-preserving:** all six Tools are single-category `EMBEDDING` (the `pll` mode stays EMBEDDING, NOT cross-category). Model spec values, validation messages, and outputs unchanged.
- **README is OUT of scope** (deferred to teardown). Do not touch it.
- Env: `python -m pytest` (bare = wrong env); this config prints dots but omits the "N passed" line — verify via exit code.

---

## Task 1: Break the import cycle + modernize `AntibodyInput`

**Files:**
- Create: `src/autobio/schemas/antibody_types.py`
- Modify: `src/autobio/schemas/antibody.py`, `src/autobio/schemas/sequences.py`, `src/autobio/utils/sequences.py`
- Modify: `tests/unit/test_schemas.py` (AntibodyInput tests + a FASTA-input test)

**Interfaces:**
- Produces: `AntibodySequence` importable from BOTH `autobio.schemas.antibody_types` (canonical) and `autobio.schemas.antibody` (re-export); `AntibodyInput.sequences: AntibodySequenceSet` with a SEQUENCE hint; `layer`/`pooling` carry hints.

### Why (import cycle)
`schemas/sequences.py` and `utils/sequences.py` both `from autobio.schemas.antibody import AntibodySequence`. So `schemas/antibody.py` cannot import `AntibodySequenceSet` from `schemas/sequences.py` (cycle). Fix: move `AntibodySequence` to a pydantic-only leaf module.

### Steps
1. **Create `src/autobio/schemas/antibody_types.py`** containing the `AntibodySequence` class (moved verbatim from `antibody.py`, including its `id`/`heavy_chain`/`light_chain` fields and the `_at_least_one_chain` model_validator). Imports: `from __future__ import annotations` + `from pydantic import BaseModel, Field, model_validator`. Nothing else.

2. **`schemas/antibody.py`:** remove the `AntibodySequence` class definition; add:
   ```python
   from autobio.schemas.antibody_types import AntibodySequence  # re-export
   from autobio.schemas.hints import Tier, Widget, ui
   from autobio.schemas.sequences import AntibodySequenceSet  # noqa: TC001 - runtime field type
   ```
   Keep `AntibodySequence` in the module namespace (the import above suffices; add it to `__all__` if the module defines one). Change `AntibodyInput`:
   ```python
   sequences: AntibodySequenceSet = Field(
       description=(
           "One or more antibody sequences: a list of AntibodySequence/dicts, "
           "FASTA text, or a path to a .fasta/.fa file."
       ),
       json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="antibody", tier=Tier.PRIMARY, order=0),
   )
   layer: int | None = Field(
       default=None,
       description=(
           "Model layer from which to extract embeddings. "
           "None uses the final layer. Only used in embedding mode."
       ),
       json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
   )
   pooling: str | None = Field(
       default=None,
       description=(
           "Pooling strategy for per-residue embeddings "
           "('mean', 'cls', 'per_residue'). Only used in embedding mode."
       ),
       json_schema_extra=ui(
           widget=Widget.SELECT, tier=Tier.PRIMARY, order=1,
           enum_labels={"mean": "Mean pool", "cls": "CLS token", "per_residue": "Per-residue"},
       ),
   )
   ```
   Do NOT change the `pooling` TYPE to `Literal` (the runner does model-aware validation and raises `AutobioError`; a Literal would duplicate that and change the error type). Do NOT add `per_position` here (Task 2).

3. **`schemas/sequences.py`:** change `from autobio.schemas.antibody import AntibodySequence` → `from autobio.schemas.antibody_types import AntibodySequence`. Nothing else changes.

4. **`utils/sequences.py`:** change `from autobio.schemas.antibody import AntibodySequence` → `from autobio.schemas.antibody_types import AntibodySequence`.

5. **Verify no cycle:** `python -c "import autobio; from autobio.schemas.antibody import AntibodySequence, AntibodyInput; from autobio.schemas.sequences import AntibodySequenceSet; print('ok')"` prints `ok`.

6. **`tests/unit/test_schemas.py`:** the existing `TestAntibodyInput` should still pass (AntibodySequenceSet accepts `list[AntibodySequence]`); run it. ADD a test proving native FASTA input works, e.g.:
   ```python
   def test_antibody_input_accepts_fasta_text(self) -> None:
       fasta = ">ab1|heavy\nEVQLVESGG\n>ab1|light\nDIQMTQSPS\n"
       inp = AntibodyInput(sequences=fasta)
       assert len(inp.sequences) == 1
       assert inp.sequences[0].id == "ab1"
       assert inp.sequences[0].heavy_chain == "EVQLVESGG"
       assert inp.sequences[0].light_chain == "DIQMTQSPS"
   ```
   (Confirm the exact FASTA pairing format `parse_antibody_fasta_string` expects by reading `utils/sequences.py`; adjust the header format in the test to match.)

7. **Run + commit:**
   ```bash
   python -m pytest tests/unit/test_schemas.py tests/unit/test_sequence_set.py tests/unit/test_antibody_lm.py -q   # antibody_lm still green (runner untouched)
   python -m pytest -m "not docker and not gpu" -q   # exit 0
   ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
   git add -A && git commit -m "schemas: move AntibodySequence to leaf module; AntibodyInput takes AntibodySequenceSet + hints"
   ```
   Expected: no import cycle; antibody_lm runner + its existing tests still pass unchanged; FASTA input works.

---

## Task 2: Migrate the 6 antibody-LM models to catalog Tools with `{embedding, pll}` modes

**Files:**
- Modify: `src/autobio/tools/antibody_lm.py`
- Modify: `src/autobio/schemas/antibody.py` (add `per_position` field)
- Modify: `src/autobio/tools/__init__.py` (TOOL_RUNNERS)
- Modify: `tests/unit/test_antibody_lm.py` (rewrite for catalog)
- Modify: `tests/integration/test_ablang2_integration.py`, `tests/integration/test_antiberta2_integration.py`, `tests/integration/test_currab_integration.py`

**Interfaces:**
- Consumes: `Mode`, `Tool`, `register`, `get_tool` from `autobio.core.catalog`; `AntibodyInput` (now with `per_position`); `_apply_extra` from base.
- Produces: 6 catalog Tools with `{embedding, pll}`; `TOOL_RUNNERS` maps the 6 model names → `AntibodyLMRunner`; the 12 flat names removed.

### Steps

1. **Add `per_position` to `AntibodyInput`** (`schemas/antibody.py`):
   ```python
   per_position: bool = Field(
       default=False,
       description="Return per-position PLL scores (pll mode only). Slower.",
       json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=11),
   )
   ```

2. **`antibody_lm.py` imports:** replace `from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry` with:
   ```python
   from autobio.core.catalog import Mode, Tool, register
   from autobio.core.registry import ToolCategory
   ```
   Keep other imports. Delete the `_CONSUMED_EXTRA_KEYS` constant.

3. **Re-key `_MODEL_CONFIG`** to model names only — drop the six `*_pll` entries (each is an identical `_ModelSpec` to its base). Keep 6 entries: `currab`, `ft_esm`, `balm_paired`, `balm_unpaired`, `ablang2`, `antiberta2` (values unchanged).

4. **Runner dispatch** in `AntibodyLMRunner`:
   - `prepare_workspace`: keep `spec = _MODEL_CONFIG[self.tool_name]`; add `assert self.current_mode is not None`; set `mode = self.current_mode.name`; build config with `"per_position": input_data.per_position` (typed field, was `extra.get(...)`); replace the flat-merge loop with `self._apply_extra(config, input_data)`. Keep the `sequences.json` write and the exact config key order + the `"pooling": input_data.pooling or "mean"` expression.
   - `_is_pll_mode`: `assert self.current_mode is not None; return self.current_mode.name == "pll"`.
   Leave `_parse_embedding_output`, `_parse_pll_output`, `_validate_inputs`, `_resolve_container_path` unchanged.

5. **Register 6 Tools via a factory.** Keep `_ANTIBODY_NOTES`, `_PLL_NOTES`, `_ANTIBODY_INPUT_FORMAT`. Delete the 12 `TOOL_REGISTRY[...] = ToolEntry(...)` blocks and add:
   ```python
   def _register_antibody_lm(
       *,
       name: str,
       display_name: str,
       image_tag: str,
       tool_description: str,
       embed_description: str,
       pll_description: str,
       keywords: tuple[str, ...],
   ) -> None:
       """Register one antibody-LM model as a catalog Tool with embedding + pll modes."""
       register(
           Tool(
               name=name,
               display_name=display_name,
               category=ToolCategory.EMBEDDING,
               description=tool_description,
               version="1.0.0",
               image_tag=image_tag,
               requires_gpu=True,
               gpu_count=1,
               default_mode="embedding",
               modes={
                   "embedding": Mode(
                       name="embedding",
                       display_name="Embed sequences",
                       description=embed_description,
                       input_schema=AntibodyInput,
                       output_schema=EmbeddingOutput,
                       default_timeout=600,
                       supports_batch=True,
                       notes=_ANTIBODY_NOTES,
                   ),
                   "pll": Mode(
                       name="pll",
                       display_name="Pseudo log-likelihood",
                       description=pll_description,
                       input_schema=AntibodyInput,
                       output_schema=AntibodyPLLOutput,
                       default_timeout=1800,
                       supports_batch=True,
                       notes=_ANTIBODY_NOTES + _PLL_NOTES,
                   ),
               },
               keywords=keywords,
               notes=_ANTIBODY_INPUT_FORMAT,
           )
       )
   ```
   Then six `_register_antibody_lm(...)` calls. For each model use: the image_tag from the recon table; `embed_description`/`pll_description` = the EXACT existing ToolEntry description strings for that model's embed/pll tools (copy verbatim from pre-branch `git show 60fce0b:src/autobio/tools/antibody_lm.py`); a `tool_description` one-liner naming the model + that it offers embedding and pll modes; sensible `keywords` (e.g. `("currab", "antibody", "embedding", "pll", "language model")`). Display names: CurrAb, ft-ESM, BALM-paired, BALM-unpaired, AbLang2, AntiBERTa2.

6. **`TOOL_RUNNERS`** (`tools/__init__.py`): remove all 12 antibody flat entries; add the 6 model names → `AntibodyLMRunner`.

7. **Rewrite `tests/unit/test_antibody_lm.py`** (follow `tests/unit/test_openmm.py`/`test_esm_if1.py` shape). Requirements:
   - `_make_runner(tool_name, config, mode="embedding")` builds `AntibodyLMRunner(tool_name, config)` and sets `runner.current_mode = get_tool(tool_name).modes[mode]`. Update the fixtures accordingly (`currab_runner` → embedding; `currab_pll_runner` → `_make_runner("currab", config, "pll")`; etc.). Parametrized tests that used `*_pll` flat names change to `(model, "pll")`.
   - **Byte-compat config tests (full-dict + key-order)** for embedding AND pll modes (at least on currab), including: defaults, `per_position=True` (pll), and a non-consumed extra key flat-merging after the fixed keys. Use the exact key order in the recon.
   - **`_apply_extra` rejection tests:** `extra={"model_name": "x"}` (collides config key) raises `AutobioError` matching "collide"; `extra={"layer": 5}` (shadows typed field) raises; `extra={"per_position": True}` (shadows typed field) raises.
   - **Port ALL existing tests:** host validation (empty sequences, no chains, invalid heavy/light chain, sequence too long, paired↔unpaired model mismatch, invalid/valid layer incl. model-aware, invalid pooling, ambiguous residues, cache paths per model), parse_output (embedding single/multiple, model_name populated, container paths resolved, pll with/without per_position). Keep identical `AutobioError` messages. `per_position` opt-in test now passes it as a field: `AntibodyInput(..., per_position=True)`.
   - **Catalog registration tests:** all 6 tools in CATALOG; each has modes `{embedding, pll}`, `default_mode == "embedding"`, `category == EMBEDDING`, `tool_categories(tool) == (ToolCategory.EMBEDDING,)`, in `list_tools(EMBEDDING)`; per-mode output_schema (Embedding/AntibodyPLL) and default_timeout (600/1800); `image_tag` per model (both modes inherit the Tool image; `Mode.image_tag is None`); `TOOL_RUNNERS[<model>] is AntibodyLMRunner`; the 12 flat names absent from `TOOL_REGISTRY` and `TOOL_RUNNERS`.
   - **`run()` lifecycle** (mock container, like esm_if1) for one model, BOTH modes, asserting `metadata.mode` and output type.
   - **`info` snapshot** via `format_tool_info_catalog(get_tool("currab"), OutputFormat.JSON)` containing both modes + notes.

8. **Update the 3 integration test files** (Docker-gated): change tool invocation from flat names to model name + `mode=` (read each file; they build `AntibodyLMRunner`/call `.run(...)`). e.g. `currab_pll` → model `currab`, `mode="pll"`. Keep input construction and assertions otherwise unchanged. Add `from autobio.core.catalog import get_tool` only if a direct `prepare_workspace` path needs `current_mode` set.

9. **Run + commit:**
   ```bash
   python -m pytest tests/unit/test_antibody_lm.py tests/unit/test_schemas.py tests/unit/test_registry_disjoint.py -q
   python -m pytest -m "not docker and not gpu" -q   # exit 0
   ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/
   git add -A && git commit -m "antibody-lm: migrate 6 models to catalog Tools with embedding/pll modes"
   ```
   Expected: 6 Tools in CATALOG; 12 flat names gone; byte-compat config + sequences.json preserved; disjointness guard passes.

---

## Self-Review checklist (controller, before dispatch)
- [ ] Task 1 leaves the runner + `test_antibody_lm.py` passing unchanged (schema-only change).
- [ ] No import cycle (`python -c "import autobio"`).
- [ ] Task 2 byte-compat: config key order + `sequences.json` preserved; `per_position` from field; `_apply_extra` adopted.
- [ ] All 6 Tools EMBEDDING-only (not cross-category); per-mode output/timeout; image per model.
- [ ] 12 flat names gone from both registries; factory used for the 6 Tools.
- [ ] README untouched.
