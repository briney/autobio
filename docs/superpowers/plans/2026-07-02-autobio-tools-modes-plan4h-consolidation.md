# Tools→Modes Plan 4h — esm_if1 + antifold Two-Class Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Consolidate `esm_if1` and `antifold` — each currently TWO flat tools backed by TWO separate runner classes — into ONE catalog `Tool` each with `{design, score}` modes served by ONE runner. These are the first **cross-category** Tools (design = INVERSE_FOLDING, score = SCORING) and the first with true per-mode **output-schema** variance.

**Architecture:** Merge each tool's two runner classes into one; `prepare_workspace` and `parse_output` dispatch on `self.current_mode.name` (simple if/else, freesasa-style — the two branches are the two old classes' bodies verbatim). Each mode keeps its own `input_schema` AND `output_schema` (design: `InverseFoldingInput`→`InverseFoldingOutput`; score: `ScoringInput`→`ScoringOutput`, reused as-is), its own `default_timeout` (600/300), and — new — its own `category` via `Mode.category` (score overrides the Tool's INVERSE_FOLDING primary with SCORING). Both runners currently do unfiltered `config.update(input_data.extra)` → adopt `self._apply_extra`. Container-side execution is untouched; each mode's `config.json` is byte-for-byte preserved.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; max line length **100**. Ruff `E,W,F,I,UP,B,SIM,TCH`; `ruff format` double quotes. Google docstrings; type hints everywhere.
- Tests: **`python -m pytest`** (bare = wrong env).
- **Scope autobio core only** — NO changes to `containers/`, workspace/`result.json` protocol, `standardize.*`, GPU allocation. The container keys on the `mode` config key (`"design"`/`"score"`) — preserve it exactly.
- **Byte-compat is the success criterion:** each mode's `config.json` (keys AND ORDER) must match the pre-migration flat tool exactly. Ship a full-dict `config.json` equality test **with an explicit `list(cfg.keys()) == list(expected.keys())` key-order assertion** per mode.
- **Reuse the shared schemas as-is** — `InverseFoldingInput`/`InverseFoldingOutput` (design) and `ScoringInput`/`ScoringOutput` (score) are the per-mode schemas. Do NOT create new input classes, do NOT modify these shared schemas, and do NOT add `x-autobio` hints to them in this PR (they're still shared with each other + openmm; hint/dedicated-input decisions are deferred to teardown). Antibody-specific params (antifold's `heavy_chain`/`light_chain`/`antigen_chain`/`regions`) stay in `extra` (their notes already describe them that way — no reword needed).
- **Cross-category:** `Tool.category = ToolCategory.INVERSE_FOLDING` (primary); the `score` Mode sets `category=ToolCategory.SCORING`; the `design` Mode leaves `category=None` (inherits the Tool's). Verify `tool_categories(tool)` returns `(INVERSE_FOLDING, SCORING)` and `list_tools(category=SCORING)` includes the tool.
- Adopt `self._apply_extra(config, input_data)` in place of `config.update(input_data.extra)`. (This makes `extra` shadowing a typed field or a config key — incl. `mode`/`structure_path` — fail fast; harmless for normal use.)
- Drop the legacy `input_format`? (These tools have no `input_format` tuples — only `notes`.) `notes` move onto the per-mode `Mode` (rendered in `info`). Reword only the esm_if1 design note that says "use esm_if1_score" → "use the score mode".
- Delete the flat `TOOL_REGISTRY[...]` blocks (disjointness guard `tests/unit/test_registry_disjoint.py` must pass). `TOOL_RUNNERS`: remove the 2 flat-name entries per tool; add a single `"esm_if1"`/`"antifold"`. Remove the now-merged second runner class (`ESMIF1ScoreRunner`/`AntiFoldScoreRunner`) and its import in `tools/__init__.py`.
- Merged exemplars: `src/autobio/tools/freesasa.py` (two modes, if/else dispatch on `self.current_mode.name`, per-mode input subclasses + `_apply_extra`), `src/autobio/tools/rosetta.py` (byte-compat test shape, `TOOL_RUNNERS` collapse). Recon: `.superpowers/sdd/recon/esm_if1.md`, `.superpowers/sdd/recon/antifold.md`.
- Commit convention `<component>: <what>`. Before each commit: `ruff check --fix`, `ruff format`, `python -m pytest -m "not docker and not gpu"`, `mypy src/` — all clean.

---

### Task 1: Consolidate `esm_if1` into a 2-mode catalog Tool

**Files:** Modify `src/autobio/tools/esm_if1.py`, `src/autobio/tools/__init__.py` (`TOOL_RUNNERS` + import); Test `tests/unit/test_esm_if1.py` (+ integration if present). Read the current `src/autobio/tools/esm_if1.py` in full. Fact sheet: `.superpowers/sdd/recon/esm_if1.md`.

**`config.json` contract per mode (byte-compat — exact order):**
- **design:** `mode` = `"design"`; `structure_path` = `/workspace/inputs/{name}`; `num_sequences` = `input_data.num_sequences`; `temperature` = `input_data.temperature`; then `chains_to_design` = `input_data.chains_to_design` (ONLY if not None); then `fixed_positions` = `input_data.fixed_positions` (ONLY if not None); then `self._apply_extra(config, input_data)`.
- **score:** `mode` = `"score"`; `structure_path` = `/workspace/inputs/{name}`; `sequences` = `input_data.sequences`; then `self._apply_extra(config, input_data)`.

**Runner transform** (`esm_if1.py`):
- Swap imports: keep `ToolCategory`; add `from autobio.core.catalog import Mode, Tool, register`. Keep the InverseFolding + Scoring schema imports (both modes use them).
- Delete both `TOOL_REGISTRY[...]` blocks. **Merge `ESMIF1ScoreRunner` into `ESMIF1Runner`** (delete `ESMIF1ScoreRunner`):
  - `prepare_workspace`: `assert self.current_mode is not None`; `mode = self.current_mode.name`; copy structure; if `mode == "design"`: `assert isinstance(input_data, InverseFoldingInput)` and build the design config; else (`"score"`): `assert isinstance(input_data, ScoringInput)` and build the score config; end with `self._apply_extra(config, input_data)`.
  - `parse_output`: `if self.current_mode.name == "design"`: build `InverseFoldingOutput` (from `data["designed_sequences"]`); else: build `ScoringOutput` (from `data["scores"]`). Return type `InverseFoldingOutput | ScoringOutput` (a valid narrowing of the base's `BaseOutput`; use `BaseOutput` if mypy objects).
- Reword `_ESM_IF1_NOTES` item 3: "Use esm_if1_score to score designed sequences." → "Use the score mode to score designed sequences." Keep `_ESM_IF1_SCORE_NOTES` as-is.

**Tool** (module bottom):

```python
ESM_IF1_TOOL = Tool(
    name="esm_if1",
    display_name="ESM-IF1",
    category=ToolCategory.INVERSE_FOLDING,
    description=(
        "ESM-IF1 (142M) inverse folding: design sequences for a backbone (design mode) "
        "or score sequences against a backbone by conditional log-likelihood (score mode)."
    ),
    version="1.0.0",
    image_tag="esm-if1:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design", display_name="Design sequences",
            description="Design protein sequences for a backbone structure (inverse folding).",
            input_schema=InverseFoldingInput, output_schema=InverseFoldingOutput,
            default_timeout=600, notes=_ESM_IF1_NOTES,
        ),
        "score": Mode(
            name="score", display_name="Score sequences",
            description="Score sequences against a backbone (conditional log-likelihood).",
            input_schema=ScoringInput, output_schema=ScoringOutput,
            default_timeout=300, category=ToolCategory.SCORING, notes=_ESM_IF1_SCORE_NOTES,
        ),
    },
    keywords=("esm-if1", "inverse folding", "sequence design", "scoring", "log-likelihood"),
)
"""Catalog Tool for ESM-IF1 (design + score modes)."""

register(ESM_IF1_TOOL)
```

**`TOOL_RUNNERS`** (`src/autobio/tools/__init__.py`): change the import to `from autobio.tools.esm_if1 import ESMIF1Runner` (drop `ESMIF1ScoreRunner`); remove the `"esm_if1"`/`"esm_if1_score"` entries; add `"esm_if1": ESMIF1Runner`.

- [ ] **Step 1:** Migrate `esm_if1.py` per the transform + per-mode contract (merge the two runner classes); update `TOOL_RUNNERS` + import.
- [ ] **Step 2:** Update `tests/unit/test_esm_if1.py` (+ integration): `_make_runner` constructs `ESMIF1Runner("esm_if1", cfg)` and sets `runner.current_mode = get_tool("esm_if1").modes[mode]` (design/score); catalog registration test (`get_tool("esm_if1")`, modes=={"design","score"}, default_mode "design", both flat names absent from `TOOL_REGISTRY`, `"esm_if1"` in `TOOL_RUNNERS` + both old keys gone, `ESMIF1ScoreRunner` removed); **cross-category test** — `tool_categories(get_tool("esm_if1")) == (INVERSE_FOLDING, SCORING)` and `"esm_if1" in list_tools(category=ToolCategory.SCORING)` and `... in list_tools(category=INVERSE_FOLDING)`; full-dict `config.json` equality + key-order test per mode (design with/without chains_to_design+fixed_positions; score); per-mode `parse_output` tests (design→InverseFoldingOutput, score→ScoringOutput); `info` snapshot (2 modes, per-mode notes + `output_schema` + score mode's `category=="scoring"`); assert `run(...).metadata.mode` per mode; extra-shadow-rejection test (e.g. `num_sequences` via `extra` in design, or `mode`). Integration → `get_runner("esm_if1").run(..., mode="design"|"score")`. Do NOT touch `containers/`.
- [ ] **Step 3:** Run `python -m pytest tests/unit/test_esm_if1.py tests/unit/test_registry_disjoint.py tests/unit/test_catalog.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 4:** ruff + mypy; commit `esm_if1: consolidate design+score into one catalog Tool with 2 modes`.

---

### Task 2: Consolidate `antifold` into a 2-mode catalog Tool

**Files:** Modify `src/autobio/tools/antifold.py`, `src/autobio/tools/__init__.py`; Test `tests/unit/test_antifold.py` (+ integration if present). Read the current `src/autobio/tools/antifold.py` in full. Fact sheet: `.superpowers/sdd/recon/antifold.md`. Structurally symmetric to Task 1, PLUS the `_validate_chain_ids` antibody-chain check in both modes.

**`config.json` contract per mode (byte-compat — exact order):**
- **design:** `mode` = `"design"`; `structure_path`; `num_sequences` = `input_data.num_sequences`; `temperature` = `input_data.temperature`; then `self._apply_extra(config, input_data)`. (Note: antifold design intentionally does NOT map `chains_to_design`/`fixed_positions` — do not add them; antibody params `heavy_chain`/`light_chain`/`antigen_chain`/`regions` flow via `extra` → `_apply_extra`.)
- **score:** `mode` = `"score"`; `structure_path`; `sequences` = `input_data.sequences`; then `self._apply_extra(config, input_data)`.

**Runner transform** (`antifold.py`):
- Swap imports (catalog). Delete both `TOOL_REGISTRY[...]` blocks. Keep `_validate_chain_ids` (module-level helper). **Merge `AntiFoldScoreRunner` into `AntiFoldRunner`** (delete `AntiFoldScoreRunner`):
  - `prepare_workspace`: `assert self.current_mode is not None`; `mode = self.current_mode.name`; `_validate_chain_ids(input_data)` (both modes); copy structure; design → `assert isinstance(input_data, InverseFoldingInput)` + design config; score → `assert isinstance(input_data, ScoringInput)` + score config; end with `self._apply_extra(config, input_data)`.
  - `parse_output`: branch on `self.current_mode.name` (design→InverseFoldingOutput, score→ScoringOutput).
- Keep `_ANTIFOLD_NOTES`/`_ANTIFOLD_SCORE_NOTES` as-is (they correctly describe the antibody params as `extra['...']`, which remain in `extra`).

**Tool** (module bottom):

```python
ANTIFOLD_TOOL = Tool(
    name="antifold",
    display_name="AntiFold",
    category=ToolCategory.INVERSE_FOLDING,
    description=(
        "AntiFold antibody-specific inverse folding (fine-tuned from ESM-IF1): design "
        "antibody sequences for a backbone (design mode) or score sequences by "
        "conditional log-likelihood (score mode). Requires heavy/light chain IDs via extra."
    ),
    version="1.0.0",
    image_tag="antifold:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design", display_name="Design sequences",
            description="Design antibody sequences for a backbone (targeting CDR/FW regions).",
            input_schema=InverseFoldingInput, output_schema=InverseFoldingOutput,
            default_timeout=600, notes=_ANTIFOLD_NOTES,
        ),
        "score": Mode(
            name="score", display_name="Score sequences",
            description="Score antibody sequences against a backbone (log-likelihood/perplexity).",
            input_schema=ScoringInput, output_schema=ScoringOutput,
            default_timeout=300, category=ToolCategory.SCORING, notes=_ANTIFOLD_SCORE_NOTES,
        ),
    },
    keywords=("antifold", "antibody", "inverse folding", "sequence design", "scoring"),
)
"""Catalog Tool for AntiFold (design + score modes)."""

register(ANTIFOLD_TOOL)
```

**`TOOL_RUNNERS`:** change the import to `from autobio.tools.antifold import AntiFoldRunner` (drop `AntiFoldScoreRunner`); remove `"antifold"`/`"antifold_score"`; add `"antifold": AntiFoldRunner`.

- [ ] **Step 1:** Migrate `antifold.py` per the transform + per-mode contract (merge the two runner classes; `_validate_chain_ids` in both); update `TOOL_RUNNERS` + import.
- [ ] **Step 2:** Update `tests/unit/test_antifold.py` (+ integration): per-mode input construction + `current_mode` set; catalog registration test (`get_tool("antifold")`, modes=={"design","score"}, both flat names gone, `AntiFoldScoreRunner` removed); cross-category test (`tool_categories == (INVERSE_FOLDING, SCORING)`, `list_tools(SCORING)` includes antifold); full-dict `config.json` equality + key-order test per mode (design does NOT include chains_to_design/fixed_positions; antibody params from `extra` appear after the fixed keys); per-mode `parse_output` tests; the `_validate_chain_ids` "at least one chain" test preserved (exact message) for BOTH modes; `info` snapshot (2 modes, score `category=="scoring"`); `run(...).metadata.mode`; extra-shadow-rejection test. Integration → `get_runner("antifold").run(..., mode=...)`. Do NOT touch `containers/`.
- [ ] **Step 3:** Run `python -m pytest tests/unit/test_antifold.py tests/unit/test_registry_disjoint.py -v`; then full `-m "not docker and not gpu"`.
- [ ] **Step 4:** ruff + mypy; commit `antifold: consolidate design+score into one catalog Tool with 2 modes`.

---

## Final verification (after all tasks)

- [ ] `ruff check src/ tests/` · `ruff format --check` · `mypy src/` · `python -m pytest -m "not docker and not gpu"` — all green. `autobio info esm_if1`/`autobio info antifold` show 2 modes with per-mode notes + `output_schema`, and `categories` listing both `inverse-folding` and `scoring`; `autobio list --category scoring` surfaces both tools; all 4 flat names (`esm_if1`/`esm_if1_score`/`antifold`/`antifold_score`) gone from `TOOL_REGISTRY` + `TOOL_RUNNERS`; the two `*ScoreRunner` classes are deleted; disjointness guard green.

## Self-Review

**1. Spec coverage:** Two-class consolidation (§5.2) done for esm_if1 + antifold — 2 runner classes each merged into 1 runner with `{design, score}` modes dispatching on `self.current_mode.name`; per-mode input AND output schemas (reused); first cross-category Tools via `Mode.category` override (§2.2 resolution #1); `_apply_extra` adopted; byte-compat config per mode (full-dict + key-order tests); flat entries + second runner classes removed; `TOOL_RUNNERS` collapsed; RunMetadata.mode auto-carries. `InverseFoldingInput`/`ScoringInput` untouched (reused as-is; hint/dedicated-input decision deferred to teardown). antibody params preserved in `extra` with preserved validation.

**2. Placeholder scan:** Full code for both `Tool` objects; runner transforms specified by exact per-mode byte-compat contracts + in-repo source + freesasa/rosetta exemplars; no "TBD".

**3. Type consistency:** Both consolidated runners read `InverseFoldingInput` (design) / `ScoringInput` (score) via isinstance narrowing after mode dispatch; `parse_output` returns `InverseFoldingOutput | ScoringOutput`. Mode `input_schema`/`output_schema`/`category` set per mode. `_apply_extra` unchanged.

## Next plans (Plan 4 continued)
- **Output-variance:** openmm (per-mode image + `SimulationOutput`; still uses `ScoringInput` — migrating it frees `ScoringInput`), antibody LMs ×6 (antibody `SequenceSet`, output variance, shared runner — the last family + first `AntibodySequenceSet` consumer).
- **Teardown:** remove `TOOL_REGISTRY`/`ToolEntry`; remove now-unused category schemas (`StructurePredictionInput`, `StructureDesignInput`, and `InverseFoldingInput` if esm_if1/antifold get dedicated inputs — decide then); hoist duplicated `_resolve_container_path` onto `ToolRunner`; dead-code + copied-but-unwired cleanups; add `x-autobio` hints where deferred; README rewrite with migration notes (mode-based invocation, dropped aliases, extra-shadowing, cross-category tools).
