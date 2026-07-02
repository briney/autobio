# autobio Tools→Modes Refactor — Implementation-Ready Design

**Status:** Approved (ready for implementation planning)
**Date:** 2026-07-02
**Supersedes/extends:** `docs/REFACTOR.md` (approved direction). This spec records the
*resolved* open questions, the grounded implementation strategy, and the phased plan structure
that `docs/REFACTOR.md` deferred to "a dedicated session."
**Scope:** `autobio` core only — registry, schemas, CLI, Python API, tests, README. **No** changes
to container execution, the workspace/`result.json` protocol, `standardize.*`, or GPU allocation.

---

## 1. Purpose

`autobio` grew as a flat catalog of ~48 tool names with JSON-only inputs and a free-form `extra`
blob — good for autonomous agents, awkward for the human-facing and programmatic consumers now
being built on top of it (fold@Scripps web app, a future HTTP API / MCP server, notebook use).
`docs/REFACTOR.md` sets the direction across three pillars — **Tools→Modes**, **rich parameter
metadata**, **native sequence/FASTA input** — and resolves the top-level design questions
(model = Tool, task = Mode; per-mode schema from a shared base; header-tagged antibody pairing).

This document turns that direction into an implementation-ready plan: it locks the remaining open
questions, records the concrete mechanics discovered in the current code, and defines the phased
sequence of work.

**Driving principle (unchanged):** add only what consumers will actually use; clean break, no
back-compat shims (no external users of the current CLI/API contract exist).

---

## 2. Decisions locked

### 2.1 Structural (from `docs/REFACTOR.md`, confirmed)

- **Tool = one coherent model/engine = one catalog card.** A Tool's distinct *uses* are **Modes**.
- **model = Tool, task = Mode.** Each model/engine is its own Tool. The mode selector is the single
  axis; there is no second "model family" axis.
- **Per-mode input schema composed from a shared tool-level base.** `info` returns the fully
  resolved schema per mode.
- **Container-side execution is untouched.** The host keeps writing each container's *existing*
  config key; `Mode.name` is host-side only (see §5.3).

### 2.2 Resolved open questions (§11 of `docs/REFACTOR.md`)

| # | Question | Resolution |
|---|----------|------------|
| 1 | Cross-category modes | Tool has a primary `category`; a `Mode` may declare an overriding `category`. `list.categories` = union across modes. |
| 2 | UI-hint namespace | Key is `x-autobio` (standard JSON Schema `x-` extension convention), one namespaced object per property. |
| 3 | `output_schema` in `info` | **Yes** — emit both input and output schema per mode. |
| 4 | `SequenceSet` shape | `Annotated[CanonicalType, BeforeValidator(normalize)]` per flavor (see §6). Canonical types unchanged; validator additionally accepts FASTA text / FASTA file. |
| 5 | FASTA header grammar | Antibody: `>{pair_id}\|{chain}`, delimiter `\|`, chain tokens case-insensitive with aliases (`heavy`/`h`/`vh`, `light`/`l`/`vl`). Generic: `>id`. |
| 6 | `supports_batch` granularity | **Per-mode** (a field on `Mode`). |
| 7 | Versioning | **Tool-level** (`Tool.version`); modes share the image/weights. |
| 8 | `keywords` | Add `Tool.keywords`; surface in `list --format json`. |

### 2.3 Plan structure & scope (this session)

- **Plan structure:** one coherent spec, executed in **sequenced phases** with review checkpoints
  (§9). Rich metadata and FASTA both attach to the per-mode input schemas that Tools→Modes creates,
  so the schemas must exist first and metadata+FASTA are applied *while migrating each tool family*
  (touch each schema once) rather than as separate global sweeps.
- **Scope:** autobio core + its tests + README. **Out of scope** (per §13 of `docs/REFACTOR.md`):
  container execution/protocol/GPU allocation, fold@Scripps changes (separate spec), and the
  HTTP API / MCP server (only ensure the contract doesn't preclude it).

---

## 3. Current-state grounding (what the code actually does)

Verified against the codebase; the plan depends on these facts.

### 3.1 Two parallel flat maps, no runner reference in the registry

- `TOOL_REGISTRY: dict[str, ToolEntry]` (`core/registry.py`). `ToolEntry` holds `image_tag`,
  `category`, `requires_gpu`, `gpu_count`, `input_schema`, `output_schema`, `default_timeout`,
  `supports_batch`, `description`, `version`, `notes`, `input_format`. **It does not store the
  runner class.**
- `TOOL_RUNNERS: dict[str, type[ToolRunner]]` (`tools/__init__.py`) maps flat name → runner class.
  `get_runner(name, config)` instantiates `runner_cls(name, config)`.
- `ToolRunner.__init__` stores `self.tool_name` and `self.entry = TOOL_REGISTRY[tool_name]`. **All
  variant dispatch happens inside `prepare_workspace`/`parse_output` by reading `self.tool_name`.**

### 3.2 Runner dispatch patterns (the tables that become `modes`)

| Pattern | Runners | Mechanism |
|---------|---------|-----------|
| **A — one class + module-level dict keyed on `self.tool_name`** | `RosettaRunner` (`_VARIANT_CONFIG`, 4), `EvoEF2Runner` (`_VARIANT_CONFIG`, 3), `ComplexaRunner` (`_VARIANT_CONFIG`, 3, incl. per-variant `ckpt_name`), `OpenMMRunner` (`_VARIANT_CONFIG`, 3), `AntibodyLMRunner` (`_MODEL_CONFIG`, 12 keys w/ pairs duplicated + `_is_pll_mode()` suffix check), `ESMRunner` (`_ESM1B_CONFIG`/`_ESM2_CHECKPOINTS` + `self.tool_name == "esm1b"` and `extra["checkpoint"]`) | `cfg = _VARIANT_CONFIG[self.tool_name]` |
| **B — one class + plain string equality** | `FreeSASARunner` (`self.tool_name == "freesasa_bsa"`, 2) | `is_bsa = self.tool_name == "freesasa_bsa"` |
| **C — separate runner class per flat name** | ESM-IF1 (`ESMIF1Runner` + `ESMIF1ScoreRunner`), AntiFold (`AntiFoldRunner` + `AntiFoldScoreRunner`) | no in-class dispatch; behavior hard-coded per class; **schema AND category differ per class** |

Schema variance across modes exists only in **OpenMM** (input+output), **ESM-IF1/AntiFold**
(input+output, via separate classes), and **AntibodyLM** (output only). All other multi-name
runners share one input+output schema across variants.

### 3.3 The container-facing "mode" key has no common name

Today the host writes different keys per tool: `mode` (antibody_lm, freesasa, esm_if1, antifold),
`protocol` (rosetta, openmm), `binary` (rosetta), `command` (evoef2), `variant`+`ckpt_name`
(complexa), and **nothing** for ESM (checkpoint folds into `model_name`). Because containers are a
non-goal, the runner **keeps writing exactly these keys**; the refactor only changes how the runner
*chooses* which values to write (from a `Mode` object instead of a tool-name string).

### 3.4 `extra` handling and the double-write hazard

`extra: dict[str, Any]` on `BaseInput` is the universal passthrough. Each runner defines
`_CONSUMED_EXTRA_KEYS` and flat-merges the *remaining* extra keys into `config.json`. **Promoting an
`extra` key to a typed field requires adding it to `_CONSUMED_EXTRA_KEYS`**, or it will be written
to `config.json` twice (once as the typed field, once from the merge). The two-class runners
(esm_if1, antifold) currently use unfiltered `config.update(input_data.extra)` — consolidation must
introduce consumed-key filtering.

### 3.5 CLI is registry-reads-only

No ML libraries imported anywhere in `src/autobio/`; importing the CLI just populates the registry
dicts (heavy work is in containers). `info` JSON is produced by a **single** call
`entry.input_schema.model_json_schema()` in `cli/formatters.py`; only the *input* schema is emitted
today. `run` validates the whole `--config` JSON via `runner.entry.input_schema.model_validate(...)`.
`list --format json` emits five keys per tool (`name`, `category`, `gpu`, `version`, `description`).
Nothing uses `json_schema_extra` yet — adding it is purely additive.

### 3.6 Sequence representation and FASTA I/O

- Antibody tools use structured `list[AntibodySequence]` (`id` / `heavy_chain` / `light_chain`, with
  an "at least one chain" validator). Everything else uses untyped `dict[str, str]`
  (`EmbeddingInput`, `StructurePredictionInput`, `ScoringInput`, `SimulationInput`).
- `utils/sequences.py` (pure Python, no deps): `parse_fasta(path: Path) -> dict[str, str]`,
  `write_fasta`, `validate_protein_sequence`, `validate_antibody_sequence`,
  `validate_nucleotide_sequence`. **`parse_fasta` currently takes a file path, not a string** — the
  refactor adds string parsing + antibody pairing.
- Only `esm` and `esmfold` stage sequences via `write_fasta`; antibody runners serialize
  `sequences.json`; structure runners `shutil.copy2` files.

### 3.7 No live consumer of flat names beyond docs

`ExperimentScope`/credit-budget system (`docs/CREDIT_SYSTEM.md`) is a future design, **not yet in
code** — no runtime coupling to migrate. `README.md` references flat names extensively (doc update,
not code).

---

## 4. Target data model

```python
# core/registry.py (final shapes decided in the implementation plan)

@dataclass(frozen=True)
class Mode:
    name: str                              # stable id, e.g. "pll", "score"
    display_name: str
    description: str
    input_schema: type[BaseInput]          # resolved per-mode model (inherits the tool base)
    output_schema: type[BaseOutput]
    default_timeout: int
    supports_batch: bool = False
    category: ToolCategory | None = None    # overrides Tool.category for taxonomy (§8)
    notes: tuple[str, ...] = ()

@dataclass(frozen=True)
class Tool:
    name: str                              # stable id / card key, e.g. "ablang2"
    display_name: str
    category: ToolCategory                 # primary category
    description: str
    version: str
    image_tag: str
    requires_gpu: bool
    gpu_count: int
    modes: dict[str, Mode]                 # 1+; insertion-ordered
    default_mode: str
    keywords: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
```

- **Registry:** `TOOL_REGISTRY: dict[str, Tool]`. `get_tool(name) -> Tool`,
  `list_tools(category=None) -> dict[str, Tool]` (filter by `category` matches a Tool whose primary
  *or* any mode category equals the filter — so cross-category Tools appear under each submenu).
- **Runner mapping:** `TOOL_RUNNERS: dict[str, type[ToolRunner]]` keyed by **Tool name** (one class
  per Tool). `get_runner(tool_name, config)` unchanged in signature.
- A single-mode Tool presents exactly like today (consumers may hide the mode selector).

### 4.1 Category taxonomy enrichment

`ToolCategory(StrEnum)` stays closed (6 values). Add a companion table of per-category display
metadata (`label`, `description`, `order`, optional `icon`) so consumers render sidebar submenus
without hardcoding. Exposed via a small `list_categories()`-style accessor (shape finalized in plan).

---

## 5. Runner changes

### 5.1 Dispatch on Mode, not tool-name string

`ToolRunner.run(...)` gains a `mode: str | None = None` parameter (defaults to
`tool.default_mode`). The base class resolves the `Mode` object and hands it to the subclass hooks
(exact mechanism — e.g. `self.mode: Mode` set before `prepare_workspace`, or passed as an argument —
finalized in the plan). Subclasses read mode config from the `Mode`/`Tool` objects instead of
parsing `self.tool_name`.

- Pattern-A tables (`_VARIANT_CONFIG`, `_MODEL_CONFIG`) become the source data for the `modes` dict.
  The per-mode container-config values (`binary`/`protocol`/`command`/`variant`/`ckpt_name`, model
  name/family/cache path, checkpoint→`model_name` resolution) move onto the mode declaration or a
  small runner-local mode-config table keyed by `Mode.name`.
- Pattern-B (`FreeSASARunner`) dispatches on `Mode.name` (`"bsa"`/`"sasa"`).

### 5.2 Consolidate two-class families

`esm_if1`/`esm_if1_score` → **one** Tool `esm_if1` with modes `{design, score}`, served by **one**
`ESMIF1Runner` that dispatches on `Mode.name`. Same for `antifold` → modes `{design, score}` via one
`AntiFoldRunner`. This introduces `_CONSUMED_EXTRA_KEYS` filtering where the two-class runners
previously did unfiltered `config.update(extra)`.

### 5.3 Preserve container config keys

For each mode, the runner writes the **same** `config.json` keys the corresponding container expects
today (§3.3). No `standardize.*`, `run.sh`, or Dockerfile changes. The container `config.json`
continues to receive a mode/variant/protocol/command key exactly as it does now.

---

## 6. Sequence & FASTA input

### 6.1 `SequenceSet`

A reusable annotated type per flavor whose field type stays the **canonical structured form** so
agents and existing JSON callers are unaffected:

- Generic: `Annotated[dict[str, str], BeforeValidator(normalize_generic_sequences)]`
- Antibody: `Annotated[list[AntibodySequence], BeforeValidator(normalize_antibody_sequences)]`

The `BeforeValidator` accepts and normalizes:

1. **Native structured** — `dict[str, str]` / `list[AntibodySequence]` (or list of dicts) — unchanged.
2. **FASTA text** — a raw string. Disambiguation: a string that starts with `>` **or** contains a
   newline is treated as FASTA text.
3. **FASTA file** — a string path ending in `.fasta`/`.fa` (resolved like other `format: "path"`
   inputs) is read from disk.

Ambiguity/validation errors identify the offending record (and line where feasible). The field's
JSON Schema declares `x-autobio: {widget: "sequence", flavor: "generic"|"antibody"}` so consumers
render a FASTA textarea + file-upload affordance; the canonical structured type remains the
declared type. (Whether to additionally advertise the accepted encodings via `anyOf` in the schema
is a plan-time detail; the validator is the source of truth for acceptance.)

### 6.2 FASTA parsing (extends `utils/sequences.py`)

- **Generic:** `>id` → `{id: sequence}`; duplicate ids are an actionable error.
- **Antibody:** header `>{pair_id}|{chain}`. Records sharing a `pair_id` pair into one
  `AntibodySequence`; a lone record → unpaired (that chain only). Chain tokens case-insensitive with
  aliases (`heavy`/`h`/`vh`, `light`/`l`/`vl`). Errors: unknown chain token, duplicate
  `pair_id|chain`, a `pair_id` with neither chain, non-protein characters.
- Add a **string-accepting** parse path (today `parse_fasta` only reads a file). Reuse
  `validate_protein_sequence` / `validate_antibody_sequence` and the existing `AntibodySequence`
  "at least one chain" rule.

Centralizing here means one parser, one antibody-pairing convention, one set of error messages
inherited by fold, API/MCP, and notebooks.

---

## 7. Rich parameter metadata

### 7.1 Promote `extra` keys

Every commonly-used key currently smuggled through `extra` (`per_position`, `checkpoint`, `nstruct`,
`score_function`, `temperature`, `chains_to_move`, `layer`, `pooling`, `n_steps`, …) becomes a
**typed field** on the relevant mode's input schema with constraints (`ge`/`le`/`Literal`) and a
description, **and is added to the runner's `_CONSUMED_EXTRA_KEYS`** (§3.4). `extra` remains on
`BaseInput` as the escape hatch, tagged `tier: "advanced"`, `widget: "json"`.

The exact key inventory per tool is enumerated from the current `_VARIANT_CONFIG`/`_MODEL_CONFIG`
tables, `_CONSUMED_EXTRA_KEYS` sets, and docstring-documented extra keys (notably `simulation.py`
documents ~20) during planning.

### 7.2 `x-autobio` hint vocabulary

Attached per property via `Field(json_schema_extra={"x-autobio": {...}})`; copied verbatim into
`model_json_schema()`.

| Key | Type | Meaning |
|-----|------|---------|
| `tier` | `"primary" \| "advanced"` | Main form vs. "Advanced". Orthogonal to required-ness. |
| `widget` | enum | `toggle`, `select`, `slider`, `number`, `text`, `textarea`, `sequence`, `file`. Hint only. |
| `group` | `str` | Logical grouping label. |
| `order` | `int` | Display order within tier/group. |
| `unit` | `str` | Unit suffix (`"Å"`, `"ns"`). |
| `step` | number | Step for slider/number. |
| `enum_labels` | `{value: label}` | Friendly labels for enum values. |
| `flavor` | `"generic" \| "antibody"` | For `widget: sequence`. |

**Graceful degradation:** a consumer that doesn't recognize a hint falls back to type-driven
rendering and treats unknown fields as `tier: advanced`. Hints never affect validation — validation
is always JSON Schema + Pydantic.

---

## 8. Public contracts

Concrete JSON shapes per `docs/REFACTOR.md` §7 (field names finalized in the plan).

- **`autobio list --format json`** — array of Tools (cards): `name`, `display_name`, `category`,
  `categories` (union across modes), `version`, `description`, `modes` (name list), `requires_gpu`,
  `gpu_count`, `keywords`.
- **`autobio info <tool> --format json`** — Tool metadata + `default_mode` + a `modes` array; each
  mode carries `name`, `display_name`, `description`, `category`, `default_timeout`,
  `supports_batch`, and its **resolved** `input_schema` (JSON Schema with `x-autobio` hints) **and**
  `output_schema`.
- **`autobio run <tool> --mode <mode> --config <json> [--gpu auto] [--timeout N] [--format json]`**
  — `--mode` optional, defaults to `default_mode`; old flat names removed. `--config` matches the
  selected mode's input schema; `sequences` may be structured JSON, FASTA text, or a FASTA file path.
- **Python API** — `get_tool("ablang2")` returns a `Tool` with `.modes`/`.default_mode`;
  `get_runner("ablang2", cfg).run(input_data, mode="pll")` (mode defaults to `default_mode`).
  `Tool`/`Mode` are importable; `info`-style JSON is available programmatically for a future API/MCP.

---

## 9. Phased implementation plan (skeleton)

One spec, executed in sequenced phases with review checkpoints. Detailed steps produced by the
writing-plans skill.

- **Phase 0 — Foundation (no tool migrated).**
  `Tool`/`Mode` dataclasses + new registry/lookup (`get_tool`/`list_tools`/category accessor);
  `x-autobio` hint helper/constants; `SequenceSet` types + string-FASTA parsing + antibody pairing
  in `utils/sequences.py`. Unit-tested in isolation. Rewrite `test_registry.py` for the new model.

- **Phase 1 — Migrate tool families (bulk; repeated per family).**
  Per family: declare the `Tool` + `modes`; split the input schema into a shared base + per-mode
  subclasses; promote `extra` keys to typed fields with `x-autobio` hints (and update
  `_CONSUMED_EXTRA_KEYS`); swap sequence fields to `SequenceSet`; rewire runner dispatch to
  `Mode`; consolidate two-class runners (esm_if1, antifold); migrate that family's unit/dispatch
  tests. Suggested order (simple → complex): singletons (esmfold, chai1, boltz1/2, proteinmpnn,
  rfd3, antipasti, baddg, stabddg, prodigy, openfold3, esm1b, esm2) → same-category multi-mode
  (rosetta, evoef2, freesasa, complexa) → output-varying (antibody_lm ×6) → cross-category
  (openmm, esm_if1, antifold, ligandmpnn). **Authoritative flat→`(tool, mode)` map regenerated
  from the live `TOOL_REGISTRY`** at the start of this phase (the §10 table in `docs/REFACTOR.md`
  is the pattern, not the inventory).

- **Phase 2 — Contracts/CLI.**
  `list`/`info`/`run --mode` JSON (§8), `output_schema` in `info`, category taxonomy metadata,
  `Tool`/`Mode` in the Python API. Add **real `info` snapshot tests** for four representative tools:
  single-mode, multi-mode same-category, cross-category, antibody. Rewrite `test_cli.py`,
  `test_formatters.py`.

- **Phase 3 — Docs/cleanup.**
  Rewrite `README.md` to Tools/Modes; remove all flat names; final full-suite pass
  (`ruff check --fix`, `ruff format`, `pytest -m "not docker and not gpu"`, `mypy src/`).

---

## 10. Test strategy

- **Unit tests** mock Docker/GPU (patch `autobio.tools.base.ContainerManager` /
  `GPUManager`), as today. Dispatch tests that currently parametrize over flat names
  (`test_antibody_lm.py`, `test_rosetta.py`) are rewritten to parametrize over `(tool, mode)` and
  assert the resulting `config.json` keys are **unchanged** from today (proving container
  compatibility).
- **Fragile spots to migrate deliberately:** `test_registry.py` (registry data model),
  `test_cli.py`, `test_formatters.py` (asserts `input_schema.properties.sequences`), and the
  parametrized dispatch tests.
- **New snapshot tests** capture the real `info --format json` payload per representative tool so the
  `list`/`info`/`run` contract (§8) is regression-locked.
- **New unit tests** for `SequenceSet` normalization: structured / FASTA-text / FASTA-file inputs,
  antibody pairing (paired, unpaired, aliases), and every documented error case.
- **Integration tests** (real containers, Docker/GPU) are updated for `--mode` addressing but are
  not required to run in the implementation environment; container behavior is unchanged.

---

## 11. Success criteria

- `ablang2` appears **once**; `autobio info ablang2` returns two modes (`embed`, `pll`), each with a
  resolved input **and** output schema. No `*_pll` or other variant-suffix flat names remain.
- Every former flat tool is reachable as a `(tool, mode)`; `run` is addressed by `--mode`.
- Commonly-used parameters are typed fields carrying `tier`/`widget` hints; `extra` is an escape
  hatch only; no `extra` key is double-written to `config.json`.
- A sequence field accepts structured JSON, FASTA text, and a FASTA file; antibody pairing uses the
  `>{pair_id}|{chain}` convention; validation errors identify the offending record.
- Container `config.json` for every `(tool, mode)` is byte-for-byte compatible with the current flat
  tool's output (verified by rewritten dispatch tests).
- `list`/`info`/`run` JSON contracts (§8) are stable, documented, and covered by snapshot tests.
- The Python API exposes `Tool`/`Mode` and `run(..., mode=...)`.
- README reflects Tools/Modes; full non-docker/gpu suite, ruff, and mypy pass.

---

## 12. Non-goals

- No changes to container execution, the workspace protocol, `result.json`/standardization, or GPU
  allocation.
- Not building the HTTP API / MCP server (only ensuring the contract doesn't preclude it).
- Not implementing fold@Scripps changes (separate spec, driven by this contract).
- Not implementing the `ExperimentScope`/credit-budget system (`docs/CREDIT_SYSTEM.md` remains a
  future design; when built it will key on `(tool, mode)`).
