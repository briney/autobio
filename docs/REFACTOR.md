# autobio Refactor — Tools, Modes, and Rich Metadata

**Status:** Design (approved direction; implementation plan to follow in a dedicated session)
**Date:** 2026-07-02
**Author:** design captured with Claude Code
**Scope:** `autobio` core (registry, schemas, CLI, Python API). No changes to container
execution, the workspace protocol, or GPU allocation.

---

## 1. Motivation

`autobio` was built as a toolkit for **autonomous agentic workflows** — hence a flat list of
tool names, JSON-only inputs, and a free-form `extra` pass-through. That design served agents
well, but `autobio` has outgrown it. A well-populated toolkit of biological models is now the
foundation for several *human-facing* and *programmatic* consumers:

- **fold@Scripps** — a multi-user web app where researchers pick a tool and fill in a form.
- A future **HTTP API / MCP server** — where each tool operation is an endpoint / MCP tool.
- Interactive use in **notebooks** (Jupyter / marimo) — where discoverable, typed, well-named
  parameters matter.
- The original **agentic** workflows — which lose nothing here and gain self-describing tools.

Three concrete problems block those consumers today. Each has a direct fix in this refactor:

| Problem | Symptom | Fix (pillar) |
|---|---|---|
| The same model appears as several confusing flat "tools" | `ablang2` and `ablang2_pll` are two entries for one model; ~30 of ~48 names are really `(model/engine) × (task)` | **Tools → Modes** (§4) |
| Parameters are an undifferentiated blob | `BaseInput.extra: dict[str, Any]` hides real, commonly-used options behind free-form JSON | **Rich parameter metadata** (§5) |
| Sequence input is JSON-only | Every consumer must reinvent FASTA handling; researchers think in FASTA | **Native sequence/FASTA input** (§6) |

**Driving principle.** Add to `autobio` *only what its consumers will actually use*. fold's
researcher UX is the concrete forcing function for the contract, but every addition is designed
to serve the API/MCP and notebook consumers equally. There is **no backward-compatibility
requirement** — there are no external users of the current CLI/API contract to preserve, and
agents will re-derive the refactored toolkit at least as easily as the current one. We take a
clean break.

---

## 2. Current state (grounding)

- **Registry (flat).** `core/registry.py` defines `TOOL_REGISTRY: dict[str, ToolEntry]` and a
  closed `ToolCategory(StrEnum)` with six values. `tools/__init__.py` defines
  `TOOL_RUNNERS: dict[str, type[ToolRunner]]`. ~48 tool names across 6 categories.
- **`ToolEntry`** carries: `image_tag`, `category`, `requires_gpu`, `gpu_count`,
  `input_schema: type[BaseInput]`, `output_schema: type[BaseOutput]`, `default_timeout`,
  `supports_batch`, `description`, `version`, `notes`, `input_format`.
- **Schemas are Pydantic v2.** JSON Schema is produced by `model_json_schema()` and surfaced by
  `autobio info <tool> --format json`. `BaseInput` has one field: `extra: dict[str, Any]`.
- **Informal variant patterns already exist** — this refactor makes them first-class:
  - `*_pll` suffix dispatch in `AntibodyLMRunner` (`self.tool_name.endswith("_pll")`).
  - `_VARIANT_CONFIG[self.tool_name]` dicts in `RosettaRunner` (4 variants), `EvoEF2Runner`
    (3), `OpenMMRunner`, `ComplexaRunner`, etc.
  - `_MODEL_CONFIG` data tables and per-name checkpoint selection in `ESMRunner`.
- **CLI (Typer):** `list`, `info`, `run`, `result`, `pull`, `images`. `run` takes
  `--config <json file>`; parameters flow only via that JSON.
- **Python API is importable** — `from autobio.tools import get_runner; runner.run(input_data)`
  runs in-process (no shell). The host package has **no ML dependencies**; `list`/`info` are
  lightweight (registry reads only). Heavy work is isolated in Docker containers.
- **FASTA I/O already exists internally** — `utils/sequences.py::parse_fasta` / `write_fasta`
  (pure Python, no BioPython). Tools call `write_fasta` to stage sequences for containers, but
  FASTA is **not** a user-facing input format.

---

## 3. Design principles

1. **One coherent model/engine = one Tool = one card.** A Tool's distinct *uses* are Modes.
2. **Self-describing over free-form.** Every commonly-used parameter is a typed, documented
   field with a UI hint. `extra` remains only as a power-user/agent escape hatch.
3. **Normalize once, at the boundary.** FASTA parsing, pairing, and validation live in
   `autobio`, so every consumer inherits them. Consumers send FASTA; `autobio` yields structured,
   validated data.
4. **The contract is the JSON Schema.** UI hints ride *inside* the schema via Pydantic
   `json_schema_extra`, so they flow through `model_json_schema()` → `info` with no new
   transport. Consumers that don't understand a hint degrade gracefully.
5. **Don't touch what works.** Container lifecycle, workspace layout, GPU allocation, and the
   result/standardize protocol are unchanged.

---

## 4. Pillar 1 — Tools → Modes

### 4.1 Concepts

- **Tool** — a coherent model or engine; the unit shown as one catalog card. Owns identity,
  display metadata, a primary category, container/resource defaults, and **one or more Modes**.
- **Mode** — a named *use* of a Tool (a task/operation). Owns its own resolved **input schema**,
  **output schema**, timeout, batch support, description, and (optionally) a category that
  overrides the Tool's for taxonomy purposes. A Tool declares a **default mode**.

A Tool with a single mode presents exactly like today (consumers hide the mode selector).

**Decision (Q2): model = Tool, task = Mode.** Each distinct model/engine is its own Tool.
`ablang2`, `antiberta2`, `balm_paired`, `balm_unpaired`, `currab`, `ft_esm` are six separate
antibody-LM Tools (each with `embed` + `pll` modes). `esm1b` and `esm2` are two separate Tools
(each single-mode). `rosetta` is one Tool with four modes. We do **not** introduce a second
"model family" selection axis on top of modes — the mode selector is the single axis.

**Decision (Q1): per-mode schema composed from a shared base.** Each mode's `input_schema` is a
Pydantic model that inherits a **shared tool-level base** (holding fields common to all modes,
e.g. `sequences`) and adds mode-specific fields. `info` returns the fully **resolved** schema per
mode, so a consumer renders exactly the fields relevant to the selected mode — no leakage, and no
conditional show/hide logic in the consumer.

### 4.2 Declaration model (proposed)

Replace the flat, per-name registry entries with a single `Tool` declaration per model/engine
that enumerates its modes. `ToolRunner` stays, but dispatches on **`(tool, mode)`** instead of
parsing `self.tool_name` suffixes — the existing `_VARIANT_CONFIG` / `_MODEL_CONFIG` tables become
explicit mode declarations.

```python
# core/registry.py  (sketch — final shapes decided in the plan)

@dataclass(frozen=True)
class Mode:
    name: str                          # stable id, e.g. "pll"
    display_name: str                  # "Pseudo-log-likelihood"
    description: str
    input_schema: type[BaseInput]      # resolved per-mode model (inherits the tool base)
    output_schema: type[BaseOutput]
    default_timeout: int
    supports_batch: bool = False
    category: ToolCategory | None = None   # overrides Tool.category for taxonomy (see §9)
    notes: tuple[str, ...] = ()

@dataclass(frozen=True)
class Tool:
    name: str                          # stable id / card key, e.g. "ablang2"
    display_name: str                  # "AbLang2"
    category: ToolCategory             # primary category
    description: str
    version: str
    image_tag: str
    requires_gpu: bool
    gpu_count: int
    modes: dict[str, Mode]             # 1+; ordered
    default_mode: str
    keywords: tuple[str, ...] = ()     # for search
    notes: tuple[str, ...] = ()
```

```python
# schemas/antibody.py  (sketch)

class AntibodyBaseInput(BaseInput):
    """Shared across all antibody-LM modes."""
    sequences: SequenceSet[AntibodySequence] = Field(   # SequenceSet: see §6
        description="Antibody sequences (structured JSON, FASTA text, or a FASTA file).",
        json_schema_extra={"x-autobio": {
            "widget": "sequence", "flavor": "antibody", "tier": "primary", "order": 0}},
    )

class AntibodyEmbedInput(AntibodyBaseInput):
    pooling: Literal["mean", "cls", "per_residue"] = Field(
        default="mean",
        json_schema_extra={"x-autobio": {"widget": "select", "tier": "primary", "order": 1}},
    )
    layer: int | None = Field(
        default=None,
        json_schema_extra={"x-autobio": {"widget": "number", "tier": "advanced"}},
    )

class AntibodyPllInput(AntibodyBaseInput):
    per_position: bool = Field(
        default=False,
        json_schema_extra={"x-autobio": {"widget": "toggle", "tier": "primary", "order": 1}},
    )
```

```python
# tools/antibody_lm.py  (sketch)

register(Tool(
    name="ablang2", display_name="AbLang2",
    category=ToolCategory.EMBEDDING,
    description="Antibody language model (45M params, 12 layers, 480-dim). Paired & unpaired.",
    version="1.0.0", image_tag="ablang2:1.0.0",
    requires_gpu=True, gpu_count=1,
    default_mode="embed",
    modes={
        "embed": Mode("embed", "Embeddings", "Extract antibody sequence embeddings.",
                      AntibodyEmbedInput, EmbeddingOutput, default_timeout=600, supports_batch=True),
        "pll":   Mode("pll", "Pseudo-log-likelihood", "Score sequences by pseudo-log-likelihood.",
                      AntibodyPllInput, AntibodyPLLOutput, default_timeout=1800, supports_batch=True),
    },
    keywords=("antibody", "embedding", "likelihood", "ablang"),
))
```

Multi-operation engines follow the same shape — the `_VARIANT_CONFIG` table becomes the `modes`
dict (e.g. `rosetta` → modes `score` / `relax` / `minimize` / `flexddg`, each with its own input
schema derived from a shared Rosetta base).

### 4.3 Runner dispatch

- `TOOL_RUNNERS` maps **tool name → runner class** (one class per Tool, as today for shared
  runners). The runner receives the selected mode and looks up mode config from the `Tool`/`Mode`
  objects rather than string-suffix parsing.
- `get_runner(tool_name, config).run(input_data, *, mode=..., gpu=..., timeout=..., output_dir=...)`
  — `mode` defaults to the tool's `default_mode`.
- The container `config.json` continues to receive a `mode`/variant key exactly as it does now;
  only the *host-side* dispatch changes.

---

## 5. Pillar 2 — Rich parameter metadata (kill the blob)

### 5.1 Promote `extra` keys to typed fields

Every commonly-used key currently smuggled through `extra` (`per_position`, `checkpoint`,
`nstruct`, `score_function`, `temperature`, `chains_to_move`, …) becomes a **real, typed Pydantic
field on the relevant mode's input schema**, with constraints (`ge`/`le`/`Literal`) and a
description. `extra` survives on `BaseInput` **only as an escape hatch** for power users and
agents; it is tagged `tier: "advanced"`, `widget: "json"`, and documented as "promote to a real
field once a key is used commonly."

### 5.2 UI-hint vocabulary

UI hints live under a single namespaced object, `x-autobio`, attached to each property via
Pydantic `Field(json_schema_extra={"x-autobio": {...}})`. Because Pydantic copies
`json_schema_extra` verbatim into `model_json_schema()`, these hints appear in the `info` payload
with **zero new transport**. Standard JSON Schema keys carry the rest (`minimum`/`maximum` from
`ge`/`le`, `enum` from `Literal`, `default`).

| Key | Type | Meaning |
|---|---|---|
| `tier` | `"primary" \| "advanced"` | Surface on the main form vs. under "Advanced". **Orthogonal to required-ness** (a required field can be advanced; an optional field can be primary). Replaces today's crude "required ⇒ primary" heuristic. |
| `widget` | enum | Preferred control: `toggle`, `select`, `slider`, `number`, `text`, `textarea`, `sequence`, `file`. A *hint*; consumers may override by type. |
| `group` | `str` | Logical grouping label for related fields on the form. |
| `order` | `int` | Display order within its tier/group. |
| `unit` | `str` | Unit suffix for numeric fields (e.g. `"Å"`, `"ns"`). |
| `step` | `number` | Step for `slider`/`number`. |
| `enum_labels` | `{value: label}` | Human labels for enum values (so dropdowns show friendly text). |
| `flavor` | `"generic" \| "antibody"` | For `widget: sequence` — selects the sequence input flavor (see §6). |

**Graceful degradation contract.** A consumer that doesn't recognize a hint (or a value) MUST
fall back to type-driven rendering and treat unknown fields as `tier: advanced`. Hints never
change validation — they are presentation only. Validation is always the JSON Schema + Pydantic.

---

## 6. Pillar 3 — Native sequence & FASTA input

### 6.1 Goal

A single sequence-input abstraction that accepts **structured JSON *or* FASTA text *or* a FASTA
file path**, normalizes to `autobio`'s structured representation, and validates — so fold, the
API/MCP, and notebooks all accept FASTA without reinventing a parser.

### 6.2 Two sequence flavors

- **Generic** (`esm*`, structure prediction, scoring, …): today `dict[str, str]` (id → sequence).
- **Antibody** (`ablang2`, `antiberta2`, `balm_*`, `currab`, `ft_esm`): today
  `list[AntibodySequence]` with `id` / `heavy_chain` / `light_chain`.

Both remain the *canonical structured forms*. The refactor adds accepted *input encodings* that
normalize to them.

### 6.3 `SequenceSet` — an accepting input type

Introduce a typed input wrapper (name TBD; `SequenceSet[T]` in sketches) whose validator accepts,
and normalizes to the canonical structured form:

1. **Native structured** — `dict[str, str]` (generic) or `list[AntibodySequence]` (antibody), as
   today. Agents and existing JSON callers are unaffected.
2. **FASTA text** — a raw FASTA string.
3. **FASTA file** — a path to a `.fasta` / `.fa` file (resolved like other `format: "path"`
   inputs; fold uploads the file, the backend stages it, `autobio` reads it).

The field's JSON Schema declares `widget: "sequence"` and `flavor`, so a consumer renders a
FASTA textarea **plus** a file-upload affordance, and (for antibody) can optionally offer
structured rows. Whatever the encoding, `autobio` receives it, parses, and validates centrally.

### 6.4 FASTA parsing rules

- **Generic FASTA:** `>id` → `{id: sequence}`. Duplicate ids are an error (actionable message).
- **Antibody FASTA — header-tagged pairing (Decision Q3):** headers encode a **pair id** and a
  **chain**: `>{pair_id}|{chain}`. Records sharing a `pair_id` are paired into one
  `AntibodySequence`; a lone record becomes an unpaired antibody (that chain only).

  ```
  >ab1|heavy
  QVQLVQSGAEVKKPGASVKVSCKASGYTF...
  >ab1|light
  DIQMTQSPSSLSASVGDRVTITCRASQ...
  >ab2|heavy          # unpaired: no ab2|light record
  EVQLLESGGGLVQPGGSLRLSCAAS...
  ```
  normalizes to:
  ```json
  [{"id": "ab1", "heavy_chain": "QVQ...", "light_chain": "DIQ..."},
   {"id": "ab2", "heavy_chain": "EVQ..."}]
  ```

  - **Chain tokens** are case-insensitive and accept common aliases (`heavy`/`h`/`vh`,
    `light`/`l`/`vl`). Exact accepted set + canonical delimiter (`|`) to be finalized in the plan;
    keep parsing lenient, error messages explicit.
  - **Errors** name the offending record (and ideally line) so consumers can surface precise
    feedback: unknown chain token, duplicate `pair_id|chain`, a `pair_id` with neither chain, or a
    validation failure (non-protein characters).

- **Validation** reuses existing helpers (`utils/sequences`, `validate_protein_sequence`) plus the
  existing `AntibodySequence` "at least one chain" rule.

### 6.5 Why in `autobio`, not fold

fold *could* parse FASTA client-side, but then the API, MCP, and notebook consumers each need
their own parser and their own antibody-pairing convention — guaranteeing drift. Centralizing in
`autobio` means one implementation, one convention, one set of error messages, inherited by all.

---

## 7. Public contracts (what consumers consume)

These JSON shapes are the concrete contract fold's catalog sync and the future API/MCP depend on.
Field names are proposed; finalize in the plan.

### 7.1 `autobio list --format json`

Lists **Tools** (cards), each summarizing its modes.

```json
[
  {
    "name": "ablang2",
    "display_name": "AbLang2",
    "category": "embedding",
    "categories": ["embedding"],          // union of mode categories (for taxonomy/filtering)
    "version": "1.0.0",
    "description": "Antibody language model ...",
    "modes": ["embed", "pll"],
    "requires_gpu": true,
    "gpu_count": 1,
    "keywords": ["antibody", "embedding", "likelihood"]
  }
]
```

### 7.2 `autobio info <tool> --format json`

Returns tool metadata + a `modes` array; each mode carries its **resolved** `input_schema`
(JSON Schema with `x-autobio` hints) and `output_schema`.

```json
{
  "name": "ablang2",
  "display_name": "AbLang2",
  "category": "embedding",
  "version": "1.0.0",
  "image_tag": "ablang2:1.0.0",
  "requires_gpu": true,
  "gpu_count": 1,
  "description": "...",
  "keywords": ["antibody", "embedding", "likelihood"],
  "default_mode": "embed",
  "modes": [
    {
      "name": "embed",
      "display_name": "Embeddings",
      "description": "...",
      "category": "embedding",
      "default_timeout": 600,
      "supports_batch": true,
      "input_schema":  { "type": "object", "properties": { "sequences": { "...": "...", "x-autobio": {"widget": "sequence", "flavor": "antibody", "tier": "primary"} }, "pooling": { "enum": ["mean","cls","per_residue"], "default": "mean", "x-autobio": {"widget": "select", "tier": "primary"} }, "layer": { "type": ["integer","null"], "x-autobio": {"widget": "number", "tier": "advanced"} } }, "required": ["sequences"] },
      "output_schema": { "...": "..." }
    },
    {
      "name": "pll",
      "display_name": "Pseudo-log-likelihood",
      "default_timeout": 1800,
      "supports_batch": true,
      "input_schema":  { "type": "object", "properties": { "sequences": { "...": "...", "x-autobio": {"widget": "sequence", "flavor": "antibody", "tier": "primary"} }, "per_position": { "type": "boolean", "default": false, "x-autobio": {"widget": "toggle", "tier": "primary"} } }, "required": ["sequences"] },
      "output_schema": { "...": "..." }
    }
  ]
}
```

### 7.3 `autobio run`

```
autobio run <tool> --mode <mode> --config <json> [--gpu auto] [--timeout N] [--format json]
```

- `--mode` is optional; defaults to the Tool's `default_mode`. (fold always sends an explicit
  mode.) Old flat names (`ablang2_pll`, `rosetta_relax`, …) are **removed**.
- `--config` JSON matches the selected mode's `input_schema`; `sequences` may be structured JSON,
  a FASTA string, or a FASTA file path (§6.3).

### 7.4 Python API

```python
from autobio import AutobioConfig
from autobio.tools import get_runner
from autobio.core.registry import get_tool

tool = get_tool("ablang2")          # -> Tool with .modes, .default_mode
runner = get_runner("ablang2", AutobioConfig.resolve())
out = runner.run(input_data, mode="pll")     # mode defaults to tool.default_mode
```

`get_tool`/`list_tools` return the new `Tool` objects; `info`-style JSON is available
programmatically for API/MCP servers to re-expose.

---

## 8. Impact on consumers (coordination — non-normative)

Written here so the `autobio` contract is designed to fit; the actual changes are separate specs.

- **fold@Scripps.** Its `Tool` DB row (keyed by `name`+`version`, storing `input_schema`) must
  grow to hold **modes**. Recommended: one `Tool` row per Tool with a `modes` JSONB (each mode's
  resolved schema + metadata) — the submit page gains a **mode selector**, the existing
  schema-driven form renders the selected mode's schema, `x-autobio` hints drive
  primary/advanced placement and widget choice, and sequence fields render FASTA paste + upload.
  `POST /runs` gains a `mode` parameter forwarded to `autobio run --mode`. The future
  **category submenus** in fold's sidebar are backed by the category taxonomy (§9).
- **API / MCP.** `info`'s `modes` map 1:1 to operations/endpoints; `input_schema`/`output_schema`
  give request/response contracts for free. Not built now — but the shape above deliberately does
  not preclude it.
- **Notebooks.** `get_tool` + typed per-mode input models make tools self-documenting via
  autocomplete and `?`.

---

## 9. Category taxonomy & cross-category modes

The closed `ToolCategory(StrEnum)` stays (good for fold's sidebar submenus). Two enhancements:

1. **Enrich the taxonomy** with per-category display metadata (label, description, order, optional
   icon) so consumers render submenus without hardcoding.
2. **Cross-category modes.** Some models legitimately span categories by task —
   `esm_if1` (inverse-folding) vs `esm_if1_score` (scoring); `antifold` vs `antifold_score`;
   `ligandmpnn` vs `ligandmpnn_build_mutant`; `openmm` minimize/relax (scoring) vs simulate
   (simulation). Under "model = Tool", these become **one Tool whose modes carry different
   categories**. **Recommendation:** a Tool has a primary `category`; a `Mode` may declare an
   overriding `category`. `list` exposes `categories` = the union across modes, so a Tool surfaces
   under every relevant sidebar submenu while still having a canonical home. (Confirm in the plan.)

---

## 10. Migration / rollout

Clean break (no back-compat shims). The plan should generate the **authoritative** flat-name →
`(tool, mode)` map by enumerating the live `TOOL_REGISTRY`; the patterns are:

| Family | Flat names today | Becomes |
|---|---|---|
| Antibody LMs (×6 models) | `ablang2`/`ablang2_pll`, `antiberta2`/`…_pll`, `balm_paired`/`…_pll`, `balm_unpaired`/`…_pll`, `currab`/`…_pll`, `ft_esm`/`…_pll` | 6 Tools, each modes `{embed, pll}` |
| Rosetta | `rosetta_score/relax/minimize/flexddg` | 1 Tool, modes `{score, relax, minimize, flexddg}` |
| EvoEF2 | `evoef2_repair/binding/build_mutant` | 1 Tool, modes `{repair, binding, build_mutant}` |
| OpenMM | `openmm_amber_minimize/relax` (scoring), `openmm_md_simulate` (simulation) | 1 Tool, modes `{minimize, relax, simulate}` — cross-category (§9) |
| ComplexA | `complexa/complexa_ligand/complexa_ame` | 1 Tool, 3 modes |
| FreeSASA | `freesasa_bsa/freesasa_sasa` | 1 Tool, modes `{bsa, sasa}` |
| ESM-IF1 | `esm_if1` (inverse-folding), `esm_if1_score` (scoring) | 1 Tool, modes `{design, score}` — cross-category |
| AntiFold | `antifold` (inverse-folding), `antifold_score` (scoring) | 1 Tool, modes `{design, score}` — cross-category |
| LigandMPNN | `ligandmpnn` (inverse-folding), `ligandmpnn_build_mutant` (scoring) | 1 Tool, modes `{design, build_mutant}` — cross-category |
| ESM (single-mode each) | `esm1b`, `esm2` | 2 Tools, 1 mode each |
| Boltz (single-mode each) | `boltz1`, `boltz2` | 2 Tools, 1 mode each |
| Other singletons | `esmfold`, `chai1`, `openfold3`, `proteinmpnn`, `rfd3`, `antipasti`, `baddg`, `stabddg`, `prodigy` | 1 Tool, 1 mode each |

> The exact tool set/counts must be regenerated from the registry during planning — treat this
> table as the *pattern*, not the final inventory.

fold has no production users yet (v1 PRs just merged), so a clean break in tool naming is
acceptable; historical `Run` rows reference tools by `(name, version)` and can be handled in
fold's own migration spec.

---

## 11. Open questions (confirm during planning)

1. **Cross-category modes** (§9): Tool primary category + per-mode `category` override, with
   `list.categories` = union. Confirm this is the taxonomy model.
2. **`x-autobio` namespace/key names** — exact vocabulary spelling and whether to nest under a
   different key.
3. **`output_schema` in `info`** — recommended **yes** (helps result rendering + API/MCP). Confirm.
4. **`SequenceSet` shape** — the concrete typed representation for "structured | FASTA text | FASTA
   file", and whether to add a convenience two-file paired-antibody input later.
5. **FASTA header grammar** — canonical delimiter (`|`) and the accepted chain-token alias set.
6. **`supports_batch` granularity** — per-mode (recommended) vs per-tool.
7. **Versioning** — tool-level (recommended; modes share weights/image) vs mode-level.
8. **Search** — surface `keywords` in `list`; whether fold search indexes them.

---

## 12. Success criteria

- `ablang2` appears **once**; `autobio info ablang2` returns two modes (`embed`, `pll`) each with a
  resolved input schema. No `*_pll` (or other variant-suffix) flat names remain.
- `run` is addressed by `--mode`; every former flat tool is reachable as a `(tool, mode)`.
- Commonly-used parameters are typed fields carrying `tier`/`widget` hints; `extra` is an
  escape hatch only.
- A sequence field accepts structured JSON, FASTA text, and a FASTA file; antibody pairing uses
  the header convention; validation errors identify the offending record.
- `list`/`info`/`run` JSON contracts (§7) are stable, documented, and covered by tests
  (including a real `info` snapshot per representative tool: single-mode, multi-mode same-category,
  cross-category, antibody).
- The Python API exposes `Tool`/`Mode` and `run(..., mode=...)`.

---

## 13. Non-goals

- No changes to container execution, the workspace protocol, `result.json`/standardization, or
  GPU allocation.
- Not building the HTTP API / MCP server here (only ensuring the contract doesn't preclude it).
- Not implementing fold@Scripps changes here (separate spec, driven by this contract).
